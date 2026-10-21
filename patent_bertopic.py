#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BERTopic-based patent topic modeling + weak-signal analysis.

Input Excel default columns:
  publication_number, application_year, abstract

Outputs:
  patent_bertopic_results.xlsx
  weak_signal_map.png
  topic_model/ (saved BERTopic model)

Example:
  python patent_bertopic.py --input patents.xlsx --text-column abstract --year-column application_year --id-column publication_number
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def parse_args():
    p = argparse.ArgumentParser(description="Patent BERTopic + weak-signal analysis")
    p.add_argument("--input", required=True, help="Input Excel (.xlsx)")
    p.add_argument("--sheet", default=0, help="Sheet name or zero-based index")
    p.add_argument("--text-column", default="abstract")
    p.add_argument("--year-column", default="application_year")
    p.add_argument("--id-column", default="publication_number")
    p.add_argument("--output", default="patent_bertopic_results.xlsx")
    p.add_argument("--model-dir", default="topic_model")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--language", default="english", help="BERTopic language argument")
    p.add_argument("--min-topic-size", type=int, default=10)
    p.add_argument("--nr-topics", default="auto")
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-components", type=int, default=5)
    p.add_argument("--umap-min-dist", type=float, default=0.0)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    p.add_argument("--hdbscan-min-samples", type=int, default=None)
    p.add_argument("--recent-years", type=int, default=3)
    p.add_argument("--min-topic-docs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-model", action="store_true")
    return p.parse_args()


def normalize_year(series: pd.Series) -> pd.Series:
    # Accept years or date-like cells.
    numeric = pd.to_numeric(series, errors="coerce")
    year = numeric.where(numeric.between(1800, 2200))
    missing = year.isna()
    if missing.any():
        dates = pd.to_datetime(series[missing], errors="coerce")
        year.loc[missing] = dates.dt.year
    return year.astype("Int64")


def load_data(args):
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.input, sheet_name=sheet, engine="openpyxl")
    required = [args.text_column, args.year_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    if args.id_column not in df.columns:
        df[args.id_column] = np.arange(1, len(df) + 1).astype(str)
    df[args.text_column] = df[args.text_column].fillna("").astype(str).str.strip()
    df[args.year_column] = normalize_year(df[args.year_column])
    df = df[(df[args.text_column] != "") & df[args.year_column].notna()].copy()
    df[args.year_column] = df[args.year_column].astype(int)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError("No usable records remain after removing empty text / invalid years.")
    return df


def build_model(args):
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as e:
        raise RuntimeError(
            "Missing packages. Install requirements.txt first: pip install -r requirements.txt"
        ) from e

    embedding_model = SentenceTransformer(args.embedding_model)
    umap_model = UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=args.umap_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english" if args.language.lower() == "english" else None)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language=args.language,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model, embedding_model


def topic_year_table(df, year_col, topic_col="topic"):
    # Denominator uses all usable documents, including BERTopic outliers.
    totals = df.groupby(year_col).size().rename("year_total")
    valid = df[df[topic_col] != -1]
    counts = valid.groupby([year_col, topic_col]).size().rename("topic_count").reset_index()
    topics = sorted(valid[topic_col].unique())
    years = np.arange(df[year_col].min(), df[year_col].max() + 1)
    if not topics:
        return pd.DataFrame(columns=[year_col, topic_col, "topic_count", "year_total", "share", "growth", "rolling_share", "rolling_growth"])
    grid = pd.MultiIndex.from_product([years, topics], names=[year_col, topic_col]).to_frame(index=False)
    x = grid.merge(counts, on=[year_col, topic_col], how="left")
    x["topic_count"] = x["topic_count"].fillna(0).astype(int)
    x = x.merge(totals.reset_index(), on=year_col, how="left")
    x["year_total"] = x["year_total"].fillna(0).astype(int)
    x["share"] = np.where(x["year_total"] > 0, x["topic_count"] / x["year_total"], 0.0)
    x = x.sort_values([topic_col, year_col]).reset_index(drop=True)
    eps = 1e-6
    prev = x.groupby(topic_col)["share"].shift(1)
    x["growth"] = np.log((x["share"] + eps) / (prev + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x["rolling_share"] = x.groupby(topic_col)["share"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    prev_roll = x.groupby(topic_col)["rolling_share"].shift(1)
    x["rolling_growth"] = np.log((x["rolling_share"] + eps) / (prev_roll + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x


def centroids_and_cohesion(embeddings, topics):
    embeddings = np.asarray(embeddings)
    topics = np.asarray(topics)
    centroids, cohesion = {}, {}
    for t in sorted(set(topics)):
        if t == -1:
            continue
        v = embeddings[topics == t]
        c = v.mean(axis=0)
        centroids[int(t)] = c
        cohesion[int(t)] = float(cosine_similarity(v, c.reshape(1, -1)).mean())
    return centroids, pd.Series(cohesion, name="cohesion", dtype=float)


def historical_topic_novelty(df, embeddings, topics, year_col):
    """Novelty against centroids of older topics using original embedding space.

    For topic k, we take its documents in its first appearance year, create an onset
    centroid, then compare it with centroids of documents belonging to OTHER topics
    strictly before that year. novelty = 1 - maximum cosine similarity.
    """
    e = np.asarray(embeddings)
    t = np.asarray(topics)
    years = df[year_col].to_numpy()
    out = {}
    for k in sorted(set(t)):
        if k == -1:
            continue
        first = int(years[t == k].min())
        onset_mask = (t == k) & (years == first)
        onset = e[onset_mask].mean(axis=0).reshape(1, -1)
        old_topics = sorted(set(t[(years < first) & (t != -1) & (t != k)]))
        old_centroids = []
        for j in old_topics:
            mask = (t == j) & (years < first)
            if mask.any():
                old_centroids.append(e[mask].mean(axis=0))
        if not old_centroids:
            out[int(k)] = np.nan
        else:
            sim = cosine_similarity(onset, np.vstack(old_centroids))[0]
            out[int(k)] = float(1.0 - np.max(sim))
    return pd.Series(out, name="novelty", dtype=float)


def make_summary(df, history, cohesion, novelty, recent_years, year_col):
    max_year = int(df[year_col].max())
    recent_start = max_year - recent_years + 1
    recent = history[history[year_col] >= recent_start]
    base = history.groupby("topic").agg(
        total_topic_count=("topic_count", "sum"),
        first_year=(year_col, lambda s: int(s.iloc[0])),
    ).reset_index()
    # Correct first year: first year with nonzero topic count.
    first_nonzero = history[history.topic_count > 0].groupby("topic")[year_col].min().rename("first_year_nonzero")
    base = base.drop(columns="first_year").merge(first_nonzero, left_on="topic", right_index=True, how="left").rename(columns={"first_year_nonzero": "first_year"})
    latest = history[history[year_col] == max_year][["topic", "topic_count", "share"]].rename(columns={"topic_count": "latest_count", "share": "recent_share"})
    metrics = recent.groupby("topic").agg(
        mean_growth=("rolling_growth", "mean"),
        recent_mean_share=("share", "mean"),
        active_recent_years=("topic_count", lambda s: int((s > 0).sum())),
    )
    metrics["persistence"] = metrics["active_recent_years"] / float(recent_years)
    result = base.merge(latest, on="topic", how="left").merge(metrics, left_on="topic", right_index=True, how="left")
    result = result.merge(cohesion, left_on="topic", right_index=True, how="left")
    result = result.merge(novelty, left_on="topic", right_index=True, how="left")
    return result


def safe_minmax(series: pd.Series, neutral=0.5):
    s = pd.to_numeric(series, errors="coerce")
    fill = s.median() if s.notna().any() else 0.0
    x = s.fillna(fill).to_numpy(dtype=float)
    if len(x) == 0:
        return np.array([])
    if np.nanmax(x) == np.nanmin(x):
        return np.full(len(x), neutral, dtype=float)
    return MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()


def score_weak_signals(summary, min_topic_docs=3):
    x = summary.copy()
    x["frequency_norm"] = safe_minmax(x["recent_mean_share"])
    x["rarity_norm"] = 1.0 - x["frequency_norm"]
    x["growth_norm"] = safe_minmax(x["mean_growth"])
    x["novelty_norm"] = safe_minmax(x["novelty"])
    x["cohesion_norm"] = safe_minmax(x["cohesion"])
    x["persistence_norm"] = safe_minmax(x["persistence"])
    x["weak_signal_score"] = (
        0.20 * x["rarity_norm"]
        + 0.25 * x["growth_norm"]
        + 0.25 * x["novelty_norm"]
        + 0.15 * x["cohesion_norm"]
        + 0.15 * x["persistence_norm"]
    )
    # Reliability guard: tiny topics remain visible, but are not promoted as candidates.
    x["eligible"] = (x["total_topic_count"] >= min_topic_docs) & (x["active_recent_years"] >= 1)
    x["weak_signal_candidate"] = x["eligible"] & (x["mean_growth"] > 0)
    x = x.sort_values(["weak_signal_candidate", "weak_signal_score"], ascending=[False, False]).reset_index(drop=True)
    x["weak_signal_rank"] = np.arange(1, len(x) + 1)
    return x


def add_topic_labels(summary, topic_model):
    info = topic_model.get_topic_info()
    name_map = dict(zip(info["Topic"], info["Name"]))
    x = summary.copy()
    x["topic_name"] = x["topic"].map(name_map)
    def words(k):
        vals = topic_model.get_topic(int(k)) or []
        return ", ".join(w for w, _ in vals[:10])
    x["representative_words"] = x["topic"].map(words)
    cols = ["weak_signal_rank", "topic", "topic_name", "representative_words"] + [c for c in x.columns if c not in {"weak_signal_rank", "topic", "topic_name", "representative_words"}]
    return x[cols]


def outlier_analysis(df, embeddings, topics, year_col):
    t = np.asarray(topics)
    idx = np.where(t == -1)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["row_index", year_col, "nearest_topic", "nearest_similarity", "outlier_novelty"])
    centroids, _ = centroids_and_cohesion(embeddings, topics)
    if not centroids:
        return pd.DataFrame({"row_index": idx, year_col: df.iloc[idx][year_col].to_numpy()})
    ids = sorted(centroids)
    matrix = np.vstack([centroids[k] for k in ids])
    sims = cosine_similarity(np.asarray(embeddings)[idx], matrix)
    best = sims.argmax(axis=1)
    return pd.DataFrame({
        "row_index": idx,
        year_col: df.iloc[idx][year_col].to_numpy(),
        "nearest_topic": [ids[i] for i in best],
        "nearest_similarity": sims[np.arange(len(idx)), best],
        "outlier_novelty": 1.0 - sims[np.arange(len(idx)), best],
    })


def save_map(summary, output_png="weak_signal_map.png"):
    import matplotlib.pyplot as plt
    x = summary.copy()
    if x.empty:
        return
    sizes = 50 + 450 * x["cohesion_norm"].fillna(0.5)
    colors = x["novelty_norm"].fillna(0.5)
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(x["recent_mean_share"], x["mean_growth"], s=sizes, c=colors, alpha=0.75)
    for _, r in x.head(min(15, len(x))).iterrows():
        plt.annotate(str(int(r["topic"])), (r["recent_mean_share"], r["mean_growth"]), fontsize=8)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Recent mean topic share")
    plt.ylabel("Recent rolling growth")
    plt.title("BERTopic Weak Signal Map")
    cb = plt.colorbar(sc)
    cb.set_label("Semantic novelty (normalized)")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = load_data(args)
    docs = df[args.text_column].tolist()
    print(f"Loaded {len(df):,} documents; years {df[args.year_column].min()}-{df[args.year_column].max()}")

    topic_model, embedding_model = build_model(args)
    print(f"Encoding with: {args.embedding_model}")
    embeddings = embedding_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    print("Fitting BERTopic...")
    topics, _ = topic_model.fit_transform(docs, embeddings)
    df["topic"] = topics

    history = topic_year_table(df, args.year_column)
    _, cohesion = centroids_and_cohesion(embeddings, topics)
    novelty = historical_topic_novelty(df, embeddings, topics, args.year_column)
    summary = make_summary(df, history, cohesion, novelty, args.recent_years, args.year_column)
    summary = score_weak_signals(summary, args.min_topic_docs)
    summary = add_topic_labels(summary, topic_model)

    topic_info = topic_model.get_topic_info()
    outliers = outlier_analysis(df, embeddings, topics, args.year_column)
    if not outliers.empty and "row_index" in outliers:
        meta_cols = [c for c in [args.id_column, args.text_column] if c in df.columns]
        meta = df.reset_index().rename(columns={"index": "row_index"})[["row_index"] + meta_cols]
        outliers = outliers.merge(meta, on="row_index", how="left")

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Weak Signal Ranking", index=False)
        history.to_excel(writer, sheet_name="Topic History", index=False)
        topic_info.to_excel(writer, sheet_name="BERTopic Topics", index=False)
        df.to_excel(writer, sheet_name="Documents", index=False)
        outliers.to_excel(writer, sheet_name="Outlier Candidates", index=False)

    save_map(summary)

    config = vars(args).copy()
    with open("analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not args.no_save_model:
        try:
            topic_model.save(args.model_dir, serialization="safetensors", save_ctfidf=True)
        except Exception as e:
            print(f"Warning: model save failed: {e}")

    print(f"Saved: {args.output}")
    print("Saved: weak_signal_map.png")
    print("Saved: analysis_config.json")


if __name__ == "__main__":
    main()
