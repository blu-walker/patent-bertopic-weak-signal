@echo off
setlocal
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python patent_bertopic.py --input patents.xlsx --text-column abstract --year-column application_year --id-column publication_number
endlocal
