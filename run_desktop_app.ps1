if (Test-Path ".\fmp\fmp\python.exe") {
    .\fmp\fmp\python.exe -m streamlit run .\Home.py
} else {
    python -m streamlit run .\Home.py
}
