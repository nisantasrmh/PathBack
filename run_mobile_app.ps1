if (Test-Path ".\fmp\fmp\python.exe") {
    .\fmp\fmp\python.exe -m streamlit run .\mobile_app.py --server.port 8502
} else {
    python -m streamlit run .\mobile_app.py --server.port 8502
}