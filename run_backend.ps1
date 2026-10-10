if (Test-Path ".\fmp\fmp\python.exe") {
    .\fmp\fmp\python.exe .\run_backend.py --port 8000
} else {
    python .\run_backend.py --port 8000
}
