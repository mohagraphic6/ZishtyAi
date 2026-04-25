@echo off
if not exist venv (
    echo Setting up for first time...
    python -m venv venv
    venv\Scripts\pip install flask python-dotenv groq requests rembg
)
echo Starting Zishty...
start http://127.0.0.1:5000/chat-ui
venv\Scripts\python app.py
pause
