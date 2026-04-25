@echo off
echo Installing PyInstaller...
venv\Scripts\pip install pyinstaller

echo Building Zishty.exe...
venv\Scripts\pyinstaller --onefile ^
  --add-data "static;static" ^
  --add-data "assistant;assistant" ^
  --add-data ".env;." ^
  --exclude-module rembg ^
  --name Zishty ^
  app.py

echo.
echo Preparing USB folder...
set USB=%USERPROFILE%\Desktop\USB_Zishty
if not exist "%USB%" mkdir "%USB%"
copy dist\Zishty.exe "%USB%\Zishty.exe"
copy .env "%USB%\.env"

echo.
echo ✅ Done! Your USB folder is ready on the Desktop: USB_Zishty
echo Copy the USB_Zishty folder to your USB drive.
echo On the other device: double-click Zishty.exe then open http://127.0.0.1:5000
pause
