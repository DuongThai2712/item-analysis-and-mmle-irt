@echo off
REM Dong goi app TRON DE (Windows). Dung san venv cua project item-analysis-and-mmle-irt.
REM Dat file nay cung thu muc voi: tronde.py, server.py, tronde_app.py, static\index.html
REM Chay bang cach bam dup.
cd /d "%~dp0"

set "VENV=D:\GitHub\item-analysis-and-mmle-irt\.venv"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" ( echo Khong tim thay "%PY%" & goto :err )

REM index.html phai nam trong static\
if not exist static\index.html (
  if exist index.html ( mkdir static 2>nul & copy /y index.html static\index.html >nul ) else ( echo Thieu static\index.html & goto :err )
)

"%PY%" -m pip install flask openpyxl pywebview pyinstaller || goto :err

"%PY%" -m PyInstaller --noconfirm --clean --onedir --windowed --name TronDeApp ^
  --add-data "static;static" ^
  --hidden-import server --hidden-import tronde --hidden-import openpyxl ^
  --collect-submodules webview ^
  tronde_app.py || goto :err

echo.
echo XONG. Chay: D:\GitHub\item-analysis-and-mmle-irt\dist\TronDeApp\TronDeApp.exe   (copy nguyen thu muc dist\TronDeApp de dung o noi khac)
pause & exit /b 0
:err
echo LOI khi dong goi - xem thong bao phia tren.
pause & exit /b 1
