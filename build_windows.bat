@echo off
REM Đóng gói app Windows. Cần Python 3.10-3.12 (đã tick "Add to PATH"). Chạy bằng cách bấm đúp file này.
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv || goto :err )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :err
pip install -r requirements.txt pywebview pyinstaller || goto :err
pyinstaller --noconfirm --clean --onedir --windowed --name ThiThuApp ^
  --add-data "static;static" ^
  --hidden-import server --hidden-import cham_diem --hidden-import ctt --hidden-import irt --hidden-import irt_series ^
  --hidden-import nhieu --hidden-import item_plot --hidden-import linking_equating ^
  --collect-submodules webview --collect-all scipy --collect-all seaborn ^
  thi_thu_app.py || goto :err
echo.
echo XONG. Chay: dist\ThiThuApp\ThiThuApp.exe   (copy nguyen thu muc dist\ThiThuApp de dung o noi khac)
pause & exit /b 0
:err
echo LOI khi dong goi - xem thong bao phia tren.
pause & exit /b 1
