"""Khởi động app: chạy server nội bộ + mở cửa sổ riêng (WebView2 trên Windows).
Nếu không có pywebview thì mở bằng trình duyệt mặc định."""
import os, socket, sys, threading, time, urllib.request, webbrowser

if sys.stdout is None or sys.stderr is None:                      # app đóng gói không console: ghi log ra file
    import server as _s
    _log = open(_s.DATA / "app.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _log
import server
from werkzeug.serving import make_server


def free_port(pref=5000):
    for p in (pref, 0):
        with socket.socket() as s:
            try: s.bind(("127.0.0.1", p)); return s.getsockname()[1]
            except OSError: continue


def main():
    port = free_port(); url = f"http://127.0.0.1:{port}/"
    srv = make_server("127.0.0.1", port, server.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for _ in range(100):                                          # chờ server sẵn sàng
        try: urllib.request.urlopen(url + "api/datasets", timeout=1); break
        except Exception: time.sleep(0.1)
    try:
        import webview
        webview.settings["ALLOW_DOWNLOADS"] = True                # cho phép nút Tải/Xuất Excel
        webview.create_window("Chấm thi thử", url, width=1360, height=860, min_size=(980, 600))
        webview.start()
    except Exception as e:
        print("Không mở được cửa sổ app, dùng trình duyệt:", e)
        webbrowser.open(url)
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt: pass
    srv.shutdown()


if __name__ == "__main__":
    main()
