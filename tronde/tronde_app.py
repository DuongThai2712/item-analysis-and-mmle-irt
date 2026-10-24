# -*- coding: utf-8 -*-
"""Launcher desktop: chay Flask (server.py) trong thread + cua so pywebview."""
import os
import sys
import socket
import threading

import webview
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

import server

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


if getattr(sys, 'frozen', False):
    # du lieu tam ghi canh file .exe (thu muc _internal co the khong tien ghi)
    base = os.path.dirname(sys.executable)
    server.WORK = os.path.join(base, 'server_data')
    server.PROJ_DIR = os.path.join(server.WORK, 'project')
    server.OUT_DIR = os.path.join(server.WORK, 'output')
os.makedirs(server.WORK, exist_ok=True)

if __name__ == '__main__':
    port = free_port()
    threading.Thread(
        target=lambda: server.app.run(host='127.0.0.1', port=port, threaded=True, use_reloader=False),
        daemon=True).start()
    webview.settings['ALLOW_DOWNLOADS'] = True      # cho nut Xuat xlsx/csv/zip
    webview.create_window('Tron de', f'http://127.0.0.1:{port}', width=1280, height=800)
    webview.start()
