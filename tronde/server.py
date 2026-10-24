# -*- coding: utf-8 -*-
"""
Web server trộn đề + bảng đáp án theo đề gốc.

Chạy:   pip install flask openpyxl
        python server.py            (mở http://127.0.0.1:5000)
        python server.py --port 8000

Tính năng:
  * Import thư mục project tex (chọn thư mục trên trình duyệt, hoặc nhập đường dẫn)
  * Nhập số mã đề sao -> tạo mã đề (dùng chung lõi với tronde.py)
  * Bảng đáp án theo thứ tự đề gốc: mỗi mã đề 4 cột A-D, ô đúng tô vàng
  * Bấm ô ở khối "Gốc" để đánh dấu / bỏ đánh dấu đáp án đúng -> các mã đề sao đổi theo
  * Xuất: dapan.xlsx, mapping_ans_merged.csv, log + bảng đáp án (zip), đề tex (zip)
"""
import io
import os
import sys
import json
import shutil
import random
import zipfile
import argparse
import threading
import contextlib

from flask import Flask, request, jsonify, send_file, send_from_directory
for _s in (sys.stdout, sys.stderr):      # tránh UnicodeEncodeError (cp1252) khi print tiếng Việt
    if _s is not None and hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')
        
import tronde as T

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, 'server_data')
PROJ_DIR = os.path.join(WORK, 'project')
OUT_DIR = os.path.join(WORK, 'output')

app = Flask(__name__, static_folder=os.path.join(BASE, 'static'), static_url_path='/static')
app.config['MAX_FORM_PARTS'] = 20000
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

S = {'P': None, 'root': None, 'plans': [], 'lock': threading.RLock()}

import traceback
from werkzeug.exceptions import HTTPException


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=f'{e.code} {e.name}: {e.description}'), e.code
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    return jsonify(error=tb[-1200:]), 500
# ------------------------------------------------------------
# tiện ích
# ------------------------------------------------------------

def err(msg, code=400):
    return jsonify(error=str(msg)), code


def find_root(base):
    """Thư mục nông nhất có đủ 1.tex + main.tex (người dùng có thể chọn thư mục
    cha, hoặc trình duyệt thêm 1 cấp thư mục gốc vào đường dẫn)."""
    best = None
    for dp, dn, fn in os.walk(base):
        if '1.tex' in fn and 'main.tex' in fn:
            depth = dp[len(base):].count(os.sep)
            if best is None or depth < best[0]:
                best = (depth, dp)
    return best[1] if best else None


def sync_outputs():
    """Ghi lại toàn bộ .tex / bảng đáp án theo marks HIỆN TẠI (sau khi người
    dùng sửa đáp án đúng gốc trên giao diện)."""
    P = S['P']
    if not P or not S['plans']:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for plan in S['plans']:
        T.render_exam(P, plan, OUT_DIR)
    T.write_all_tables(P, S['plans'], OUT_DIR)


def state_json():
    P = S['P']
    if not P:
        return {'loaded': False, 'n': 0, 'warnings': [], 'questions': [], 'exams': []}
    n = P.n_questions
    exams = []
    for plan in S['plans']:
        by = T.plan_by_orig(plan)
        exams.append({
            'index': plan['index'], 'code': plan['code'],
            # với mỗi câu gốc q: [STT trên đề sao, 4 chữ cái GỐC ở vị trí A,B,C,D]
            'map': [[by[q]['new'], ''.join(T.LETTERS[i] for i in by[q]['perm'])]
                    for q in range(1, n + 1)],
        })
    return {
        'loaded': True, 'root': S['root'], 'n': n,
        'warnings': P.warnings,
        'questions': [dict(q, marks=T.letters_str(P.marks[q['n']])) for q in P.questions],
        'exams': exams,
    }


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/state')
def api_state():
    with S['lock']:
        return jsonify(state_json())


@app.post('/api/import')
def api_import():
    with S['lock']:
        try:
            files = request.files.getlist('files')
            if files:
                shutil.rmtree(PROJ_DIR, ignore_errors=True)
                os.makedirs(PROJ_DIR, exist_ok=True)
                saved = 0
                for f in files:
                    rel = (f.filename or '').replace('\\', '/')
                    parts = [p for p in rel.split('/') if p not in ('', '.')]
                    if not parts or '..' in parts or ':' in parts[0]:
                        continue
                    dest = os.path.join(PROJ_DIR, *parts)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    f.save(dest)
                    saved += 1
                if not saved:
                    return err('Không nhận được file nào.')
                base = PROJ_DIR
            else:
                path = (request.get_json(silent=True) or {}).get('path', '').strip().strip('"').strip("'")
                if not path or not os.path.isdir(path):
                    return err(f'Đường dẫn không tồn tại: {path!r}')
                base = path
            root = find_root(base)
            if not root:
                return err("Không tìm thấy thư mục có đủ '1.tex' và 'main.tex' trong project đã chọn.")
            P = T.load_project(root)
            S['P'], S['root'], S['plans'] = P, root, []
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            return jsonify(state_json())
        except T.BraceMismatchError as e:
            return err(f'Lỗi ngoặc {{}} trong file nội dung:\n{e}')
        except (FileNotFoundError, UnicodeDecodeError) as e:
            return err(e)


@app.post('/api/generate')
def api_generate():
    with S['lock']:
        P = S['P']
        if not P:
            return err('Chưa import project.')
        try:
            n = int((request.get_json(silent=True) or {}).get('n', 0))
        except (TypeError, ValueError):
            return err('Số mã đề không hợp lệ.')
        if not 1 <= n <= 50:
            return err('Số mã đề phải từ 1 đến 50.')
        rng = random.SystemRandom()
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        plans = []
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # tracker in rất nhiều dòng -> chỉ giữ trong log
                for k in range(1, n + 1):
                    plan = T.make_exam_plan(P, k, rng, total_exams=n)
                    T.render_exam(P, plan, OUT_DIR)
                    plans.append(plan)
            S['plans'] = plans
            T.write_all_tables(P, plans, OUT_DIR)
        except T.BraceMismatchError as e:
            return err(f'Lỗi ngoặc {{}} trong file nội dung:\n{e}')
        print(f"Đã tạo {n} mã đề: {', '.join(p['code'] for p in plans)}")
        return jsonify(state_json())


@app.post('/api/marks')
def api_marks():
    """body: {q: số câu gốc, marks: 'AC'}  -> đặt tập đáp án đúng GỐC của câu q."""
    with S['lock']:
        P = S['P']
        if not P:
            return err('Chưa import project.')
        body = request.get_json(silent=True) or {}
        try:
            q = int(body['q'])
            letters = str(body.get('marks', ''))
        except (KeyError, TypeError, ValueError):
            return err('Dữ liệu không hợp lệ.')
        if q not in P.marks or any(ch not in T.LETTERS for ch in letters):
            return err('Câu hoặc đáp án không hợp lệ.')
        P.marks[q] = {T.LETTERS.index(ch) for ch in letters}
        return jsonify(ok=True)


@app.get('/api/log/<code>')
def api_log(code):
    with S['lock']:
        for plan in S['plans']:
            if plan['code'] == code:
                return jsonify(text=plan['tracker'].text())
    return err('Không có mã đề này.', 404)


def _need_exams():
    return S['P'] is not None and bool(S['plans'])


@app.get('/api/export/xlsx')
def export_xlsx():
    with S['lock']:
        if not _need_exams():
            return err('Chưa có mã đề nào.')
        sync_outputs()
        return send_file(os.path.join(OUT_DIR, 'dapan.xlsx'), as_attachment=True,
                         download_name='dapan.xlsx')


@app.get('/api/export/csv')
def export_csv():
    with S['lock']:
        if not _need_exams():
            return err('Chưa có mã đề nào.')
        sync_outputs()
        return send_file(os.path.join(OUT_DIR, 'mapping_ans_merged.csv'), as_attachment=True,
                         download_name='mapping_ans_merged.csv')


def _zip_response(name, writer):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        writer(z)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=name, mimetype='application/zip')


@app.get('/api/export/logs')
def export_logs():
    """Log từng mã đề + bảng đáp án (txt) + mapping csv + dapan.xlsx + cảnh báo."""
    with S['lock']:
        if not _need_exams():
            return err('Chưa có mã đề nào.')
        sync_outputs()
        P = S['P']

        def w(z):
            for plan in S['plans']:
                c = plan['code']
                z.write(os.path.join(OUT_DIR, c, f'log_{c}.txt'), f'log/log_{c}.txt')
                z.write(os.path.join(OUT_DIR, c, f'answer_table_{c}.txt'), f'answer_table/answer_table_{c}.txt')
            z.write(os.path.join(OUT_DIR, 'mapping_ans_merged.csv'), 'mapping_ans_merged.csv')
            z.write(os.path.join(OUT_DIR, 'dapan.xlsx'), 'dapan.xlsx')
            z.writestr('canh_bao_file_goc.txt', '\n'.join(P.warnings) or 'Không có cảnh báo.')
        return _zip_response('log_va_dap_an.zip', w)


@app.get('/api/export/tex')
def export_tex():
    """Toàn bộ project tex của các mã đề sao (mỗi mã đề 1 thư mục)."""
    with S['lock']:
        if not _need_exams():
            return err('Chưa có mã đề nào.')
        sync_outputs()

        def w(z):
            for dp, _, fn in os.walk(OUT_DIR):
                for f in fn:
                    full = os.path.join(dp, f)
                    z.write(full, os.path.relpath(full, OUT_DIR))
        return _zip_response('de_tex.zip', w)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5001)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    print(f'Mở trình duyệt tại http://{a.host}:{a.port}')
    app.run(host=a.host, port=a.port, threaded=True)
