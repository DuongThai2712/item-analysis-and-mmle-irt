"""Web chấm thi thử - backend. Chạy: python server.py  ->  http://127.0.0.1:5000
Toàn bộ tính toán gọi lại mã của bạn: cham_diem, irt, irt_series (ô 21 notebook), nhieu (ô 17), linking_equating, item_plot."""
import io, json, os, re, shutil, sqlite3, sys, tempfile, time, contextlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

os.environ.setdefault("MPLBACKEND", "Agg")                          # không mở cửa sổ matplotlib
FROZEN = getattr(sys, "frozen", False)                              # True khi chạy từ app đóng gói (PyInstaller)
HERE = Path(sys._MEIPASS) if FROZEN else Path(__file__).resolve().parent   # nơi chứa static/ và các module
sys.path.insert(0, str(HERE))
import cham_diem, ctt, irt, irt_series, nhieu                      # noqa: E402
import linking_equating as le                                       # noqa: E402
ip = le._import_item_plot()                                         # item_plot (có stub nếu thiếu pygam/statsmodels)

def _data_dir():                                                    # dữ liệu lưu cạnh app; nếu không ghi được thì vào thư mục người dùng
    base = Path(os.environ.get("THITHU_DATA") or ((Path(sys.executable).parent if FROZEN else HERE) / "data"))
    for d in (base, Path.home() / "ThiThuApp_data"):
        try: d.mkdir(parents=True, exist_ok=True); (d / ".w").write_text("1"); return d
        except OSError: continue
    raise SystemExit("Không có thư mục ghi được để lưu dữ liệu")
DATA = _data_dir()
DB = DATA / "app.db"
SEC = le.TESTS                                                      # (mã, tên, đầu, cuối)
CODES = [s[0] for s in SEC]
app = Flask(__name__, static_folder=str(HERE / "static"))


# ------------------------------------------------------------------ tiện ích
def san(o):
    if isinstance(o, dict): return {str(k): san(v) for k, v in o.items()}
    if isinstance(o, (np.ndarray, pd.Series)): return san(o.tolist())
    if isinstance(o, (list, tuple)): return [san(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if o is pd.NA or o is None: return None
    return o

def J(o, code=200): return jsonify(san(o)), code
def records(df): return san(df.astype(object).where(df.notna(), None).to_dict("records"))

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.executescript("""CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY, name TEXT, ts TEXT, n INTEGER);
    CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, ds INTEGER, kind TEXT, name TEXT, ts TEXT, config TEXT, items TEXT, persons TEXT, extra TEXT);""")
    return c

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def dsdir(i): return DATA / f"ds_{i}"

@app.errorhandler(Exception)
def err(e):
    import traceback; traceback.print_exc()
    return J({"error": f"{type(e).__name__}: {e}"}, 400)


# ------------------------------------------------------------------ chấm điểm (cham_diem.py)
@app.post("/api/dataset")
def new_dataset():
    f, m, mm = request.files.get("data"), request.files.get("mapping"), request.files.get("made")
    if not f or not m: raise ValueError("Cần file bài làm và file mapping")
    con = db(); cur = con.execute("INSERT INTO datasets(name,ts,n) VALUES(?,?,0)", (request.form.get("name") or f.filename, now()))
    d = dsdir(cur.lastrowid); d.mkdir(parents=True)
    f.save(d / "data.csv"); m.save(d / "mapping.csv")
    if mm and mm.filename: mm.save(d / "made.csv")
    cham_diem.DATA_FILE, cham_diem.MAP_FILE = "data.csv", "mapping.csv"
    cham_diem.MADE_FILE = "made.csv" if (d / "made.csv").exists() else None
    buf, cwd = io.StringIO(), os.getcwd()
    try:
        os.chdir(d)
        with contextlib.redirect_stdout(buf): cham_diem.main()                 # chạy đúng cham_diem.main()
    except BaseException as e:
        os.chdir(cwd); shutil.rmtree(d, ignore_errors=True); con.execute("DELETE FROM datasets WHERE id=?", (cur.lastrowid,)); con.commit()
        raise ValueError(f"Chấm điểm lỗi: {e}")
    os.chdir(cwd)
    n = len(pd.read_csv(d / "data" / "ket_qua_cham_diem.csv"))
    con.execute("UPDATE datasets SET n=? WHERE id=?", (n, cur.lastrowid)); con.commit()
    return J({"id": cur.lastrowid, "log": buf.getvalue()})

@app.get("/api/datasets")
def datasets(): return J([dict(r) for r in db().execute("SELECT * FROM datasets ORDER BY id DESC")])

@app.delete("/api/dataset/<int:i>")
def del_dataset(i):
    c = db(); c.execute("DELETE FROM datasets WHERE id=?", (i,)); c.execute("DELETE FROM runs WHERE ds=? AND kind='irt'", (i,)); c.commit()
    shutil.rmtree(dsdir(i), ignore_errors=True); return J({"ok": 1})

def frames(ds):
    d = dsdir(ds) / "data"
    ch = pd.read_csv(d / "ket_qua_cham_diem.csv"); ans = pd.read_csv(d / "bai_lam_de_goc.csv")
    for c in ("Gioi", "MaDe"):
        if c not in ch.columns: ch[c] = ""
    F = dict(zip(CODES, ip.tach_phan(ch)))
    for k in F: F[k] = ip.tinh_diem(F[k])
    return ch, ans, F

@app.get("/api/score/<int:ds>")
def score(ds):
    ch, ans, F = frames(ds)
    return J({"columns": list(ch.columns), "rows": records(ch), "ans": records(ans),
              "sections": [{"code": c, "name": n, "lo": a, "hi": b} for c, n, a, b in SEC]})

@app.get("/api/score/<int:ds>/csv/<which>")
def score_csv(ds, which):
    f = {"ans": "bai_lam_de_goc.csv", "score": "ket_qua_cham_diem.csv"}.get(which, "thong_ke_ty_le_dung.xlsx")
    return send_file(dsdir(ds) / "data" / f, as_attachment=True)


# ------------------------------------------------------------------ CTT (ctt.py + ô 11-17 notebook)
def b_category(x):
    return ("Rất dễ" if x > 0.9 else "Dễ" if x > 0.75 else "Tương đối dễ" if x > 0.6 else "Bình thường" if x > 0.4
            else "Tương đối khó" if x > 0.25 else "Khó" if x > 0.1 else "Rất khó") if np.isfinite(x) else None

def a_category(x):
    return ("Kém" if x <= 0 else "Chưa tốt" if x <= .2 else "Chấp nhận được" if x <= .4 else "Tương đối tốt" if x <= .6
            else "Tốt" if x <= .8 else "Rất tốt" if x <= 1 else "Quá tốt") if np.isfinite(x) else None

def correct_keys(ds):
    out = {}
    for made, qs in cham_diem.parse_mapping(dsdir(ds) / "mapping.csv").items():
        for q, info in qs.items():
            if info: out.setdefault(q, Counter()).update("".join(l for l, ok in info[1] if ok))
    return {q: "".join(sorted(c for c, n in cnt.items() if n >= max(cnt.values()) / 2)) if cnt else "" for q, cnt in out.items()}

def ctt_tables(ds, F, ans):
    base = ["SBD", "MaDe", "Gioi", "Raw", "Null"]                    # ctt.py gốc loại đúng 5 cột này; bỏ Theta/True... nếu có
    G = {c: F[c][base + [x for x in F[c].columns if str(x).startswith("Cau")]] for c in CODES}
    diff = pd.concat([ctt.cal_diff(G[c][G[c]["Null"] < 30]) for c in CODES]).rename("CTT").to_frame()
    disc = pd.concat([ctt.cal_disc(G[c][G[c]["Null"] < 30]) for c in CODES]).rename("D-Index").to_frame()
    t = diff.join(disc); t["Độ khó"] = t["CTT"].map(b_category); t["Độ phân biệt"] = t["D-Index"].map(a_category)
    nh = pd.concat([nhieu.do_nhieu(data=ans, chamdiem=F[c], start=lo, end=hi) for c, _, lo, hi in SEC])
    return t, nh

@app.get("/api/ctt/<int:ds>")
def ctt_api(ds):
    ch, ans, F = frames(ds)
    t, nh = ctt_tables(ds, F, ans); keys = correct_keys(ds)
    raw = {c: F[c]["Raw"].tolist() for c in CODES}; raw["TOTAL"] = sum((F[c]["Raw"] for c in CODES)).tolist()
    stats = {k: pd.Series(v).describe().round(3).to_dict() for k, v in raw.items()}
    items, nhrows = [], []
    for c, _, lo, hi in SEC:
        v = ans.merge(F[c][["SBD", "Null"]], on="SBD"); v = v[v["Null"] < 30]
        for q in range(lo, hi + 1):
            n = f"Cau{q}"; s = v[n]
            items.append({"item": q, "section": c, **t.loc[n].to_dict()})
            r = {"item": q, "section": c, "key": keys.get(q, ""), **{f"pbcc_{o}": nh.loc[n, o] for o in ["A", "B", "C", "D", "Null"]},
                 **{f"p_{o}": float((s == o).mean()) for o in "ABCD"}, "p_Null": float(s.isna().mean())}
            nhrows.append(r)
    return J({"raw": raw, "stats": stats, "items": items, "nhieu": nhrows})


# ------------------------------------------------------------------ IRT (irt.py + irt_series.py) + log
def run_row(rid):
    r = db().execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    if not r: raise ValueError("Không có lần chạy này")
    return r

@app.post("/api/irt")
def irt_run():
    p = request.get_json(); ds = int(p["ds"]); ch, ans, F = frames(ds)
    ex = {k: v for k, v in (p.get("exclude") or {}).items() if str(v).strip()}
    a_init = b_init = None
    if p.get("init_run"):                                           # lấy tham số lần chạy cũ làm giá trị khởi đầu
        it = pd.DataFrame(json.loads(run_row(int(p["init_run"]))["items"])).set_index(["Test", "Item"])
        a_init = {c: it.loc[c]["a"].reindex([f"Cau{i}" for i in range(lo, hi + 1)]).fillna(1.0).to_numpy() for c, _, lo, hi in SEC}
        b_init = {c: it.loc[c]["b"].reindex([f"Cau{i}" for i in range(lo, hi + 1)]).fillna(0.0).to_numpy() for c, _, lo, hi in SEC}
    series = [[F[c], hi, c] for c, _, _, hi in SEC]
    items, persons = irt_series.run_irt_series(series, exclude=ex, a_init=a_init, b_init=b_init,
                                               theta_method=p.get("method", "MLE"), verbose=False)
    for c in CODES: irt_series.cal_true_score(F[c], items.loc[c])
    P = ch[["SBD"]].copy()
    for c in CODES:
        P[f"{c}_Raw"], P[f"{c}_Theta"], P[f"{c}_SE"], P[f"{c}_True"] = F[c]["RawKept"], F[c]["Theta"], F[c]["SE"], F[c]["True"].astype(float)
    P["Total_True"] = P[[f"{c}_True" for c in CODES]].sum(axis=1)
    it = items.reset_index(); it["review"] = (it["p_value"] < .05) & ((it["se_b"] > 10) | (it["se_a"] > 3))
    cfg = {"exclude": ex, "method": p.get("method", "MLE"), "init_run": p.get("init_run")}
    c = db(); cur = c.execute("INSERT INTO runs(ds,kind,name,ts,config,items,persons,extra) VALUES(?,?,?,?,?,?,?,?)",
                              (ds, "irt", p.get("name") or f"Lần chạy {now()}", now(), json.dumps(cfg), json.dumps(records(it)),
                               json.dumps(records(P)), "{}")); c.commit()
    return J({"id": cur.lastrowid})

@app.get("/api/runs/<int:ds>")
def runs(ds):
    return J([{**{k: r[k] for k in ("id", "name", "ts")}, "config": json.loads(r["config"])}
              for r in db().execute("SELECT * FROM runs WHERE ds=? AND kind='irt' ORDER BY id DESC", (ds,))])

@app.delete("/api/run/<int:rid>")
def del_run(rid): c = db(); c.execute("DELETE FROM runs WHERE id=?", (rid,)); c.commit(); return J({"ok": 1})

@app.get("/api/run/<int:rid>")
def run_get(rid):
    r = run_row(rid); items = json.loads(r["items"]); cfg = json.loads(r["config"])
    # câu bị loại: gắn tham số gần nhất ở các lần chạy trước để đối chiếu / dùng lại
    hist = [(x["id"], pd.DataFrame(json.loads(x["items"]))) for x in
            db().execute("SELECT * FROM runs WHERE ds=? AND kind='irt' AND id<? ORDER BY id DESC", (r["ds"], rid))]
    for it in items:
        if not it.get("Used"):
            for hid, h in hist:
                m = h[(h["Item"] == it["Item"]) & (h["Used"] == True) & h["a"].notna()]   # noqa: E712
                if len(m): it["prev_a"], it["prev_b"], it["prev_run"] = float(m.iloc[0]["a"]), float(m.iloc[0]["b"]), hid; break
    return J({"id": rid, "name": r["name"], "ts": r["ts"], "ds": r["ds"], "config": cfg, "items": items, "persons": json.loads(r["persons"])})

@app.get("/api/item_history/<int:ds>/<item>")
def item_history(ds, item):
    out = []
    for r in db().execute("SELECT * FROM runs WHERE ds=? AND kind='irt' ORDER BY id", (ds,)):
        for it in json.loads(r["items"]):
            if it["Item"] == item: out.append({"run": r["id"], "name": r["name"], "ts": r["ts"], **{k: it.get(k) for k in ("Used", "a", "b", "se_a", "se_b", "p_value", "Fit_Label")}})
    return J(out)

def run_frames(rid):
    r = run_row(rid); ch, ans, F = frames(r["ds"]); P = pd.DataFrame(json.loads(r["persons"]))
    for c in CODES:
        F[c] = F[c].merge(P[["SBD", f"{c}_Raw", f"{c}_Theta", f"{c}_SE", f"{c}_True"]].rename(
            columns={f"{c}_Raw": "RawKept", f"{c}_Theta": "Theta", f"{c}_SE": "SE", f"{c}_True": "True"}), on="SBD", how="left")
    return r, ch, ans, F, pd.DataFrame(json.loads(r["items"])).set_index(["Test", "Item"]), P


# ------------------------------------------------------------------ ICC / TCC
def parse_items(s):
    out = []
    for t in re.split(r"[,\s;]+", str(s).strip()):
        if re.fullmatch(r"\d+-\d+", t): a, b = map(int, t.split("-")); out += range(a, b + 1)
        elif t.isdigit(): out.append(int(t))
    return sorted(set(out))

@app.post("/api/curves")
def curves():
    p = request.get_json(); r, ch, ans, F, items, P = run_frames(int(p["run"]))
    code = p["section"]; lo, hi = next((a, b) for c, _, a, b in SEC if c == code)
    sel = [q for q in (parse_items(p.get("items", "")) or range(lo, hi + 1)) if lo <= q <= hi]
    it = items.loc[code]; d = F[code]; th = np.linspace(-6, 6, 241); nb = int(p.get("bins", 10))
    used = [q for q in sel if bool(it.loc[f"Cau{q}", "Used"])]
    out = {"theta": th, "items": [], "excluded": [q for q in sel if q not in used]}
    for q in used:
        a, b = float(it.loc[f"Cau{q}", "a"]), float(it.loc[f"Cau{q}", "b"])
        t_o, p_o, n_o = ip._observed_curve(d["Theta"], d[f"Cau{q}"], nb, True)
        out["items"].append({"item": q, "a": a, "b": b, "exp": irt.irt_probability(th, a, b).ravel(), "iif": irt.cal_iif(a, b, th), "obs_x": t_o, "obs_y": p_o,
                             "infit": it.loc[f"Cau{q}", "Infit_MSQ"], "outfit": it.loc[f"Cau{q}", "Outfit_MSQ"]})
    if used:                                                                    # TCC nhóm câu đã chọn
        a = it.loc[[f"Cau{q}" for q in used], "a"].to_numpy(float); b = it.loc[[f"Cau{q}" for q in used], "b"].to_numpy(float)
        U = np.where(d[[f"Cau{q}" for q in used]].to_numpy(float) == 1, 1, 0).sum(axis=1)
        tt = d["Theta"].to_numpy(float); ok = np.isfinite(tt); g = np.array_split(np.argsort(tt[ok], kind="stable"), nb)
        tif = sum(irt.cal_iif(float(x), float(y), th) for x, y in zip(a, b)); out["tif"] = {"y": tif, "se": 1 / np.sqrt(np.maximum(tif, 1e-9)), "n": len(used)}
        out["tcc"] = {"exp": irt.irt_probability(th, a, b).sum(axis=1), "n": len(used),
                      "obs_x": [tt[ok][i].mean() for i in g], "obs_y": [U[ok][i].mean() for i in g]}
    return J(out)


# ------------------------------------------------------------------ Điểm thực + xuất file
@app.get("/api/export/run/<int:rid>")
def export_run(rid):
    r, ch, ans, F, items, P = run_frames(rid); t, nh = ctt_tables(r["ds"], F, ans)
    f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name
    with pd.ExcelWriter(f, engine="openpyxl") as w:                                # giống ô 35 notebook
        for c in CODES: F[c].to_excel(w, sheet_name=c, index=False)
        items.reset_index().to_excel(w, sheet_name="items", index=False)
        P.to_excel(w, sheet_name="true_score", index=False)
        t.to_excel(w, sheet_name="ctt"); nh.to_excel(w, sheet_name="nhieu")
        pd.DataFrame([json.loads(r["config"])]).astype(str).to_excel(w, sheet_name="config", index=False)
    return send_file(f, as_attachment=True, download_name=f"results_run{rid}.xlsx")

@app.get("/api/export/params/<int:rid>")                                         # file tham số dạng sheet X (dùng cho neo đề)
def export_params(rid):
    items = pd.DataFrame(json.loads(run_row(rid)["items"])); items = items[items["Used"] == True]   # noqa: E712
    items["id"] = items["Item"].str.extract(r"(\d+)").astype(int)
    out = items[["Test", "id", "a", "b"]].rename(columns={"Test": "test"}); out.loc[out["test"].duplicated(), "test"] = None
    f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name; out.to_excel(f, sheet_name="X", index=False)
    return send_file(f, as_attachment=True, download_name=f"tham_so_run{rid}.xlsx")


# ------------------------------------------------------------------ Neo đề (linking_equating.py)
def table(text, cols):
    rows = [re.split(r"[,;\t ]+", l.strip()) for l in (text or "").splitlines() if l.strip() and not l.lstrip()[0].isalpha()]
    return pd.DataFrame(rows, columns=cols).apply(pd.to_numeric)

def link_args(o):
    a = ["--no-plots", "--boot", str(o.get("boot", 0)), "--d-x", str(o.get("d_x", irt.D)), "--d-y", str(o.get("d_y", 1.0)),
         "--weights", o.get("weights", "normal"), "--a-min", str(o.get("a_min", .06)), "--b-max", str(o.get("b_max", 4))]
    if o.get("one_sided"): a.append("--one-sided")
    if o.get("purify"): a += ["--purify", str(o["purify"])]
    return le.parse_args(a)

def link_items(Ps, Pt, anc, args, d_s, d_t):
    """Chỉ dùng tham số: cùng logic process_test (screen_anchors + Stocking-Lord), cho hướng bất kỳ."""
    res = []
    for code, label, lo, hi in SEC:
        ids = list(range(lo, hi + 1))
        A_ = anc[anc.S.between(lo, hi) & anc.Dst.between(lo, hi)].reset_index(drop=True)
        if len(A_) < 3: continue
        s, t = Ps.loc[A_.S], Pt.loc[A_.Dst]
        a_irt, b_s, a_s, a_t, b_t = s.a.to_numpy(float), s.b.to_numpy(float), d_s * s.a.to_numpy(float), d_t * t.a.to_numpy(float), t.b.to_numpy(float)
        used, why = le.screen_anchors(a_irt, b_s, a_t, b_t, args.a_min, args.b_max)
        if used.sum() < 3: continue
        g = le.make_grid(args.quad_points, args.theta_lim); w = le.make_weights(g)
        A, B, crit = le.fit_stocking_lord(a_s[used], b_s[used], a_t[used], b_t[used], g, w, g, w, not args.one_sided)
        c1, c2 = le.sl_components(A, B, a_s[used], b_s[used], a_t[used], b_t[used], g, w, g, w, not args.one_sided)
        rm = le.anchor_rmsd(A, B, a_s, b_s, a_t, b_t, g, w)
        aS, bS = d_s * Ps.loc[ids, "a"].to_numpy(float), Ps.loc[ids, "b"].to_numpy(float)
        aT, bT = d_t * Pt.loc[ids, "a"].to_numpy(float), Pt.loc[ids, "b"].to_numpy(float)
        a_l, b_l = aS / A, A * bS + B
        amap = {int(x): int(y) for x, y, u in zip(A_.S, A_.Dst, used) if u}
        fa, fb, st = a_l.copy(), b_l.copy(), []
        for k, i in enumerate(ids):
            if i in amap: j = ids.index(amap[i]); fa[k], fb[k] = aT[j], bT[j]; st.append("Neo (theo đề đích)")
            else: st.append("Link")
        xl = pd.DataFrame({"test": code, "id": ids, "Status": st, "a_nguồn(D=1)": aS, "b_nguồn": bS, "a_link(D=1)": a_l, "b_link": b_l,
                           "a_final(D=1)": fa, "b_final": fb, "a_final(irt.py)": fa / irt.D})
        at = pd.DataFrame({"test": code, "Nguồn": A_.S, "Đích": A_.Dst, "Dùng": used, "Lý do loại": why, "a_nguồn(D=1)": a_s, "b_nguồn": b_s,
                           "a_link": a_s / A, "b_link": A * b_s + B, "a_đích(D=1)": a_t, "b_đích": b_t, "ICC_RMSD": rm})
        tcc = le.tcc_logit(g, a_t[used], b_t[used]) - le.tcc_logit((g - B) / A, a_s[used], b_s[used])
        sm = {"Test": code, "Tên": label, "N_anchor": len(A_), "N_used": int(used.sum()), "A": A, "B": B, "SLcrit": crit,
              "SLcrit_nguồn": c1, "SLcrit_đích": c2, "TCC_RMSD(anchor)": float(np.sqrt(np.sum(w * tcc ** 2)))}
        res.append(dict(code=code, label=label, ids=ids, A=A, B=B, summary=sm, x_linked=xl, anchor_tbl=at,
                        aX_L=aS, bX=bS, aX_link=a_l, bX_link=b_l, aY_L=aT, bY=bT, persons=None))
    if not res: raise ValueError("Không phần nào đủ ≥3 câu neo hợp lệ")
    return res

def link_payload(res):
    th = np.linspace(-6, 6, 241); out = []
    for r in res:
        out.append({"code": r["code"], "label": r["label"], "A": r["A"], "B": r["B"], "summary": r["summary"],
                    "anchors": records(r["anchor_tbl"]), "items": records(r["x_linked"]), "theta": th, "n": len(r["ids"]),
                    "tcc_before": le.tcc_logit(th, r["aX_L"], r["bX"]), "tcc_after": le.tcc_logit(th, r["aX_link"], r["bX_link"]),
                    "tcc_target": le.tcc_logit(th, r["aY_L"], r["bY"]), "persons": r.get("persons")})
    return out

def read_xlsx_params(sheets):
    out = {}
    for k in ("X", "Y"):
        d = sheets[k].copy(); d["id"] = d["id"].astype(int); out[k] = d.set_index("id")[["a", "b"]].astype(float)
    return out["X"], out["Y"], sheets["Anchor"].dropna().astype(int).rename(columns={"X": "S", "Y": "Dst"})

@app.post("/api/link")
def link():
    o = json.loads(request.form.get("opts", "{}")); direction = o.get("direction", "XY"); args = link_args(o)
    f = request.files.get("file"); path = None; sheets = {}
    if f and f.filename:
        path = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name; f.save(path); sheets = pd.read_excel(path, sheet_name=None)
    X = Y = anc = None
    if sheets: X, Y, anc = read_xlsx_params(sheets)
    if o.get("x_text", "").strip(): X = table(o["x_text"], ["id", "a", "b"]).astype({"id": int}).set_index("id")
    if o.get("y_text", "").strip(): Y = table(o["y_text"], ["id", "a", "b"]).astype({"id": int}).set_index("id")
    if o.get("a_text", "").strip(): anc = table(o["a_text"], ["S", "Dst"]).astype(int)
    if X is None or Y is None or anc is None: raise ValueError("Thiếu tham số đề X, đề Y hoặc câu neo (nhập tay hoặc tải file Excel)")
    manual = any(o.get(k, "").strip() for k in ("x_text", "y_text", "a_text"))
    full = bool(path) and direction == "XY" and not manual and all(f"{c}_X" in sheets for c in CODES)
    rid_files = None
    if full:                                                                      # đầy đủ: process_test của linking_equating.py
        _, Xp, Yp, anchors = le.read_workbook(path)
        results = [le.process_test(c, n, lo, hi, sheets, Xp, Yp, anchors, args) for c, n, lo, hi in SEC]
        for r in results:
            pc = pd.DataFrame(r["person_cols"]); pc.insert(0, "SBD", r["dX"]["SBD"].to_numpy()); pc.insert(1, "Raw", r["raw"])
            r["persons"] = records(pc)
        chk, chk_sum = le.link_check(sheets, results)
        outp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name; shutil.copy(path, outp); args.backup = False
        le.write_workbook(path, outp, results, chk, chk_sum, args); rid_files = outp
    else:                                                                         # chỉ tham số, X→Y hoặc Y→X
        Ps, Pt, a = (X, Y, anc) if direction == "XY" else (Y, X, anc.rename(columns={"S": "Dst", "Dst": "S"}))
        ds_, dt_ = (args.d_x, args.d_y) if direction == "XY" else (args.d_y, args.d_x)
        results = link_items(Ps, Pt, a, args, ds_, dt_)
        outp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name
        with pd.ExcelWriter(outp, engine="openpyxl") as w:
            pd.DataFrame([r["summary"] for r in results]).set_index("Test").T.to_excel(w, sheet_name="Link_Summary")
            pd.concat([r["anchor_tbl"] for r in results]).to_excel(w, sheet_name="Anchor_Link", index=False)
            pd.concat([r["x_linked"] for r in results]).to_excel(w, sheet_name="X_Linked", index=False)
        rid_files = outp
    payload = {"mode": "full" if full else "params", "direction": direction, "results": link_payload(results)}
    c = db(); cur = c.execute("INSERT INTO runs(ds,kind,name,ts,config,items,persons,extra) VALUES(?,?,?,?,?,?,?,?)",
                              (0, "link", o.get("name") or f"Neo {direction} {now()}", now(), json.dumps(o), "[]", "[]", json.dumps(san(payload))))
    c.commit(); shutil.copy(rid_files, DATA / f"link_{cur.lastrowid}.xlsx")
    return J({"id": cur.lastrowid, **payload})

@app.get("/api/links")
def links(): return J([{"id": r["id"], "name": r["name"], "ts": r["ts"]} for r in db().execute("SELECT * FROM runs WHERE kind='link' ORDER BY id DESC")])

@app.get("/api/link/<int:rid>")
def link_get(rid): return J({"id": rid, **json.loads(run_row(rid)["extra"])})

@app.get("/api/link/<int:rid>/xlsx")
def link_xlsx(rid): return send_file(DATA / f"link_{rid}.xlsx", as_attachment=True, download_name=f"neo_de_{rid}.xlsx")

@app.get("/api/template")
def template():
    f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name
    ex = lambda: pd.DataFrame({"test": ["TV"] + [None] * 4, "id": range(1, 6), "a": [.5, .6, .7, .8, .9], "b": [-1, -.5, 0, .5, 1]})
    with pd.ExcelWriter(f, engine="openpyxl") as w:
        ex().to_excel(w, sheet_name="X", index=False); ex().to_excel(w, sheet_name="Y", index=False)
        pd.DataFrame({"X": range(1, 6), "Y": range(1, 6)}).to_excel(w, sheet_name="Anchor", index=False)
        pd.DataFrame({"SBD": [1], **{f"Cau{i}": [1] for i in range(1, 31)}}).to_excel(w, sheet_name="TV_X", index=False)
    return send_file(f, as_attachment=True, download_name="mau_neo_de.xlsx")

@app.get("/")
def index(): return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
