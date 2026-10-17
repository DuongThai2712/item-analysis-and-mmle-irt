"""
CHẤM ĐIỂM + CHUYỂN BÀI LÀM VỀ ĐỀ GỐC

Cách dùng:
    python cham_diem.py dot1_2027.csv mapping_ans_merged.csv
    python cham_diem.py dot1_2027.csv mapping_ans_merged.csv made_thisinh.csv   # (tuỳ chọn) file SBD -> MaDe

Quy ước file mapping (phân cách ';'): mỗi dòng 1 mã đề, cột CauN = câu gốc N,
ô có dạng "{STT câu ở đề sao}-{4 chữ cái}", vd "2-DCA*B":
    - 4 chữ cái = đáp án GỐC ứng với vị trí A, B, C, D của đề sao
    - dấu * đặt ngay sau chữ cái là đáp án đúng (có thể có nhiều dấu *)
    - không có dấu * nào  -> tất cả thí sinh được điểm câu đó

Dữ liệu thi: cột CauK = đáp án thí sinh chọn ở câu thứ K CỦA ĐỀ SAO (A/B/C/D, trống = bỏ).
Mã đề lấy từ cột MaDe trong file dữ liệu, hoặc từ file thứ 3 (SBD,MaDe).
Nếu cả hai đều không có -> suy đoán mã đề theo mức khớp cao nhất (CÓ CẢNH BÁO, xem cuối file).
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ====================== CẤU HÌNH ======================
DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "dot1_2027.csv"
MAP_FILE = sys.argv[2] if len(sys.argv) > 2 else "mapping_ans_merged.csv"
MADE_FILE = sys.argv[3] if len(sys.argv) > 3 else None   # file SBD,MaDe (tuỳ chọn)
LETTERS = "ABCD"
# ======================================================


def clean_made(x):
    """Chuẩn hoá mã đề: bỏ ký tự tổ hợp/ký tự lạ/khoảng trắng, chỉ giữ chữ số."""
    return re.sub(r"\D", "", unicodedata.normalize("NFD", str(x)))


def parse_mapping(path):
    """Trả về {made: {goc: (pos, [(chu_goc, is_correct) x4])}}"""
    m = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    m.columns = [c.strip() for c in m.columns]
    out = {}
    for _, r in m.iterrows():
        made = clean_made(r["MaDe"])
        qs = {}
        for col in m.columns[1:]:
            q = int(re.findall(r"\d+", col)[0])
            cell = r[col]
            if not isinstance(cell, str) or not cell.strip():
                qs[q] = None                       # câu lỗi / không có -> bỏ qua
                continue
            mt = re.fullmatch(r"(\d+)-([A-D*]+)", cell.strip().upper())
            if not mt:
                raise ValueError(f"Mã đề {made}, {col}: ô '{cell}' sai định dạng")
            pos, body = int(mt.group(1)), mt.group(2)
            opts = []
            for ch in body:
                if ch == "*":
                    if not opts:
                        raise ValueError(f"Mã đề {made}, {col}: '*' đứng đầu ô '{cell}'")
                    opts[-1][1] = True
                else:
                    opts.append([ch, False])
            if sorted(o[0] for o in opts) != list(LETTERS):
                raise ValueError(f"Mã đề {made}, {col}: '{cell}' không phải hoán vị của ABCD")
            qs[q] = (pos, [tuple(o) for o in opts])
        out[made] = qs
    return out


def convert_student(row, qs):
    """Với 1 thí sinh + 1 mã đề -> (dict goc->chữ cái gốc hoặc '', dict goc->1/0/-1/None)"""
    ans, score = {}, {}
    for q, info in qs.items():
        if info is None:
            ans[q], score[q] = "", np.nan
            continue
        pos, opts = info
        raw = row.get(f"Cau{pos}")
        raw = raw.strip().upper() if isinstance(raw, str) else ""
        has_key = any(o[1] for o in opts)
        if raw in tuple(LETTERS):
            letter, ok = opts[LETTERS.index(raw)]
            ans[q] = letter
            score[q] = 1 if (ok or not has_key) else 0
        else:                                         # bỏ trống / ký tự lạ
            ans[q] = ""
            score[q] = 1 if not has_key else -1       # câu không có đáp án đúng: ai cũng được điểm
    return ans, score


def main():
    df = pd.read_csv(DATA_FILE, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    maps = parse_mapping(MAP_FILE)
    n_q = max(max(qs) for qs in maps.values())

    # ---------- xác định mã đề ----------
    note = None
    if "MaDe" in df.columns:
        df["MaDe"] = df["MaDe"].map(clean_made)
    elif MADE_FILE:
        mm = pd.read_csv(MADE_FILE, dtype=str, encoding="utf-8-sig")
        mm.columns = [c.strip() for c in mm.columns]
        df = df.merge(mm[["SBD", "MaDe"]], on="SBD", how="left")
        df["MaDe"] = df["MaDe"].map(clean_made)
    else:
        best, margin = [], []
        for _, r in df.iterrows():
            s = {}
            for made, qs in maps.items():
                _, sc = convert_student(r, qs)
                v = [x for x in sc.values() if x in (0, 1)]
                s[made] = np.mean(v) if v else 0
            top = sorted(s.items(), key=lambda kv: -kv[1])
            best.append(top[0][0])
            margin.append(round(top[0][1] - top[1][1], 4))
        df["MaDe"] = best
        df["MaDe_DoLech"] = margin
        note = "MaDe được SUY ĐOÁN theo mức khớp cao nhất (không có trong dữ liệu)."

    unknown = set(df["MaDe"]) - set(maps)
    if unknown:
        print(f"[!] Bỏ qua thí sinh có mã đề không có trong mapping: {sorted(unknown)}")
        df = df[df["MaDe"].isin(maps)].copy()

    # ---------- chuyển về đề gốc + chấm ----------
    info_cols = [c for c in ["SBD", "Name", "Gioi", "MaDe", "MaDe_DoLech"] if c in df.columns]
    ans_rows, sc_rows = [], []
    for _, r in df.iterrows():
        a, s = convert_student(r, maps[r["MaDe"]])
        base = {c: r[c] for c in info_cols}
        ans_rows.append({**base, **{f"Cau{q}": a.get(q, "") for q in range(1, n_q + 1)}})
        sc_rows.append({**base, **{f"Cau{q}": s.get(q, np.nan) for q in range(1, n_q + 1)}})
    ans_df = pd.DataFrame(ans_rows)
    sc_df = pd.DataFrame(sc_rows)
    qcols = [f"Cau{q}" for q in range(1, n_q + 1)]
    sc_df[qcols] = sc_df[qcols].astype("Int64")          # 1/0/-1, ô trống nếu câu lỗi

    # ---------- thống kê tỉ lệ đúng theo câu gốc ----------
    # Tỉ lệ đúng = số đúng / số thí sinh có làm (loại bỏ trống). Tỉ lệ bỏ trống tính trên tổng.
    def stat(sub):
        x = sub[qcols].astype(float)
        done = x.where(x != -1)                          # bỏ trống -> NaN
        return pd.DataFrame({
            "SoLam": done.notna().sum(),
            "TyLeDung": done.mean().round(4),
            "TyLeBoTrong": (x == -1).mean().round(4),
        })

    st = stat(sc_df).add_prefix("ToanMau_")
    for made in sorted(maps):
        sub = sc_df[sc_df["MaDe"] == made]
        if len(sub):
            st = st.join(stat(sub)[["SoLam", "TyLeDung", "TyLeBoTrong"]].add_prefix(f"{made}_"))
    st.index = [int(c[3:]) for c in st.index]
    st.index.name = "CauGoc"

    # ---------- xuất ----------
    out = Path(Path(DATA_FILE).stem)
    out.mkdir(exist_ok=True)
    ans_df.to_csv(out / "bai_lam_de_goc.csv", index=False, encoding="utf-8-sig")
    sc_df.to_csv(out / "ket_qua_cham_diem.csv", index=False, encoding="utf-8-sig")
    st.reset_index().to_excel(out / "thong_ke_ty_le_dung.xlsx", index=False)

    print(f"Đã xuất vào thư mục: {out}/")
    print(f"  Số thí sinh: {len(df)} | Số câu gốc: {n_q}")
    print(df["MaDe"].value_counts().sort_index().to_string())
    if note:
        print("\n[CẢNH BÁO]", note)
        print("  Độ lệch (điểm cao nhất - cao nhì) < 0.05:", int((df['MaDe_DoLech'] < 0.05).sum()), "thí sinh")


if __name__ == "__main__":
    main()