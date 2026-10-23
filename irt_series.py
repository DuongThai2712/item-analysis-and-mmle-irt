"""Trích NGUYÊN VĂN từ notebook (ô 21): run_irt_series, cal_true_score."""
import re
import numpy as np
import pandas as pd
import irt

NON_ITEM_COLS = ['SBD', 'MaDe', 'Gioi', 'Raw', 'Null', 'Theta', 'SE', 'True', 'RawKept',
                 'Lz', 'Infit_MSQ', 'Outfit_MSQ', 'Flag_Lz', 'Flag_MSQ', 'Flag']


def _item_columns(df, item_prefix="Cau"):
    """Các cột câu hỏi của một phần thi (số câu = len(kết quả))."""
    return [c for c in df.columns if str(c).startswith(item_prefix)]


def _parse_exclude(spec, item_cols, item_prefix="Cau"):
    """spec: None | '3,4,14' | [3, 4, 'Cau14'] | 3  ->  danh sách tên cột cần loại."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [t for t in re.split(r"[,\s;]+", spec.strip()) if t]
    elif isinstance(spec, (int, np.integer)):
        spec = [spec]
    out = []
    for s in spec:
        name = f"{item_prefix}{int(str(s).replace(item_prefix, '').strip())}"
        if name not in item_cols:
            raise ValueError(f"Không có câu '{name}'. Các câu hiện có: {item_cols[0]}..{item_cols[-1]}")
        if name not in out:
            out.append(name)
    return out


def _slice_init(init, name, end_idx, n_total, keep_pos):
    """init: None | dict {name: mảng} | mảng/Series dài 120 (cắt theo end_idx); chỉ giữ các câu được dùng."""
    if init is None:
        return None
    if isinstance(init, dict):
        v = init.get(name)
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
    else:
        arr = np.asarray(init, dtype=float)
        if arr.ndim != 1 and arr.size == max(arr.shape):
            arr = arr.ravel()                     # (n,1) hoặc (1,n) -> (n,)
        if arr.ndim != 1:
            raise ValueError(f"[{name}] init phải là vector 1 chiều, nhưng nhận shape {arr.shape}")
        arr = arr[end_idx - n_total:end_idx]
    if arr.ndim != 1 or len(arr) != n_total:
        raise ValueError(f"[{name}] init phải có {n_total} giá trị (số câu của phần), nhận shape {arr.shape}")
    return arr[keep_pos]


def run_irt_series(series_list, exclude=None, interactive=False,
                   a_init=None, b_init=None, theta_method="MLE",
                   item_prefix="Cau", verbose=True, **fit_kwargs):
    """
    series_list : [[df, end_idx, name], ...]
    exclude     : dict {tên phần: danh sách câu cần loại}, ví dụ {'TV': [3, 4, 14, 30], 'TA': 'Cau5, Cau7'}
    interactive : True -> hỏi nhập từ bàn phím cho phần nào chưa có trong `exclude`
    n_items     : tự lấy theo số cột câu hỏi của từng phần (sau khi loại câu).
    **fit_kwargs: chuyển thẳng cho irt.fit_2pl, ví dụ msq_range=(0.5, 1.5), lz_cut=-2.0, fit_stats=False.

    Trả về:
      items_all   : index (Test, Item), cột a, b, se_a, se_b, Infit_MSQ, Outfit_MSQ, Fit_Label, Chi2, Chi2_df, p_value,
                    Used (False = câu bị loại, các giá trị = NaN)
      persons_all : index (Test, chỉ số gốc), cột SBD, Theta, SE, Lz, Infit_MSQ, Outfit_MSQ, Flag_Lz, Flag_MSQ, Flag
    Mỗi df gốc được thêm cột RawKept (điểm thô tính trên các câu được giữ), Theta, SE.
    """
    if exclude is not None and not isinstance(exclude, dict):
        raise TypeError("exclude phải là dict theo tên phần, ví dụ {'TV': [3, 4]}")

    items_list, persons_list, keys = [], [], []

    for df_original, end_idx, name in series_list:
        all_cols = _item_columns(df_original, item_prefix)
        n_total = len(all_cols)

        spec = exclude.get(name) if exclude else None
        if interactive and not (exclude and name in exclude):
            spec = input(f"[{name}] {n_total} câu. Nhập số câu cần loại (vd: 3,4,14), Enter để bỏ qua: ")
        excluded = _parse_exclude(spec, all_cols, item_prefix)
        kept = [c for c in all_cols if c not in excluded]
        n_items = len(kept)
        if n_items < 5:
            raise ValueError(f"[{name}] chỉ còn {n_items} câu sau khi loại.")
        keep_pos = [all_cols.index(c) for c in kept]

        # Điểm thô và số câu đã làm, chỉ tính trên các câu được giữ
        U = df_original[kept].to_numpy(float)
        ok = np.isfinite(U) & (U != irt.MISSING)
        correct = np.where(ok, U, 0.0).sum(axis=1)
        answered = ok.sum(axis=1)

        # câu bỏ trống tính là sai (khớp fit_2pl): đúng hết = đúng toàn bộ n_items câu, sai hết = đúng 0 câu
        valid_mask = (correct > 0) & (correct < n_items)      # bỏ đúng/sai hết
        all_wrong = correct == 0
        all_right = correct == n_items

        if verbose:
            print(f"[{name}] số câu = {n_total}, loại {len(excluded)} {excluded}, dùng {n_items}; "
                  f"hợp lệ {valid_mask.sum()}/{len(df_original)}, sai hết {all_wrong.sum()}, đúng hết {all_right.sum()}")

        items_fit, persons_valid = irt.fit_2pl(
            df_original.loc[valid_mask, kept], verbose=verbose, theta_method=theta_method,
            a_init=_slice_init(a_init, name, end_idx, n_total, keep_pos),
            b_init=_slice_init(b_init, name, end_idx, n_total, keep_pos), **fit_kwargs)

        theta = np.full(len(df_original), np.nan)
        se = np.full(len(df_original), np.nan)
        theta[valid_mask] = persons_valid["Theta"].to_numpy()
        se[valid_mask] = persons_valid["SE"].to_numpy()
        theta[all_wrong] = -6
        theta[all_right] = 6

        # chỉ số person-fit từ fit_2pl (Lz, Infit_MSQ, Outfit_MSQ, Flag_*); người đúng/sai hết = NaN
        extra_cols = [c for c in persons_valid.columns if c not in ("Theta", "SE")]
        for c in extra_cols:
            vals = persons_valid[c]
            if vals.dtype == bool:
                full = pd.array([pd.NA] * len(df_original), dtype="boolean")
                full[np.flatnonzero(valid_mask)] = vals.to_numpy()
            else:
                full = np.full(len(df_original), np.nan)
                full[valid_mask] = vals.to_numpy()
            df_original[c] = full

        df_original["RawKept"] = correct.astype(int)
        df_original["Theta"] = theta
        df_original["SE"] = se
        df_original.attrs["excluded"] = excluded

        items_full = items_fit.reindex(all_cols)
        items_full["Used"] = items_full.index.isin(kept)
        items_list.append(items_full)

        cols = [c for c in ["SBD"] if c in df_original.columns] + ["Theta", "SE"] + extra_cols
        persons_list.append(df_original[cols].copy())
        keys.append(name)

    items_all = pd.concat(items_list, keys=keys, names=["Test", "Item"])
    persons_all = pd.concat(persons_list, keys=keys, names=["Test"])
    return items_all, persons_all


def cal_true_score(df, item_params, scale_max=300):
    """item_params: items_all.loc[tên phần]. Chỉ dùng các câu Used; điểm thô lấy từ RawKept."""
    p = item_params[item_params["Used"]] if "Used" in item_params.columns else item_params
    cols = list(p.index)
    df["True"] = df.apply(
        lambda row: irt.true_score(theta=row["Theta"], raw=row["RawKept"],
                                   data=row[cols], item_params=p, scale_max=scale_max),
        axis=1,
    ).astype("Int64")
    return df
