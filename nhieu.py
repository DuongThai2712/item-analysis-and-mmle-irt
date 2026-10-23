"""Trích NGUYÊN VĂN từ notebook (ô 17): do_nhieu."""
import numpy as np
import pandas as pd
import ctt

def do_nhieu(data: pd.DataFrame, chamdiem, start, end):
    # Merge 1 lần
    data = data.merge(chamdiem[['SBD', 'Raw', 'Null']], on='SBD', how='left')
    # ans_row = answer.iloc[0]

    # # Kiểm tra định dạng multi (có dấu '-' trong ô đáp án)
    # is_multi_format = any(
    #     '-' in str(ans_row.get(f'Cau{i}', ''))
    #     for i in range(1, 121)
    #     if '-' in str(ans_row.get(f'Cau{i}', ''))
    # )

    # # Nếu là multi-format, build bảng map: {stt_goc: dap_an_dung}
    # if is_multi_format:
    #     correct_map = {}  # {stt_goc: dap_an}
    #     for i in range(1, 121):
    #         cell_val = str(ans_row.get(f'Cau{i}', '')).strip()
    #         if '-' in cell_val:
    #             parts = cell_val.split('-', 1)
    #             try:
    #                 stt_goc = int(parts[0].strip())
    #                 dap_an = parts[1].strip().upper()
    #                 correct_map[stt_goc] = dap_an
    #             except (ValueError, IndexError):
    #                 correct_map[i] = ''
    #         elif cell_val and cell_val.upper() != 'NAN':
    #             correct_map[i] = cell_val.upper()

    # DataFrame lưu kết quả tổng hợp
    df_all = pd.DataFrame(columns=["A", "B", "C", "D", "Null"])

    # Lọc thí sinh hợp lệ
    valid = data[data['Null'] < 30]

    for stt in range(start, end + 1):
        std = valid['Raw'].std()
        cau_col = f"Cau{stt}"

        # # Lấy đáp án đúng theo định dạng
        # if is_multi_format:
        #     correct = correct_map.get(stt, '')
        # else:
        #     correct = str(ans_row.get(cau_col, '')).strip().upper()

        n_total = valid.shape[0]
        pbcc_dict = {}

        for opt in ["A", "B", "C", "D"]:
            true_group  = valid.loc[valid[cau_col] == opt, 'Raw']
            false_group = valid.loc[valid[cau_col] != opt, 'Raw']
            if true_group.shape[0] == 0 or false_group.shape[0] == 0:
                pb = np.nan
            else:
                p = true_group.shape[0] / valid[cau_col].shape[0]
                pb = round(ctt.cal_pbcc(true_group, false_group, std, p),4)

            pbcc_dict[opt] = pb

        true_group = valid.loc[valid[cau_col].isna(), 'Raw']
        false_group = valid.loc[valid[cau_col].notna(), 'Raw']
        count = true_group.shape[0]
        percent = count / n_total if n_total > 0 else 0
        pbcc_dict["Null"] = round(ctt.cal_pbcc(true_group, false_group, std, percent),4) if count > 0 and false_group.shape[0] > 0 else np.nan

        # PBCC của đáp án đúng
        #pbcc_ans = pbcc_dict.get(correct, np.nan)

        df_all.loc[f'Cau{stt}'] = [
            pbcc_dict["A"],
            pbcc_dict["B"],
            pbcc_dict["C"],
            pbcc_dict["D"],
            pbcc_dict["Null"],

        ]

    return df_all


