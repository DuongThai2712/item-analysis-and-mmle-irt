import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from pygam import GAM, s
import os
import re
import irt
import linking_equating as le

# tách thành 4 DataFrame tương ứng với 4 phần thì, mỗi phần 30 câu hỏi
def tach_phan(df_chamdiem):
    df_info = df_chamdiem[['SBD', 'MaDe', 'Gioi']]
    df_TV = pd.concat([df_info, df_chamdiem[[f'Cau{i}' for i in range(1, 31)]]], axis=1)
    df_TA = pd.concat([df_info, df_chamdiem[[f'Cau{i}' for i in range(31, 61)]]], axis=1)
    df_TO = pd.concat([df_info, df_chamdiem[[f'Cau{i}' for i in range(61, 91)]]], axis=1)
    df_KH = pd.concat([df_info, df_chamdiem[[f'Cau{i}' for i in range(91, 121)]]], axis=1)
    return df_TV, df_TA, df_TO, df_KH

# hàm tính điểm thô (số câu đúng) và đếm số câu thí sinh bỏ trống
def tinh_diem(df_chamdiem):
    df_chamdiem['Raw'] = df_chamdiem[[f'{i}' for i in df_chamdiem.columns if i.startswith('Cau')]].apply(lambda x: sum(x == 1), axis=1)
    df_chamdiem['Null'] = df_chamdiem[[f'{i}' for i in df_chamdiem.columns if i.startswith('Cau')]].apply(lambda x: sum(x == -1), axis=1)
    return df_chamdiem

# # Hàm chuyển đổi đáp án thành giá trị (0 là sai hoặc không làm, 1 là đúng, -1 là trống)
# def chamDiem(x, answer):
#     ma_de = x['MaDe']
#     row = answer.loc[answer['MaDe'] == ma_de]
    
#     for i in range(1, 121):
#         thi_sinh_dap_an = str(x.get(f'Cau{i}', '')).strip().upper()
        
#         # Lấy đáp án đúng nếu tồn tại
#         if not row.empty:
#             dap_an_dung = str(row.iloc[0].get(f'Cau{i}', '')).strip().upper()
#         else:
#             dap_an_dung = ''

#         # Xử lý các trường hợp đặc biệt
#         if pd.isna(dap_an_dung):     # Xử lý chặn trên và sai đề
#             x[f'Cau{i}'] = 1
#         elif pd.isna(x[f'Cau{i}']): # xử lý bỏ trống câu hỏi
#             x[f'Cau{i}'] = -1
#         else:
#             # Tách các đáp án đúng theo dấu /
#             cac_dap_an = [da.strip() for da in dap_an_dung.split('/')]
#             # Xử lý kết quả bài làm
#             x[f'Cau{i}'] = 1 if thi_sinh_dap_an in cac_dap_an else 0

#     return x

# Hàm chuyển đổi đáp án thành giá trị (0 là sai hoặc không làm, 1 là đúng, -1 là trống)
def chamDiem(x, answer):
    ma_de = x['MaDe']
    row = answer.loc[answer['MaDe'] == ma_de]
    
    # Kiểm tra xem đáp án có dùng định dạng [STT]-[Đáp án] không
    is_multi_format = False
    position_map = {}  # {stt_cau_goc: dap_an_dung}
    
    if not row.empty:
        # Kiểm tra ô đầu tiên có dữ liệu
        for i in range(1, 121):
            cell_val = str(row.iloc[0].get(f'Cau{i}', '')).strip()
            if '-' in cell_val:
                is_multi_format = True
                break
    
    if is_multi_format and not row.empty:
        # Parse toàn bộ đáp án: mỗi ô có dạng "STT-ĐÁP_ÁN"
        # Cau{i} trong file đáp án = vị trí thứ i trong đề này
        # Giá trị = "STT_GỐC-ĐÁP_ÁN"
        col_to_original = {}  # {col_index_trong_de_nay: (stt_goc, dap_an)}
        for i in range(1, 121):
            cell_val = str(row.iloc[0].get(f'Cau{i}', '')).strip()
            if '-' in cell_val:
                parts = cell_val.split('-', 1)
                try:
                    stt_goc = int(parts[0].strip())
                    dap_an = parts[1].strip().upper()
                    col_to_original[i] = (stt_goc, dap_an)
                except (ValueError, IndexError):
                    col_to_original[i] = (i, '')
            elif cell_val and cell_val != 'NAN':
                col_to_original[i] = (i, cell_val.upper())
            else:
                col_to_original[i] = (i, '')
        
        # Đọc bài làm thí sinh theo vị trí cột (Cau1..Cau120)
        # rồi map về câu gốc để chấm
        # Kết quả ghi vào cột theo STT GỐC
        
        # Thu thập đáp án thí sinh theo vị trí cột
        thi_sinh_answers = {}  # {col_index: dap_an_thi_sinh}
        for i in range(1, 121):
            val = x.get(f'Cau{i}', '')
            if pd.isna(val) or str(val).strip() == '':
                thi_sinh_answers[i] = None
            else:
                thi_sinh_answers[i] = str(val).strip().upper()
        
        # Reset tất cả cột về -1 (mặc định bỏ trống)
        result = {}
        for i in range(1, 121):
            result[i] = -1
        
        # Chấm điểm: duyệt theo cột đề này
        for col_i, (stt_goc, dap_an_dung) in col_to_original.items():
            if stt_goc < 1 or stt_goc > 120:
                continue
            
            thi_sinh_dap_an = thi_sinh_answers.get(col_i)
            
            if not dap_an_dung or dap_an_dung == 'NAN':
                # Không có đáp án → câu đó tất cả được điểm
                result[stt_goc] = 1
            elif thi_sinh_dap_an is None:
                # Thí sinh bỏ trống
                result[stt_goc] = -1
            else:
                cac_dap_an = [da.strip() for da in dap_an_dung.split('/')]
                result[stt_goc] = 1 if thi_sinh_dap_an in cac_dap_an else 0
        
        # Ghi kết quả vào x theo STT GỐC
        for stt_goc, diem in result.items():
            x[f'Cau{stt_goc}'] = diem
    
    else:
        # Định dạng cũ (1 đề, không có mapping) → dùng logic gốc
        for i in range(1, 121):
            thi_sinh_dap_an = str(x.get(f'Cau{i}', '')).strip().upper()
            
            if not row.empty:
                dap_an_dung = str(row.iloc[0].get(f'Cau{i}', '')).strip().upper()
            else:
                dap_an_dung = ''
            
            if pd.isna(x.get(f'Cau{i}', float('nan'))) or row.empty:
                if row.empty or pd.isna(row.iloc[0].get(f'Cau{i}', float('nan'))):
                    x[f'Cau{i}'] = 1
                else:
                    x[f'Cau{i}'] = -1
            else:
                if not dap_an_dung or dap_an_dung == 'NAN':
                    x[f'Cau{i}'] = 1
                elif pd.isna(x[f'Cau{i}']):
                    x[f'Cau{i}'] = -1
                else:
                    cac_dap_an = [da.strip() for da in dap_an_dung.split('/')]
                    x[f'Cau{i}'] = 1 if thi_sinh_dap_an in cac_dap_an else 0
    
    return x

def ketQuaCham(df, answer):
    df_chamdiem = df.copy()
    df_chamdiem = df_chamdiem.apply(lambda x: chamDiem(x, answer), axis=1)
    return df_chamdiem

# Vẽ biểu đồ thành phần 
def draw_plot(df, col_name: str, title: str, range):
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', 'cursive', 'fantasy'
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(24, 12))

    sns.histplot(df[f'{col_name}TV'], bins=30, binrange=range, ax=axes[0, 0], kde=False, color="b")
    axes[0, 0].set_xlabel('Tiếng Việt')
    axes[0, 0].set_ylabel('Số lượng')
    #thêm giá trị vào từng cột
    for p in axes[0, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 0].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)
    axes[0,0].set_xlim(range[0], range[1])

    sns.histplot(df[f'{col_name}TA'], bins=30, binrange=range, ax=axes[0, 1], kde=False, color="r")
    axes[0, 1].set_xlabel('Tiếng Anh')
    axes[0, 1].set_ylabel('Số lượng')
    for p in axes[0, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 1].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)
    axes[0,1].set_xlim(range[0], range[1])

    sns.histplot(df[f'{col_name}TO'], bins=30, binrange=range, ax=axes[1, 0], kde=False, color="orange")
    axes[1, 0].set_xlabel('Toán')
    axes[1, 0].set_ylabel('Số lượng')
    for p in axes[1, 0].patches:   
        height = p.get_height()
        if height > 0:
            axes[1, 0].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)
    axes[1,0].set_xlim(range[0], range[1])

    sns.histplot(df[f'{col_name}KH'], bins=30, binrange=range, ax=axes[1, 1], kde=False, color="g")
    axes[1, 1].set_xlabel('Tư duy khoa học')
    axes[1, 1].set_ylabel('Số lượng')
    for p in axes[1, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[1, 1].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)
    axes[1,1].set_xlim(range[0], range[1])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.xlim(range[0], range[1])
    plt.savefig(f"figures/{title}.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_total(data: pd.DataFrame, range, title: str, xlabel: str, ylabel: str, lim, color):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12,6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] 

    sns.histplot(data, bins=24, binrange=range, kde=True, color=color)
    #thêm giá trị vào từng cột
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{int(height)}', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center',va='bottom', fontsize=10)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(lim[0], lim[1])
    plt.savefig(f"figures/{title}.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_total_with_stats(df_true, title, color):
    total = df_true.shape[0]
    max_v = df_true['Total_True'].max()
    min_v = df_true['Total_True'].min()
    mean = df_true['Total_True'].mean()
    std = df_true['Total_True'].std()
    median = df_true['Total_True'].median()

    plot_total(
        df_true['Total_True'],
        title=title,
        range=(1,1200),
        color=color,
        xlabel='Điểm thực',
        ylabel='Số lượng',
        lim=(0, 1200)
    )
    stats_text = f"Tổng số thí sinh dự thi: {total}\nCao nhất: {max_v}\nThấp nhất: {min_v:.0f}\nTrung bình: {np.round(mean,2)}\nTrung vị: {np.round(median,2)}\nĐộ lệch chuẩn: {np.round(std,2)}"
    # Thêm một mục "giả" vào legend để hiển thị chú thích kết quả thống kê
    plt.plot([], ' ', label=stats_text)
    plt.legend(loc='upper left', fontsize=12)
    plt.savefig(f"figures/{title}.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_item(data_1, data_2, data_3, data_4, title, order, palette, size):
    sns.set_style("whitegrid")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=size)

    data1 = data_1.copy()
    data2 = data_2.copy()
    data3 = data_3.copy()
    data4 = data_4.copy()

    data1["Đề"] = "Đề 1"
    data2["Đề"] = "Đề 2"
    data3["Đề"] = "Đề 3"
    data4["Đề"] = "Đề 4"

    merged = pd.concat(
        [data1, data2, data3, data4],
        ignore_index=True
    )

    sns.countplot(
        data=merged,
        x='Phân loại',
        palette=palette,
        order=order,
        hue='Đề'
    )

    plt.title(title, fontsize=14)
    plt.xlabel(None)
    plt.ylabel('Số lượng')

    # thêm nhãn trên mỗi cột
    for p in plt.gca().patches:
        height = p.get_height()

        if height > 0:
            plt.gca().annotate(
                f'{int(height)}',
                (
                    p.get_x() + p.get_width() / 2,
                    height
                ),
                ha='center',
                va='bottom',
                fontsize=10
            )

    plt.tight_layout()
    plt.show()


def oxy_item(item_params, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(24, 12))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Scatter plots
    plt.scatter(y=item_params['a'].iloc[0:30], x=item_params['b'].iloc[0:30], color='b', label='Tiếng Việt')
    plt.scatter(y=item_params['a'].iloc[30:60], x=item_params['b'].iloc[30:60], color='r', label='Tiếng Anh')
    plt.scatter(y=item_params['a'].iloc[60:90], x=item_params['b'].iloc[60:90], color='orange', label='Toán')
    plt.scatter(y=item_params['a'].iloc[90:120], x=item_params['b'].iloc[90:120], color='g', label='Tư duy khoa học')

    # Gán nhãn số câu
    for i in range(len(item_params)):
        plt.annotate(str(i+1), (item_params['b'].iloc[i], item_params['a'].iloc[i]), 
                    textcoords="offset pixels", xytext=(6, 6), ha='right')

    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.title(title, fontsize=16)
    plt.legend(title='Dạng câu hỏi')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figures/{title}.png", dpi=150, bbox_inches='tight')
    plt.show()

# def plot_one(ax, theta, right, title_txt, color:str, lim_low, lim_high):
#         ax.scatter(theta, right, color=color, alpha=0.3)
#         gam = GAM(s(0, n_splines=20)).fit(theta, right)
#         # theta_grid = np.linspace(theta.min(), theta.max(), 300).reshape(-1, 1)
#         pred_grid = np.linspace(lim_low, lim_high, 1000).reshape(-1, 1)
#         raw_pred = gam.predict(pred_grid)         
#         # vẽ đường cong fit
#         # ax.plot(smoothed[:,0], smoothed[:,1], linewidth=2, color='black')
#         ax.plot(pred_grid, raw_pred, linewidth=2, color=color)

#         ax.set_title(title_txt)
#         ax.set_xlabel("Theta")
#         ax.set_ylabel("Điểm thô")
#         ax.set_xlim(lim_low,lim_high)
#         ax.set_ylim(0, 32)

def plot_one(ax, theta, right, title_txt, color: str, lim_low, lim_high, ymax=32, n_splines=20):
    theta = np.asarray(theta, dtype=float)
    right = np.asarray(right, dtype=float)
    ok = np.isfinite(theta) & np.isfinite(right)       # bỏ thí sinh không ước lượng theta
    theta, right = theta[ok], right[ok]

    ax.set_title(title_txt)
    ax.set_xlabel("Theta")
    ax.set_ylabel("Điểm thô")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(0, ymax)
    if len(theta) == 0:
        return
    ax.scatter(theta, right, color=color, alpha=0.3)

    n_spl = min(n_splines, max(len(np.unique(theta)) - 1, 4))
    if len(theta) < 30 or len(np.unique(theta)) < 5:   # quá ít dữ liệu để fit đường cong
        return
    gam = GAM(s(0, n_splines=n_spl)).fit(theta.reshape(-1, 1), right)

    # chỉ vẽ trong khoảng có dữ liệu, tránh ngoại suy ở hai đầu
    lo, hi = max(lim_low, theta.min()), min(lim_high, theta.max())
    grid = np.linspace(lo, hi, 1000).reshape(-1, 1)
    pred = np.clip(gam.predict(grid), 0, ymax)
    ax.plot(grid, pred, linewidth=2, color=color)

def draw_box_plot(
    data_1, data_2, data_3, data_4,
    x, y,
    pallete_1, pallete_2, pallete_3, pallete_4,
    title
):
    # Layout 2x2, kích thước 12x12
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Đề 1
    sns.boxplot(
        data=data_1,
        x=x,
        y=y,
        palette=pallete_1,
        ax=axes[0]
    )
    axes[0].set_title('Đề 1')
    axes[0].set_xlabel(None)

    # Đề 2
    sns.boxplot(
        data=data_2,
        x=x,
        y=y,
        palette=pallete_2,
        ax=axes[1]
    )
    axes[1].set_title('Đề 2')
    axes[1].set_xlabel(None)

    # Đề 3
    sns.boxplot(
        data=data_3,
        x=x,
        y=y,
        palette=pallete_3,
        ax=axes[2]
    )
    axes[2].set_title('Đề 3')
    axes[2].set_xlabel(None)

    # Đề 4
    sns.boxplot(
        data=data_4,
        x=x,
        y=y,
        palette=pallete_4,
        ax=axes[3]
    )
    axes[3].set_title('Đề 4')
    axes[3].set_xlabel(None)

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


# ======================================================================
# BỔ SUNG: biểu đồ điểm thô - theta, boxplot / countplot cho 1 đề
# ======================================================================
def _raw_col(df):
    """Dùng RawKept (điểm thô sau khi loại câu) nếu có, ngược lại dùng Raw."""
    return 'RawKept' if 'RawKept' in df.columns else 'Raw'


def _n_cau(df):
    """Số câu hỏi (cột Cau...) của một phần thi."""
    return sum(1 for c in df.columns if str(c).startswith('Cau'))


def drawTTC(df_TV, df_TA, df_TO, df_KH, title: str):
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(24, 12))

    # ------------------------- chỉnh trục -------------------------
    for ax in axes.flat:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    # ---------------------- gọi các subplot ----------------------
    # ymax theo số câu của từng phần (không mặc định 32 nữa)
    plot_one(axes[0][0], df_TV["Theta"], df_TV[_raw_col(df_TV)], "Tiếng Việt", color='b', lim_low=-6, lim_high=6, ymax=_n_cau(df_TV) + 2)
    plot_one(axes[0][1], df_TA["Theta"], df_TA[_raw_col(df_TA)], "Tiếng Anh", color='r', lim_low=-6, lim_high=6, ymax=_n_cau(df_TA) + 2)
    plot_one(axes[1][0], df_TO["Theta"], df_TO[_raw_col(df_TO)], "Toán", color='y', lim_low=-6, lim_high=6, ymax=_n_cau(df_TO) + 2)
    plot_one(axes[1][1], df_KH["Theta"], df_KH[_raw_col(df_KH)], "Tư duy khoa học", color='g', lim_low=-6, lim_high=6, ymax=_n_cau(df_KH) + 2)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/{title}.png", dpi=150, bbox_inches='tight')
    plt.show()


#hiện tương quan giữa theta và độ khó b, vẽ 2 đường KDE
def plot_kde_theta_b(ax, theta, diff, title, color):
        sns.kdeplot(
            x=theta,
            color=color, fill=False, alpha=0.7, label="Năng lực (theta)", ax=ax
        )
        sns.kdeplot(
            x=diff,
            color=color, fill=False, alpha=1.0, linestyle="--", label="Độ khó (b)", ax=ax
        )

        ax.set_xlim(-6, 6)
        ax.set_ylabel("Mật độ")
        ax.set_title(f"{title}")
        ax.legend()
        plt.tight_layout()
        plt.show()
    

# ----------------------------------------------------------------------------------------------
# Vẽ
# ----------------------------------------------------------------------------------------------
def _stat_handle(ax, text, fontsize=9):
    """Thêm một mục 'giả' vào legend để in thông số (giống plot_total_with_stats trong item_plot.py)."""
    ax.plot([], " ", label=text)



def plot_tcc_anchor(r, save_dir, ax=None, dpi=150):
    theta = np.linspace(-6, 6, 601)
    a_x, b_x = r["aX_irt"], r["bX"]
    n = len(r["ids"])
    t_before = irt.irt_probability(theta, a_x, b_x).sum(axis=1)
    t_after = irt.irt_probability(theta, le.to_irt(r["aX_link"]), r["bX_link"]).sum(axis=1)
    t_y = irt.irt_probability(theta, le.to_irt(r["aY_L"]), r["bY"]).sum(axis=1)
 
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(theta, t_before, color=le.COL_X, ls="--", lw=2, label="X – trước link (thang X)")
    ax.plot(theta, t_after, color=le.COL_XL, lw=2, label="X – sau link (thang Y)")
    ax.plot(theta, t_y, color=le.COL_Y, lw=2, label="Y")
    s = r["summary"]
    _stat_handle(ax, f"A = {r['A']:.4f}, B = {r['B']:.4f}\nSố neo dùng: {s['N_used']}/{s['N_anchor']}\n"
                     f"SLcrit = {r['crit']:.5f}\nTCC RMSD (neo) = {s['TCC_RMSD(anchor)']:.3f}")
    ax.set_title(f"{r['label']}")
    ax.set_xlabel("Theta")
    ax.set_ylabel("Điểm thô kỳ vọng")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, n)
    ax.grid(True, alpha=0.3)
    le.style_axes_zero(ax)
    ax.legend(loc="upper left", fontsize=9)
    if own:
        fig.tight_layout()
        path = os.path.join(save_dir, f"TCC_{r['code']}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
 

def draw_one_box_plot(data, x, y, pallete, title, de_label='Đề 1', figsize=(6, 6)):
    # Giống draw_box_plot nhưng chỉ vẽ 1 ô (1 đề)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        palette=pallete,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(None)

    plt.tight_layout()
    plt.savefig(f"figures/{title}.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_item_1test_only(data, title, order, palette, size, de_label='Đề 1'):
    # Giống plot_item nhưng chỉ cho 1 đề
    sns.set_style("whitegrid")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=size)

    data1 = data.copy()
    data1["Đề"] = de_label

    sns.countplot(
        data=data1,
        x='Phân loại',
        palette=palette,
        order=order,
        hue='Đề'
    )

    plt.title(title, fontsize=14)
    plt.xlabel(None)
    plt.ylabel('Số lượng')

    # thêm nhãn trên mỗi cột
    for p in plt.gca().patches:
        height = p.get_height()

        if height > 0:
            plt.gca().annotate(
                f'{int(height)}',
                (
                    p.get_x() + p.get_width() / 2,
                    height
                ),
                ha='center',
                va='bottom',
                fontsize=10
            )

    plt.tight_layout()
    plt.show()


# ======================================================================
# BỔ SUNG: ICC / IIF theo từng câu hỏi (expected + observed), tự động lưu hình
# ======================================================================
def _item_no(name):
    """'Cau12' -> '12' (nếu không có số thì giữ nguyên tên)."""
    m = re.search(r'(\d+)$', str(name))
    return m.group(1) if m else str(name)


def _observed_curve(theta, u, n_bins=10, missing_as_wrong=True):
    """
    Đường observed: chia thí sinh thành n_bins nhóm có số người bằng nhau theo theta,
    trả về (theta trung bình, tỉ lệ đúng, số người) của từng nhóm.
    Câu bỏ trống (-1) tính là sai (khớp cách fit_2pl) nếu missing_as_wrong=True, ngược lại bị bỏ qua.
    """
    theta = np.asarray(theta, dtype=float)
    u = np.asarray(u, dtype=float)
    if missing_as_wrong:
        u = np.where(u == irt.MISSING, 0.0, u)
        ok = np.isfinite(theta) & np.isfinite(u)
    else:
        ok = np.isfinite(theta) & np.isfinite(u) & (u != irt.MISSING)
    theta, u = theta[ok], u[ok]
    if len(theta) < n_bins * 3:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(theta, kind='stable')
    groups = np.array_split(order, n_bins)
    t_obs = np.array([theta[g].mean() for g in groups])
    p_obs = np.array([u[g].mean() for g in groups])
    n_obs = np.array([len(g) for g in groups])
    return t_obs, p_obs, n_obs


# Vẽ ICC (Item Characteristic Curve)
def draw_ICC(ax, a, b, label, color, theta_estimates, responses=None, n_bins=10, title=None,
             missing_as_wrong=True, infit=None, outfit=None, legend_loc='upper left'):
    theta = np.linspace(-6, 6, 1000)
    plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', 'cursive', 'fantasy'
    plt.rcParams['font.serif'] = ['Times New Roman']
    p = irt.irt_probability(theta, a, b).ravel()
    ax.plot(theta, p, color=color, alpha=0.8, linewidth=2, label='Expected')   # đường fit (mô hình)
    if responses is not None:                                                  # đường observed
        t_obs, p_obs, _ = _observed_curve(theta_estimates, responses, n_bins, missing_as_wrong)
        if len(t_obs) > 0:
            ax.plot(t_obs, p_obs, color='black', marker='o', markersize=5, linewidth=1,
                    label='Observed')
    ax.set_title(title if title is not None else f'Đường đặc trưng - {label}')
    ax.set_xlabel('Theta')
    ax.set_ylabel('P')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    # chú thích chỉ số fit của câu, trình bày giống plot_total_with_stats
    if infit is not None and outfit is not None and np.isfinite(infit) and np.isfinite(outfit):
        stats_text = f"Infit MSQ: {infit:.2f}\nOutfit MSQ: {outfit:.2f}"
        # Thêm một mục "giả" vào legend để hiển thị chú thích kết quả thống kê
        ax.plot([], ' ', label=stats_text)
    ax.legend(loc=legend_loc, fontsize=12)




# Vẽ IIF (Item Information Function)
def draw_IIF(ax, a, b, label, color, theta_estimates, responses=None, n_bins=10, title=None,
             use_D=True, missing_as_wrong=True):
    plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', 'cursive', 'fantasy'
    plt.rcParams['font.serif'] = ['Times New Roman']
    theta = np.linspace(-6, 6, 1000)
    iif = irt.cal_iif(a, b, theta, use_D)
    theta_ok = np.asarray(theta_estimates, dtype=float)
    theta_ok = theta_ok[np.isfinite(theta_ok)]
    iif_estimates = irt.cal_iif(a, b, theta_ok, use_D)  # Vector hóa
    ax.plot(theta, iif, color=color, alpha=0.8, linewidth=2, label='Expected')  # đường fit (mô hình)
    # observed: thông tin tại từng nhóm theta, tính từ tỉ lệ đúng quan sát được p_obs
    #   IIF_obs = k * p_obs * (1 - p_obs)   (cùng hệ số k với đường expected)
    if responses is not None:
        t_obs, p_obs, _ = _observed_curve(theta_estimates, responses, n_bins, missing_as_wrong)
        if len(t_obs) > 0:
            k = (irt.D * a) ** 2 if use_D else a ** 2
            ax.plot(t_obs, k * p_obs * (1 - p_obs), color='red', linestyle='--', marker='o',
                    markersize=4, linewidth=1.5, label='Observed')
    # kẻ một đường thẳng đi qua cực trị lớn nhất của IIF
    if len(iif_estimates) > 0:
        max_iif = np.max(iif_estimates)
        ax.axhline(max_iif, color='gray', linestyle='--', alpha=0.5)
        # chú thích giá trị cực trị
        ax.annotate(f'{max_iif:.4f}', xy=(theta_ok[np.argmax(iif_estimates)], max_iif), xytext=(-6, max_iif), ha='left')
    ax.set_title(title if title is not None else f'Đường thông tin - {label}')
    ax.set_xlabel('Theta')
    ax.set_ylabel('IIF')
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')


def plot_item_curves(df, item_params, test_name, lan='lan1', save_dir='plots', kinds=('ICC', 'IIF'),
                     color='b', n_bins=10, theta_col='Theta', show=False, dpi=150,
                     title_fmt='{kind} Câu {item} - a = {a:.2f}, b = {b:.2f}',
                     dir_fmt='{save_dir}/{lan}/{test}', name_fmt='{kind}_{item}.png',
                     missing_as_wrong=True, use_D=True):
    """
    Vẽ ICC và IIF cho TỪNG câu của một phần thi, tự động lưu mỗi hình thành một file.

    df          : DataFrame một phần thi (cột Cau... và cột theta_col)
    item_params : DataFrame index = tên câu, cột a, b (và Used nếu có; câu Used=False bị bỏ qua)
    Quy ước lưu : {save_dir}/{lan}/{test}/{ICC|IIF}_{Cau}.png
                  vd plots/lan1/TV/ICC_Cau12.png  (đổi bằng dir_fmt / name_fmt)
    Trả về      : DataFrame [Item, Kind, Path]
    """
    folder = dir_fmt.format(save_dir=save_dir, lan=lan, test=test_name)
    os.makedirs(folder, exist_ok=True)
    theta = df[theta_col].to_numpy(dtype=float)

    params = item_params
    if 'Used' in params.columns:
        params = params[params['Used'].astype(bool)]
    params = params[np.isfinite(params['a']) & np.isfinite(params['b'])]

    saved = []
    for item in params.index:
        if item not in df.columns:
            continue
        a, b = float(params.loc[item, 'a']), float(params.loc[item, 'b'])
        u = df[item].to_numpy(dtype=float)
        for kind in kinds:
            fig, ax = plt.subplots(figsize=(8, 8))
            title = title_fmt.format(kind=kind, item=_item_no(item), a=a, b=b)
            if kind == 'ICC':
                infit = params.loc[item, 'Infit_MSQ'] if 'Infit_MSQ' in params.columns else None
                outfit = params.loc[item, 'Outfit_MSQ'] if 'Outfit_MSQ' in params.columns else None
                draw_ICC(ax, a, b, item, color, theta, responses=u, n_bins=n_bins,
                         title=title, missing_as_wrong=missing_as_wrong,
                         infit=infit, outfit=outfit)
            else:
                draw_IIF(ax, a, b, item, color, theta, responses=u, n_bins=n_bins,
                         title=title, use_D=use_D, missing_as_wrong=missing_as_wrong)
            path = os.path.join(folder, name_fmt.format(kind=kind, item=item, test=test_name, lan=lan))
            fig.tight_layout()
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)
            saved.append((item, kind, path))
    return pd.DataFrame(saved, columns=['Item', 'Kind', 'Path'])


def plot_anchor_curves(r, save_dir, dpi=150):
    """ICC và IIF (lưu thành 2 file riêng) cho từng câu neo; legend ghi a, b của X gốc, X sau link, Y."""
    theta = np.linspace(-6, 6, 1000)
    folder = os.path.join(save_dir, "anchors", r["code"])
    os.makedirs(folder, exist_ok=True)
    paths = []
    for k, (xid, yid) in enumerate(zip(r["anchors"].X, r["anchors"].Y)):
        used = bool(r["used"][k])
        aX, bX = r["ax_irt"][k], r["bx_a"][k]                 # thang irt.py (sheet X)
        aXL, bXL = r["ax_a"][k] / r["A"], r["A"] * r["bx_a"][k] + r["B"]   # logit, thang Y
        aY, bY = r["ay_a"][k], r["by_a"][k]                    # logit (D=1)
        curves = [
            ("X gốc (thang X)", aX, bX, le.COL_X, "--"),
            ("X sau link (thang Y)", aXL / irt.D, bXL, le.COL_XL, "-"),
            ("Y", aY / irt.D, bY, le.COL_Y, "-"),
        ]
        stats = (f"X gốc: a = {aX:.3f} (D={irt.D:g}) | {aX * irt.D:.3f} (D=1), b = {bX:.3f}\n"
                 f"X sau link: a = {aXL:.3f}, b = {bXL:.3f}  (D=1)\n"
                 f"Y: a = {aY:.3f}, b = {bY:.3f}  (D=1)\n")
        tag = "" if used else "  [bị loại khỏi link]"
        for kind in ("ICC", "IIF"):
            fig, ax = plt.subplots(figsize=(8, 8))
            for lab, a_irt, b_, col, ls in curves:
                if kind == "ICC":
                    y = irt.irt_probability(theta, a_irt, b_).ravel()
                else:
                    y = irt.cal_iif(a_irt, b_, theta)
                ax.plot(theta, y, color=col, ls=ls, lw=2, alpha=0.85, label=lab)
            _stat_handle(ax, stats)
            ax.set_title(f"{kind} câu neo X{xid} ↔ Y{yid} – {r['label']}{tag}")
            ax.set_xlabel("Theta")
            ax.set_ylabel("P" if kind == "ICC" else "IIF")
            ax.set_xlim(-6, 6)
            if kind == "ICC":
                ax.set_ylim(0, 1)
            # legend (kèm a, b) đặt dưới trục để không che đường cong
            ax.legend(fontsize=8.5, frameon=True)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = os.path.join(folder, f"{kind}_X{xid}_Y{yid}.png")
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
    return paths

