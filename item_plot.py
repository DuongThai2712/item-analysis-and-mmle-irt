import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from pygam import GAM, s

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

# Hàm chuyển đổi đáp án thành giá trị (0 là sai hoặc không làm, 1 là đúng, -1 là trống)
def chamDiem(x, answer):
    ma_de = x['MaDe']
    row = answer.loc[answer['MaDe'] == ma_de]
    
    for i in range(1, 121):
        thi_sinh_dap_an = str(x.get(f'Cau{i}', '')).strip().upper()
        
        # Lấy đáp án đúng nếu tồn tại
        if not row.empty:
            dap_an_dung = str(row.iloc[0].get(f'Cau{i}', '')).strip().upper()
        else:
            dap_an_dung = ''

        # Xử lý các trường hợp đặc biệt
        if pd.isna(dap_an_dung):     # Xử lý chặn trên và sai đề
            x[f'Cau{i}'] = 1
        elif pd.isna(x[f'Cau{i}']): # xử lý bỏ trống câu hỏi
            x[f'Cau{i}'] = -1
        else:
            # Tách các đáp án đúng theo dấu /
            cac_dap_an = [da.strip() for da in dap_an_dung.split('/')]
            # Xử lý kết quả bài làm
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
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 9))

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
    plt.show()

def plot_total(data: pd.DataFrame, range, title: str, xlabel: str, ylabel: str, lim, color):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10,5))
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

def plot_item(data_1, data_2, title, order, palette, size):
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    plt.figure(figsize=size)

    data1 = data_1.copy()
    data2 = data_2.copy()

    data1["Đề"] = "Đề 1"
    data2["Đề"] = "Đề 2"
    merged = pd.concat([data1, data2], ignore_index=True)
    sns.countplot(data=merged, x='Phân loại', palette=palette, order=order, hue='Đề')
    plt.title(title, fontsize=14)
    plt.xlabel(None)
    plt.ylabel('Số lượng')

    #thêm nhãn trên mỗi cột
    for p in plt.gca().patches:
        height = p.get_height()
        if height >0:
            plt.gca().annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=10)

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

def plot_one(ax, theta, right, title_txt, color:str):
        ax.scatter(theta, right, color=color, alpha=0.3)
        gam = GAM(s(0, n_splines=20)).fit(theta, right)
        # theta_grid = np.linspace(theta.min(), theta.max(), 300).reshape(-1, 1)
        theta_grid = np.linspace(-6, 6, 400).reshape(-1, 1)
        raw_pred = gam.predict(theta_grid)         
        # vẽ đường cong fit
        # ax.plot(smoothed[:,0], smoothed[:,1], linewidth=2, color='black')
        ax.plot(theta_grid, raw_pred, linewidth=2, color=color)

        ax.set_title(title_txt)
        ax.set_xlabel("Theta")
        ax.set_ylabel("Điểm thô")
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 32)

def draw_box_plot(data_1, data_2, x, y, pallete_1, pallete_2, title):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    sns.boxplot(data=data_1, x=x, y=y, palette=pallete_1, ax=axes[0])
    axes[0].set_title('Đề 1')
    axes[0].set_xlabel(None)
    sns.boxplot(data=data_2, x=x, y=y, palette=pallete_2, ax=axes[1])
    axes[1].set_title('Đề 2')
    axes[1].set_xlabel(None)
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.show()
