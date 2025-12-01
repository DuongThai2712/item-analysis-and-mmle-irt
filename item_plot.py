import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


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

    sns.histplot(df[f'{col_name}TA'], bins=30, binrange=range, ax=axes[0, 1], kde=False, color="r")
    axes[0, 1].set_xlabel('Tiếng Anh')
    axes[0, 1].set_ylabel('Số lượng')
    for p in axes[0, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 1].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)

    sns.histplot(df[f'{col_name}TO'], bins=30, binrange=range, ax=axes[1, 0], kde=False, color="orange")
    axes[1, 0].set_xlabel('Toán')
    axes[1, 0].set_ylabel('Số lượng')
    for p in axes[1, 0].patches:   
        height = p.get_height()
        if height > 0:
            axes[1, 0].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)

    sns.histplot(df[f'{col_name}KH'], bins=30, binrange=range, ax=axes[1, 1], kde=False, color="g")
    axes[1, 1].set_xlabel('Tư duy khoa học')
    axes[1, 1].set_ylabel('Số lượng')
    for p in axes[1, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[1, 1].annotate(f'{int(height)}', 
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=12)

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

def oxy_item(ab, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(24, 12))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    # Scatter plots
    plt.scatter(y=ab['a_est'], x=ab['b_est'], color='b', label='Tiếng Việt')
    plt.scatter(y=ab['a_est'], x=ab['b_est'], color='r', label='Tiếng Anh')
    plt.scatter(y=ab['a_est'], x=ab['b_est'], color='orange', label='Toán')
    plt.scatter(y=ab['a_est'], x=ab['b_est'], color='g', label='Tư duy khoa học')

    # Gán nhãn số câu
    for i in range(len(ab)):
        plt.annotate(str(i+1), (ab['b_est'].iloc[i], ab['a_est'].iloc[i]), 
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

def wrap_labels(ax, width=10):
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(t.get_text(), width)) for t in ax.get_xticklabels()]
    )
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(t.get_text(), width)) for t in ax.get_yticklabels()]
    )

def heatmap_pair(df, col1, col2, title, cmap, order1=None, order2=None, ax=axes):
    if order1 is not None:
        df[col1] = pd.Categorical(df[col1], categories=order1, ordered=True)
    if order2 is not None:
        df[col2] = pd.Categorical(df[col2], categories=order2, ordered=True)

    ct = pd.crosstab(df[col1], df[col2])

    #plt.figure(figsize=(6, 6))
    ax = sns.heatmap(ct, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)

    ax.set_title(title)
    plt.xlabel(col2)
    plt.ylabel(col1)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # Tự động xuống dòng
    wrap_labels(ax, width=8)

    plt.tight_layout()