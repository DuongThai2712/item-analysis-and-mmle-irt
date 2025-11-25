import numpy as np
import pandas as pd
# point-biserial correlation
def cal_pbcc(true_group, false_group, std, id_value) -> float:
    if id_value <= 0.025:
        id_value = 0.025 - 1e-6  # Tránh chia cho 0
    elif id_value >= 0.925:
        id_value = 0.925 + 1e-6
    mean_diff = true_group.mean() - false_group.mean()
    r = (mean_diff / std) * np.sqrt(id_value * (1 - id_value))
    return r

# Tính xác suất CTT
def cal_diff(data: pd.DataFrame):
    b = []
    for j in data.drop(columns=['SBD', 'Total', 'Null']).columns:
        true = data.eq(1).sum()
        all = data.count()    # Loại hết những trường hợp bỏ full bài
        #để index theo cột câu hỏi
        b[j] = true / all
    return b

# Tính độ phân biệt CTT
def cal_disc(data: pd.DataFrame):
    a = []
    for j in data.drop(columns=['SBD', 'Total', 'Null']).columns:
        group = int(data.shape[0]*0.27)    # Chia lấy phân vị để tính độ phân biệt
        upper = data[data['SBD']!=0].sort_values(by='Total', ascending=False).head(group)      # Nhóm cao điểm (loại bỏ trường hợp chặn trên)
        lower = data[(data['Total'] > 0)].sort_values(by='Total', ascending=True).head(group) # Nhóm thấp điểm (loại bỏ những trường hợp có số câu đúng = 0 và trường hợp chặn dưới)
        a[j] = ((upper - lower) / group) * 2
    return a
