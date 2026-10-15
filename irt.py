import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, chi2
from scipy.integrate import quad
from scipy.special import hermite
import numpy.polynomial.hermite as herm


D = 1.702          # hằng số scaling về thang ogive
MISSING = -1
 
 
# Hàm tính độ phân biệt bằng point-biserial correlation
def cal_disc(r):
    r = np.nan_to_num(r, nan=1e-6) 
    return r/np.sqrt(1-r**2)
    
# Tính độ khó
def cal_diff(p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log((1 - p) / p)#np.clip(np.log((1 - p) / p), -6, 6)

# Hàm xác suất đúng theo mô hình IRT 2PL

def irt_probability(theta, a, b):
    """P(đúng) theo 2PL. Trả về mảng (K, J)."""
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    return expit(D * a * (theta - b))
 
 
def _prep(U):
    """Tách dữ liệu thành Um (thiếu -> 0) và M (1 nếu có dữ liệu)."""
    U = np.where(U == MISSING, 0.0, U)        # bỏ trống = sai, vẫn tính vào likelihood
    U = np.asarray(U, dtype=float)
    mask = np.isfinite(U) & (U != MISSING)
    Um = np.where(mask, U, 0.0)
    return Um, mask.astype(float)
 
 
def _gh_nodes(K):
    """Nút và log-trọng số Gauss-Hermite cho N(0,1)."""
    x, w = np.polynomial.hermite.hermgauss(K)
    theta = x * np.sqrt(2.0)
    w = w / np.sqrt(np.pi)
    return theta, np.log(np.maximum(w, 1e-300))
 
 
def _posterior(Um, M, a, b, theta, logw):
    """E-step: trả về W (N,K) = P(theta_k | u_i) và log-lik biên của từng người."""
    P = np.clip(irt_probability(theta, a, b), 1e-12, 1 - 1e-12)   # (K,J)
    logP, log1mP = np.log(P), np.log1p(-P)
    L = Um @ logP.T + (M - Um) @ log1mP.T + logw                  # (N,K)
    denom = logsumexp(L, axis=1)
    W = np.exp(L - denom[:, None])
    return W, denom
 
 
def marginal_loglik(U, a, b, K=41):
    """Log-likelihood biên của toàn bộ dữ liệu."""
    Um, M = _prep(U)
    theta, logw = _gh_nodes(K)
    return _posterior(Um, M, a, b, theta, logw)[1].sum()
 
 
# ----------------------------------------------------------------------
# M-step cho một item (Newton / Fisher scoring có backtracking)
# ----------------------------------------------------------------------
def _safe_b(a, c, eps=1e-6):
    """b = -c/a, tránh chia cho 0 khi a ~ 0 (a có thể âm nếu không chặn)."""
    a = np.asarray(a, float)
    a_safe = np.where(np.abs(a) < eps, np.where(a < 0, -eps, eps), a)
    return -np.asarray(c, float) / a_safe
 
 
def _mstep_item(n_k, r_k, theta, a, c, reg, a_bounds, n_newton=10):
    """
    M-step cho một item theo tham số (a, c) với c = -a*b, eta = D*(a*theta + c).
    Q lõm theo (a, c) nên Fisher scoring rất ổn định và KHÔNG cần chặn b.
    Penalty nhẹ: a ~ N(1, 1/reg), c ~ N(0, 1/reg).
    """
    def Q(a_, c_):
        P = np.clip(expit(D * (a_ * theta + c_)), 1e-12, 1 - 1e-12)
        return (np.sum(r_k * np.log(P) + (n_k - r_k) * np.log1p(-P))
                - 0.5 * reg * ((a_ - 1.0) ** 2 + c_ ** 2))
 
    q0 = Q(a, c)
    va, vc = D * theta, D * np.ones_like(theta)
    for _ in range(n_newton):
        P = np.clip(expit(D * (a * theta + c)), 1e-12, 1 - 1e-12)
        res = r_k - n_k * P
        g = np.array([np.sum(res * va) - reg * (a - 1.0),
                      np.sum(res * vc) - reg * c])
        w = n_k * P * (1.0 - P)
        I = np.array([[np.sum(w * va * va), np.sum(w * va * vc)],
                      [np.sum(w * va * vc), np.sum(w * vc * vc)]])
        I += (reg + 1e-8) * np.eye(2)
        try:
            delta = np.linalg.solve(I, g)
        except np.linalg.LinAlgError:
            break
 
        step, accepted = 1.0, False
        while step > 1e-4:
            a_new = np.clip(a + step * delta[0], *a_bounds)
            c_new = c + step * delta[1]
            q_new = Q(a_new, c_new)
            if q_new >= q0:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        change = max(abs(a_new - a), abs(c_new - c))
        a, c, q0 = a_new, c_new, q_new
        if change < 1e-8:
            break
    return a, c

 
# ----------------------------------------------------------------------
# MMLE - EM
# ----------------------------------------------------------------------
def mmle(U, name="MMLE", max_iter=300, K=41, tol=1e-6, tol_param=1e-4,
         reg=1e-2, a_bounds=(0.05, 4.0), b_bounds=None,
         a_init=None, b_init=None, verbose=True, return_ll=False):
    """
    MMLE-EM (Bock & Aitkin, 1981) cho 2PL.
    Ước lượng trong không gian (a, c), c = -a*b, rồi trả về b = -c/a.
    b_bounds=None (mặc định): b nằm trong (-inf, inf). Nếu truyền (lo, hi) thì chỉ cắt kết quả cuối.
    reg > 0 : penalty nhẹ (MAP). reg=0: ML thuần.
    """
    Um, M = _prep(U)
    N, J = Um.shape
    theta, logw = _gh_nodes(K)
 
    a = np.ones(J) if a_init is None else np.clip(np.asarray(a_init, float), *a_bounds)
    if b_init is None:
        cnt = np.maximum(M.sum(axis=0), 1)
        b0 = cal_diff(Um.sum(axis=0) / cnt)
        b0 = b0 - b0.mean()
    else:
        b0 = np.asarray(b_init, float)
    c = -a * b0
 
    if verbose:
        print(f"Start {name}: N={N}, J={J}, K={K}")
 
    prev_ll = -np.inf
    for it in range(1, max_iter + 1):
        b = -c / a
        W, denom = _posterior(Um, M, a, b, theta, logw)
        ll = denom.sum()
        n_kj = W.T @ M
        r_kj = W.T @ Um
 
        a_old, c_old = a.copy(), c.copy()
        for j in range(J):
            a[j], c[j] = _mstep_item(n_kj[:, j], r_kj[:, j], theta, a[j], c[j], reg, a_bounds)
 
        d_ll = ll - prev_ll
        d_par = max(np.max(np.abs(a - a_old)), np.max(np.abs(c - c_old)))
        if verbose and (it == 1 or it % 10 == 0):
            print(f"Iter {it}: LL = {ll:.4f}  dLL = {d_ll:.3e}  dParam = {d_par:.3e}")
        if abs(d_ll) < tol and d_par < tol_param:
            if verbose:
                print(f"Hội tụ tại iter {it}: LL = {ll:.4f}")
            break
        prev_ll = ll
 
    b = -c / a
    if b_bounds is not None:
        b = np.clip(b, *b_bounds)
    ll = marginal_loglik(U, a, b, K)
    return (a, b, ll) if return_ll else (a, b)
 
 
 
# ----------------------------------------------------------------------
# Ước lượng năng lực theta
# ----------------------------------------------------------------------
def _eap(Um, M, a, b, prior_mean=0.0, prior_std=1.0, n_points=121):
    grid = np.linspace(-6, 6, n_points)
    logw = norm.logpdf(grid, prior_mean, prior_std)
    W, _ = _posterior(Um, M, a, b, grid, logw)
    eap = W @ grid
    psd = np.sqrt(np.maximum(W @ grid ** 2 - eap ** 2, 0.0))
    return eap, psd
 
 
def _mle(Um, M, a, b):
    N = Um.shape[0]
    th, se = np.zeros(N), np.zeros(N)
    for i in range(N):
        v = M[i] > 0
        u, aa, bb = Um[i, v], a[v], b[v]
 
        def nll(t):
            P = np.clip(expit(D * aa * (t - bb)), 1e-12, 1 - 1e-12)
            return -np.sum(u * np.log(P) + (1 - u) * np.log1p(-P))
 
        res = minimize_scalar(nll, bounds=(-6, 6), method="bounded")
        t = float(res.x)
        P = expit(D * aa * (t - bb))
        info = np.sum((D * aa) ** 2 * P * (1 - P))
        th[i] = t
        se[i] = 1.0 / np.sqrt(info) if info > 0 else np.inf
    return th, se
 
 
def _split_params(item_params=None, a=None, b=None):
    """Nhận item_params dạng list[(a,b)], mảng (J,2), DataFrame(a,b) hoặc a, b riêng."""
    if item_params is not None:
        if isinstance(item_params, pd.DataFrame):
            return item_params["a"].to_numpy(float), item_params["b"].to_numpy(float)
        arr = np.asarray(item_params, dtype=float)
        return arr[:, 0], arr[:, 1]
    return np.asarray(a, float), np.asarray(b, float)
 
 
def theta_estimate(responses, item_params=None, a=None, b=None, method="MLE",
                   prior_mean=0.0, prior_std=1.0, return_se=False):
    """
    Tương thích code cũ: theta_estimate(U, item_params=list(zip(a, b))) -> list theta (MLE).
      method = "MLE": cực đại likelihood (theta bị chặn trong [-6, 6])
      method = "EAP": trung bình posterior, prior N(prior_mean, prior_std^2)
    return_se=True -> trả về (theta, se). Dữ liệu thiếu (-1) được bỏ qua đúng cách.
    Lưu ý: MLE không xác định với người đúng/sai hết, hãy lọc trước (như run_irt_series).
    """
    #Um, M = _prep(np.atleast_2d(np.asarray(responses, dtype=float)))
    R = np.atleast_2d(np.asarray(responses, dtype=float))
    Um, M = _prep(np.where(R == MISSING, 0.0, R))
    a, b = _split_params(item_params, a, b)
    if method.upper() == "EAP":
        th, se = _eap(Um, M, a, b, prior_mean, prior_std)
    else:
        th, se = _mle(Um, M, a, b)
    return (th, se) if return_se else [float(t) for t in th]
 
 
# ----------------------------------------------------------------------
# Sai số chuẩn của tham số câu hỏi (Hessian số của log-lik biên)
# ----------------------------------------------------------------------
def item_se_observed(U, a, b, K=41, h=1e-2):
    """
    SE của (a_j, b_j) từ thông tin quan sát biên (observed marginal information),
    tính bằng sai phân hữu hạn trên log-lik biên, các item khác giữ cố định.
    Trả về DataFrame cột se_a, se_b.
    """
    Um, M = _prep(U)
    a, b = np.asarray(a, float), np.asarray(b, float)
    theta, logw = _gh_nodes(K)
    P = np.clip(irt_probability(theta, a, b), 1e-12, 1 - 1e-12)
    logP, log1mP = np.log(P), np.log1p(-P)
    L = Um @ logP.T + (M - Um) @ log1mP.T + logw
 
    out = []
    for j in range(len(a)):
        uj, wj = Um[:, [j]], (M[:, j] - Um[:, j])[:, None]
        base = L - uj * logP[:, j][None, :] - wj * log1mP[:, j][None, :]
 
        def f(aj, bj):
            p = np.clip(expit(D * aj * (theta - bj)), 1e-12, 1 - 1e-12)
            Lj = base + uj * np.log(p)[None, :] + wj * np.log1p(-p)[None, :]
            return logsumexp(Lj, axis=1).sum()
 
        ha, hb = min(h, 0.5 * a[j]), h
        f0 = f(a[j], b[j])
        Haa = (f(a[j] + ha, b[j]) - 2 * f0 + f(a[j] - ha, b[j])) / ha ** 2
        Hbb = (f(a[j], b[j] + hb) - 2 * f0 + f(a[j], b[j] - hb)) / hb ** 2
        Hab = (f(a[j] + ha, b[j] + hb) - f(a[j] + ha, b[j] - hb)
               - f(a[j] - ha, b[j] + hb) + f(a[j] - ha, b[j] - hb)) / (4 * ha * hb)
        info = -np.array([[Haa, Hab], [Hab, Hbb]])
        try:
            cov = np.linalg.inv(info)
            d = np.diag(cov)
            se = np.where(d > 0, np.sqrt(np.abs(d)), np.inf)
        except np.linalg.LinAlgError:
            se = np.array([np.inf, np.inf])
        out.append(se)
    return pd.DataFrame(out, columns=["se_a", "se_b"])
 
 
# ----------------------------------------------------------------------
# Kiểm định độ phù hợp item (Yen's Q1, chia nhóm theo phân vị theta)
# ----------------------------------------------------------------------
def chi_square(df, item_param, theta_col="Theta", num_bins=10):
    n_groups = num_bins
    """
    df         : DataFrame chứa cột theta_col và các cột câu hỏi (0/1/-1)
    item_param : DataFrame, index = tên câu, cột 'a', 'b'
    """
    theta_all = df[theta_col].to_numpy(float)
    rows = []
    for item in item_param.index:
        if item not in df.columns:
            continue
        u = df[item].to_numpy(float)
        valid = np.isfinite(u) & (u != MISSING)
        valid &= np.isfinite(theta_all)
        t, x = theta_all[valid], u[valid]
        a = float(item_param.loc[item, "a"])
        b = float(item_param.loc[item, "b"])
 
        if len(x) < n_groups * 5:
            rows.append([item, a, b, np.nan, np.nan, np.nan])
            continue
 
        p = expit(D * a * (t - b))
        order = np.argsort(t, kind="stable")
        stat, used = 0.0, 0
        for g in np.array_split(order, n_groups):
            ng = len(g)
            O, E = x[g].sum(), p[g].sum()
            var = E * (1.0 - E / ng)          # = ng * pbar * (1 - pbar)
            if var <= 1e-9:
                continue
            stat += (O - E) ** 2 / var
            used += 1
        dof = used - 2
        pval = chi2.sf(stat, dof) if dof > 0 else np.nan
        rows.append([item, a, b, stat, dof, round(pval, 4) if dof > 0 else np.nan])
 
    return pd.DataFrame(rows, columns=["Item", "a", "b", "Chi2", "df", "p_value"]).set_index("Item")
 
 
# ----------------------------------------------------------------------
# Điểm thực quy đổi theo đường đặc trưng đề thi (TCC)
# ----------------------------------------------------------------------
def true_score(theta, raw, data: pd.Series, item_params: pd.DataFrame, scale_max=300):
    # if not np.isfinite(theta) or not np.isfinite(raw):
    #     return np.nan
    items = [c for c in data.index if c in item_params.index]      # toàn bộ câu của đề
    n = len(items)
    # if n == 0 or raw <= 0:
    #     return 0
    # answered = [c for c in items if np.isfinite(data[c]) and data[c] != MISSING]
    # if raw == n and len(answered) == n:
    #     return int(scale_max)
    if n == 0 or not np.isfinite(raw):
            return np.nan
    if raw <= 0:
            return 0                       # sai hết hoặc bỏ trống hết
    if raw >= n:
            return int(scale_max)          # đúng hết toàn bộ câu tính điểm
    if not np.isfinite(theta):
            return np.nan
    a = item_params.loc[items, "a"].to_numpy(float)
    b = item_params.loc[items, "b"].to_numpy(float)
    tcc = irt_probability(float(theta), a, b).sum()
    return int(np.clip(np.round(tcc / n * scale_max), 0, scale_max))
    
 
# ----------------------------------------------------------------------
# Pipeline gọn
# ----------------------------------------------------------------------
def fit_2pl(df_items: pd.DataFrame, K=41, reg=1e-2, verbose=True, theta_method="EAP",
            a_init=None, b_init=None):
    """
    df_items: DataFrame (N x J) chỉ chứa các cột câu hỏi 0/1/-1.
    Trả về (item_params, person_params).
    """
    U = df_items.to_numpy(float)
    U = np.where(U == MISSING, 0.0, U)   # bỏ trống = sai
    a, b, ll = mmle(U, K=K, reg=reg, verbose=verbose, return_ll=True,
                    a_init=a_init, b_init=b_init)
    se = item_se_observed(U, a, b, K=K)
    items = pd.DataFrame({"a": a, "b": b, "se_a": se["se_a"].to_numpy(), "se_b": se["se_b"].to_numpy()},
                         index=df_items.columns)
    th, psd = theta_estimate(U, a=a, b=b, method=theta_method, return_se=True)
    persons = pd.DataFrame({"Theta": th, "SE": psd}, index=df_items.index)
    if verbose:
        print(f"Marginal LL = {ll:.2f}, AIC = {-2 * ll + 2 * 2 * len(a):.2f}")
    return items, persons
 
# ----------------------------------------------------------------------
# Các hàm tương thích với irt.py cũ (giữ nguyên tên và chữ ký)
# ----------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or np.trapz
 
 
def neg_log_likelihood(theta, responses, item_params):
    """-log L(theta | responses) cho 1 người; bỏ qua câu thiếu (-1). Trả về float."""
    a, b = _split_params(item_params)
    u = np.asarray(responses, dtype=float).ravel()
    u = np.where(u == MISSING, 0.0, u)
    v = np.isfinite(u) & (u != MISSING)
    t = float(np.asarray(theta).ravel()[0])
    P = np.clip(expit(D * a[v] * (t - b[v])), 1e-12, 1 - 1e-12)
    return float(-np.sum(u[v] * np.log(P) + (1 - u[v]) * np.log1p(-P)))
 
 
def log_likelihood(U, a_list, b_list, theta_grid=None, gh_weights=None, eps=1e-12):
    """Log-lik biên. Nếu theta_grid/gh_weights được truyền thì dùng đúng lưới đó."""
    Um, M = _prep(U)
    if theta_grid is None or gh_weights is None:
        theta_grid, logw = _gh_nodes(41)
    else:
        theta_grid = np.asarray(theta_grid, float)
        logw = np.log(np.maximum(np.asarray(gh_weights, float), 1e-300))
    return _posterior(Um, M, np.asarray(a_list, float), np.asarray(b_list, float),
                      theta_grid, logw)[1].sum()
 
 
def posterior(theta_grid, responses, item_params, prior_mean=0, prior_std=1, eps=1e-12):
    """Posterior chuẩn hoá (tích phân = 1) của theta trên theta_grid cho 1 người."""
    grid = np.asarray(theta_grid, float)
    a, b = _split_params(item_params)
    u = np.asarray(responses, dtype=float).ravel()
    u = np.where(u == MISSING, 0.0, u)
    v = np.isfinite(u) & (u != MISSING)
    P = np.clip(irt_probability(grid, a[v], b[v]), eps, 1 - eps)
    logpost = (norm.logpdf(grid, prior_mean, prior_std)
               + u[v] @ np.log(P).T + (1 - u[v]) @ np.log1p(-P).T)
    post = np.exp(logpost - logpost.max())
    return post / _trapz(post, grid)
 
 
def ability_se(responses, item_params, theta_estimate=None, prior_mean=0, prior_std=1,
               theta_min=-6, theta_max=6, num_points=1001):
    """Posterior SD của theta (tính quanh trung bình posterior; theta_estimate chỉ để tương thích)."""
    grid = np.linspace(theta_min, theta_max, num_points)
    post = posterior(grid, responses, item_params, prior_mean, prior_std)
    m = _trapz(grid * post, grid)
    return float(np.sqrt(max(_trapz((grid - m) ** 2 * post, grid), 0.0)))
 
 
def all_ability_se(responses_matrix, item_params, theta_estimates=None, prior_mean=0, prior_std=1,
                   theta_min=-6, theta_max=6, num_points=1001):
    return np.array([ability_se(r, item_params, None, prior_mean, prior_std,
                                theta_min, theta_max, num_points)
                     for r in np.asarray(responses_matrix)])
 
 
def item_se(a, b, prior_mean=0, prior_std=1, theta_min=-6, theta_max=6,
            num_quad=50, reg=1e-6, N=1):
    """
    SE xấp xỉ từ thông tin Fisher (complete-data) của 1 item, nhân với N người.
    Nhớ truyền N = số thí sinh (mặc định N=1 giống code cũ nhưng SE sẽ bị phóng đại).
    Muốn SE chính xác hơn dùng item_se_observed(U, a, b).
    """
    nodes, weights = np.polynomial.hermite.hermgauss(num_quad)
    th = prior_mean + np.sqrt(2) * prior_std * nodes
    w = weights / np.sqrt(np.pi)
    p = np.clip(expit(D * a * (th - b)), 1e-12, 1 - 1e-12)
    q = 1 - p
    va, vb = D * (th - b), -D * a * np.ones_like(th)
    pq = p * q
    I = N * np.array([[np.sum(w * pq * va * va), np.sum(w * pq * va * vb)],
                      [np.sum(w * pq * va * vb), np.sum(w * pq * vb * vb)]])
    try:
        d = np.diag(np.linalg.inv(I + reg * np.eye(2)))
        return tuple(np.where(d > 0, np.sqrt(np.abs(d)), np.inf))
    except np.linalg.LinAlgError:
        return np.inf, np.inf
 
 
def all_item_se(item_params, prior_mean=0, prior_std=1, theta_min=-6, theta_max=6,
                num_quad=50, reg=1e-6, N=1, U=None):
    """Nếu truyền U (ma trận đáp án) thì dùng SE quan sát biên; ngược lại dùng item_se(N=N)."""
    a, b = _split_params(item_params)
    if U is not None:
        return item_se_observed(U, a, b).to_numpy()
    return np.array([item_se(x, y, prior_mean, prior_std, theta_min, theta_max,
                             num_quad, reg, N) for x, y in zip(a, b)])
 
 
# ----------------------------------------------------------------------
# Demo: mô phỏng và kiểm tra khả năng khôi phục tham số
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, J = 3000, 30
    a_true = rng.lognormal(0.0, 0.3, J)
    b_true = rng.normal(0, 1, J)
    th_true = rng.normal(0, 1, N)
    P = irt_probability(th_true, a_true, b_true)
    U = (rng.random((N, J)) < P).astype(float)
    U[rng.random((N, J)) < 0.05] = MISSING                # 5% dữ liệu thiếu
 
    cols = [f"Cau{j + 1}" for j in range(J)]
    df = pd.DataFrame(U, columns=cols)
 
    items, persons = fit_2pl(df)
    print("\nCorr(a_hat, a_true) =", np.corrcoef(items["a"], a_true)[0, 1].round(3))
    print("Corr(b_hat, b_true) =", np.corrcoef(items["b"], b_true)[0, 1].round(3))
    print("RMSE a =", np.sqrt(np.mean((items["a"] - a_true) ** 2)).round(3))
    print("RMSE b =", np.sqrt(np.mean((items["b"] - b_true) ** 2)).round(3))
    print("Corr(theta_hat, theta_true) =", np.corrcoef(persons["Theta"], th_true)[0, 1].round(3))
    print("\n", items.head())
 
    fit = chi_square(pd.concat([df, persons[["Theta"]]], axis=1), items[["a", "b"]])
    print("\n", fit)
 
    row = df.iloc[0]
    raw = int((row == 1).sum())
    print("\nTrue score thí sinh 0:", true_score(persons.loc[0, "Theta"], raw, row, items))
 