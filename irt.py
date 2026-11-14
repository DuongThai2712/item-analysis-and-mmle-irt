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

# ----------------------------------------------------------------------
# Mô hình 1PL / 2PL / 3PL / 4PL
#   P(theta) = c + (d - c) * expit(D * a * (theta - b))
#   1PL: một a chung cho mọi câu, c = 0, d = 1     2PL: a, b      3PL: a, b, c      4PL: a, b, c, d
# Tham số không ước lượng nhận giá trị mặc định: a = 1, c = 0, d = 1.
# ----------------------------------------------------------------------
MODELS = (1, 2, 3, 4)
A_DEFAULT, C_DEFAULT, D_DEFAULT = 1.0, 0.0, 1.0
FREE = {1: ("b",), 2: ("a", "b"), 3: ("a", "b", "c"), 4: ("a", "b", "c", "d")}   # tham số riêng từng câu


def _row(x, default=None):
    """None -> None; còn lại -> mảng (1, J) (NaN thay bằng default nếu có)."""
    if x is None:
        return None
    x = np.asarray(x, dtype=float).reshape(1, -1)
    return np.where(np.isfinite(x), x, default) if default is not None else x


def cal_iif(a, b, theta, use_D=True, c=None, d=None):
    """
    Hàm thông tin câu hỏi. use_D=True: I đúng với mô hình dùng D = 1.702.
      2PL: I = (D a)^2 P Q
      4PL: I = (D a)^2 (d - c)^2 psi^2 (1 - psi)^2 / (P (1 - P)),  psi = expit(D a (theta - b))
    """
    psi = irt_probability(theta, a, b).ravel()
    k = (D * a) ** 2 if use_D else a ** 2
    if c is None and d is None:
        return k * psi * (1 - psi)
    c = C_DEFAULT if c is None or not np.isfinite(c) else c
    d = D_DEFAULT if d is None or not np.isfinite(d) else d
    P = np.clip(c + (d - c) * psi, 1e-12, 1 - 1e-12)
    return k * (d - c) ** 2 * (psi * (1 - psi)) ** 2 / (P * (1 - P))


def irt_probability(theta, a, b, c=None, d=None):
    """P(đúng) theo 4PL (c = 0, d = 1 nếu không truyền -> 2PL). Trả về mảng (K, J)."""
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    psi = expit(D * a * (theta - b))
    if c is None and d is None:
        return psi
    c = _row(c, C_DEFAULT) if c is not None else C_DEFAULT
    d = _row(d, D_DEFAULT) if d is not None else D_DEFAULT
    return c + (d - c) * psi


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


def _posterior(Um, M, a, b, theta, logw, c=None, d=None):
    """E-step: trả về W (N,K) = P(theta_k | u_i) và log-lik biên của từng người."""
    P = np.clip(irt_probability(theta, a, b, c, d), 1e-12, 1 - 1e-12)   # (K,J)
    logP, log1mP = np.log(P), np.log1p(-P)
    L = Um @ logP.T + (M - Um) @ log1mP.T + logw                       # (N,K)
    denom = logsumexp(L, axis=1)
    W = np.exp(L - denom[:, None])
    return W, denom


def marginal_loglik(U, a, b, K=41, c=None, d=None):
    """Log-likelihood biên của toàn bộ dữ liệu."""
    Um, M = _prep(U)
    theta, logw = _gh_nodes(K)
    return _posterior(Um, M, a, b, theta, logw, c, d)[1].sum()


# ----------------------------------------------------------------------
# M-step cho một item (Newton / Fisher scoring có backtracking) - 1PL, 2PL
# ----------------------------------------------------------------------
def _safe_b(a, c, eps=1e-6):
    """b = -c/a, tránh chia cho 0 khi a ~ 0 (a có thể âm nếu không chặn)."""
    a = np.asarray(a, float)
    a_safe = np.where(np.abs(a) < eps, np.where(a < 0, -eps, eps), a)
    return -np.asarray(c, float) / a_safe


def _clip_c(a, c, b_bounds):
    """Giữ b = -c/a trong b_bounds (a > 0): c trong [-a*b_hi, -a*b_lo]."""
    if b_bounds is None:
        return c
    lo, hi = b_bounds
    return float(np.clip(c, -a * hi if hi is not None else -np.inf, -a * lo if lo is not None else np.inf))


def _mstep_item(n_k, r_k, theta, a, c, reg, a_bounds, n_newton=10, b_bounds=None, fix_a=False):
    """
    M-step cho một item theo tham số (a, c) với c = -a*b, eta = D*(a*theta + c).
    Q lõm theo (a, c) nên Fisher scoring rất ổn định. Penalty nhẹ: a ~ N(1, 1/reg), c ~ N(0, 1/reg).
    b_bounds: chiếu c để b nằm trong khoảng cho phép. fix_a=True (1PL): chỉ cập nhật c, a giữ nguyên.
    """
    def Q(a_, c_):
        P = np.clip(expit(D * (a_ * theta + c_)), 1e-12, 1 - 1e-12)
        return (np.sum(r_k * np.log(P) + (n_k - r_k) * np.log1p(-P))
                - 0.5 * reg * ((0.0 if fix_a else (a_ - 1.0) ** 2) + c_ ** 2))

    c = _clip_c(a, c, b_bounds)
    q0 = Q(a, c)
    va, vc = D * theta, D * np.ones_like(theta)
    for _ in range(n_newton):
        P = np.clip(expit(D * (a * theta + c)), 1e-12, 1 - 1e-12)
        res = r_k - n_k * P
        w = n_k * P * (1.0 - P)
        if fix_a:
            gc = np.sum(res * vc) - reg * c
            delta = np.array([0.0, gc / (np.sum(w * vc * vc) + reg + 1e-8)])
        else:
            g = np.array([np.sum(res * va) - reg * (a - 1.0),
                          np.sum(res * vc) - reg * c])
            I = np.array([[np.sum(w * va * va), np.sum(w * va * vc)],
                          [np.sum(w * va * vc), np.sum(w * vc * vc)]])
            I += (reg + 1e-8) * np.eye(2)
            try:
                delta = np.linalg.solve(I, g)
            except np.linalg.LinAlgError:
                break

        step, accepted = 1.0, False
        while step > 1e-4:
            a_new = a if fix_a else np.clip(a + step * delta[0], *a_bounds)
            c_new = _clip_c(a_new, c + step * delta[1], b_bounds)
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


def _common_a(n_kj, r_kj, theta, b, a0, reg, a_bounds):
    """1PL: cập nhật a chung (b của từng câu giữ cố định) bằng tối ưu một chiều."""
    def nq(a_):
        P = np.clip(expit(D * a_ * (theta[:, None] - b[None, :])), 1e-12, 1 - 1e-12)
        return -(np.sum(r_kj * np.log(P) + (n_kj - r_kj) * np.log1p(-P)) - 0.5 * reg * (a_ - 1.0) ** 2)
    r = minimize_scalar(nq, bounds=a_bounds, method="bounded", options={"xatol": 1e-6})
    return float(r.x) if r.fun <= nq(a0) else float(a0)


# ----------------------------------------------------------------------
# M-step 3PL / 4PL (L-BFGS-B có chặn, prior Beta cho c và d)
# ----------------------------------------------------------------------
def _mstep_item_4pl(n_k, r_k, theta, x0, free, bounds, reg, c_prior, d_prior, maxiter=40):
    """
    x0   : (a, b, c, d) hiện tại;  free: tên các tham số được ước lượng (vd ('a','b','c')).
    Q    = sum_k r_k log P + (n_k - r_k) log(1-P) + log prior
    prior: a ~ N(1, 1/reg), b ~ N(0, 1/reg), c ~ Beta(c_prior), d ~ Beta(d_prior)  (MAP như BILOG).
    """
    names = ("a", "b", "c", "d")
    idx = [names.index(f) for f in free]
    full = np.array(x0, float)
    (ac, bc), (ad, bd) = c_prior, d_prior

    def unpack(x):
        p = full.copy()
        p[idx] = x
        return p

    def nll_grad(x):
        a, b, c, d = unpack(x)
        psi = expit(D * a * (theta - b))
        P = np.clip(c + (d - c) * psi, 1e-12, 1 - 1e-12)
        ll = np.sum(r_k * np.log(P) + (n_k - r_k) * np.log1p(-P))
        g_P = (r_k - n_k * P) / (P * (1 - P))
        s = (d - c) * psi * (1 - psi)
        grads = {"a": np.sum(g_P * s * D * (theta - b)), "b": np.sum(g_P * -s * D * a),
                 "c": np.sum(g_P * (1 - psi)), "d": np.sum(g_P * psi)}
        # prior
        ll -= 0.5 * reg * ((a - 1.0) ** 2 + b ** 2)
        grads["a"] -= reg * (a - 1.0)
        grads["b"] -= reg * b
        if "c" in free:
            ll += (ac - 1) * np.log(c) + (bc - 1) * np.log1p(-c)
            grads["c"] += (ac - 1) / c - (bc - 1) / (1 - c)
        if "d" in free:
            ll += (ad - 1) * np.log(d) + (bd - 1) * np.log1p(-d)
            grads["d"] += (ad - 1) / d - (bd - 1) / (1 - d)
        return -ll, -np.array([grads[f] for f in free])

    x = np.clip(full[idx], [-np.inf if bb[0] is None else bb[0] for bb in bounds],
                [np.inf if bb[1] is None else bb[1] for bb in bounds])
    r = minimize(nll_grad, x, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
    return unpack(r.x if np.all(np.isfinite(r.x)) else x)


def _norm_bounds(bounds, default):
    """(lo, hi) với None/NaN -> mặc định; trả về tuple float hoặc None (không chặn)."""
    if bounds is None:
        return default
    lo, hi = bounds
    lo = default[0] if lo is None or not np.isfinite(lo) else float(lo)
    hi = default[1] if hi is None or not np.isfinite(hi) else float(hi)
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"Giới hạn không hợp lệ: ({lo}, {hi})")
    return lo, hi


# ----------------------------------------------------------------------
# MMLE - EM
# ----------------------------------------------------------------------
def mmle(U, name="MMLE", max_iter=300, K=41, tol=1e-6, tol_param=1e-4,
         reg=1e-2, a_bounds=(0.05, 4.0), b_bounds=None,
         a_init=None, b_init=None, verbose=True, return_ll=False,
         model=2, c_bounds=(0.0, 0.5), d_bounds=(0.5, 1.0),
         c_prior=(5.0, 17.0), d_prior=(17.0, 2.0), c_init=None, d_init=None, return_cd=False):
    """
    MMLE-EM (Bock & Aitkin, 1981) cho 1PL / 2PL / 3PL / 4PL.
      model=1 : một a chung cho mọi câu (ước lượng), b riêng; c = 0, d = 1
      model=2 : a, b riêng từng câu (ước lượng trong không gian (a, c = -a*b) như bản cũ)
      model=3 : a, b, c (đoán mò, prior Beta(c_prior)); d = 1
      model=4 : a, b, c, d (trượt, prior Beta(d_prior))
    a_bounds, b_bounds : giới hạn ước lượng (b_bounds=None: b không bị chặn).
    reg > 0 : penalty nhẹ (MAP). reg=0: ML thuần.
    Trả về (a, b) [+ ll]; return_cd=True -> (a, b, c, d) [+ ll].
    """
    model = int(model)
    if model not in MODELS:
        raise ValueError("model phải là 1, 2, 3 hoặc 4")
    a_bounds = _norm_bounds(a_bounds, (0.05, 4.0))
    a_bounds = (max(a_bounds[0], 1e-3), a_bounds[1])
    b_bounds = None if b_bounds is None else _norm_bounds(b_bounds, (None, None))
    if b_bounds == (None, None):
        b_bounds = None
    Um, M = _prep(U)
    N, J = Um.shape
    theta, logw = _gh_nodes(K)

    if verbose:
        print(f"Start {name} ({model}PL): N={N}, J={J}, K={K}")

    if model >= 3:
        # khởi đầu ấm bằng 2PL rồi ước lượng thêm c (và d)
        a, b = mmle(U, name=name + "/init", max_iter=min(max_iter, 150), K=K, tol=1e-4, tol_param=1e-3,
                    reg=reg, a_bounds=a_bounds, b_bounds=b_bounds, a_init=a_init, b_init=b_init, verbose=False)
        cb, db_ = _norm_bounds(c_bounds, (0.0, 0.5)), _norm_bounds(d_bounds, (0.5, 1.0))
        cb = (max(cb[0], 1e-6), cb[1])
        db_ = (db_[0], min(db_[1], 1 - 1e-6))
        c = np.full(J, np.clip(0.15, *cb)) if c_init is None else np.clip(np.nan_to_num(np.asarray(c_init, float), nan=0.15), *cb)
        d = np.ones(J) if model == 3 else (np.full(J, np.clip(0.97, *db_)) if d_init is None
                                            else np.clip(np.nan_to_num(np.asarray(d_init, float), nan=0.97), *db_))
        free = FREE[model]
        bl = b_bounds or (None, None)
        bnds = {"a": a_bounds, "b": bl, "c": cb, "d": db_}
        bounds = [bnds[f] for f in free]
        prev_ll = -np.inf
        for it in range(1, max_iter + 1):
            W, denom = _posterior(Um, M, a, b, theta, logw, c, d)
            ll = denom.sum()
            n_kj, r_kj = W.T @ M, W.T @ Um
            old = np.c_[a, b, c, d].copy()
            for j in range(J):
                a[j], b[j], c[j], d[j] = _mstep_item_4pl(n_kj[:, j], r_kj[:, j], theta, old[j], free, bounds,
                                                         reg, c_prior, d_prior)
            d_ll = ll - prev_ll
            d_par = float(np.max(np.abs(np.c_[a, b, c, d] - old)))
            if verbose and (it == 1 or it % 10 == 0):
                print(f"Iter {it}: LL = {ll:.4f}  dLL = {d_ll:.3e}  dParam = {d_par:.3e}")
            if abs(d_ll) < tol and d_par < tol_param:
                if verbose:
                    print(f"Hội tụ tại iter {it}: LL = {ll:.4f}")
                break
            prev_ll = ll
        ll = marginal_loglik(U, a, b, K, c, d)
        out = (a, b, c, d) if return_cd else (a, b)
        return out + (ll,) if return_ll else out

    # ---- 1PL / 2PL ----
    if a_init is None:
        a = np.ones(J)
    else:
        a = np.clip(np.nan_to_num(np.asarray(a_init, float), nan=1.0), *a_bounds)
    if model == 1:
        a = np.full(J, float(np.clip(np.nanmean(a), *a_bounds)))
    if b_init is None:
        cnt = np.maximum(M.sum(axis=0), 1)
        b0 = cal_diff(Um.sum(axis=0) / cnt)
        b0 = b0 - b0.mean()
    else:
        b0 = np.nan_to_num(np.asarray(b_init, float), nan=0.0)
    if b_bounds is not None:
        b0 = np.clip(b0, *[v if v is not None else s * np.inf for v, s in zip(b_bounds, (-1, 1))])
    c = -a * b0

    prev_ll = -np.inf
    for it in range(1, max_iter + 1):
        b = -c / a
        W, denom = _posterior(Um, M, a, b, theta, logw)
        ll = denom.sum()
        n_kj = W.T @ M
        r_kj = W.T @ Um

        a_old, c_old = a.copy(), c.copy()
        for j in range(J):
            a[j], c[j] = _mstep_item(n_kj[:, j], r_kj[:, j], theta, a[j], c[j], reg, a_bounds,
                                     b_bounds=b_bounds, fix_a=(model == 1))
        if model == 1:                                           # a chung, b giữ nguyên
            b_cur = -c / a
            a_new = _common_a(n_kj, r_kj, theta, b_cur, a[0], reg, a_bounds)
            a = np.full(J, a_new)
            c = -a * b_cur

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
        b = np.clip(b, *[v if v is not None else s * np.inf for v, s in zip(b_bounds, (-1, 1))])
    ll = marginal_loglik(U, a, b, K)
    out = (a, b, np.zeros(J), np.ones(J)) if return_cd else (a, b)
    return out + (ll,) if return_ll else out
 
 
 
# ----------------------------------------------------------------------
# Ước lượng năng lực theta
# ----------------------------------------------------------------------
def _eap(Um, M, a, b, prior_mean=0.0, prior_std=1.0, n_points=121, c=None, d=None):
    grid = np.linspace(-6, 6, n_points)
    logw = norm.logpdf(grid, prior_mean, prior_std)
    W, _ = _posterior(Um, M, a, b, grid, logw, c, d)
    eap = W @ grid
    psd = np.sqrt(np.maximum(W @ grid ** 2 - eap ** 2, 0.0))
    return eap, psd


def _mle(Um, M, a, b, c=None, d=None):
    N, J = Um.shape
    c = np.zeros(J) if c is None else np.nan_to_num(np.asarray(c, float), nan=C_DEFAULT)
    d = np.ones(J) if d is None else np.nan_to_num(np.asarray(d, float), nan=D_DEFAULT)
    th, se = np.zeros(N), np.zeros(N)
    for i in range(N):
        v = M[i] > 0
        u, aa, bb, cc, dd = Um[i, v], a[v], b[v], c[v], d[v]

        def nll(t):
            P = np.clip(cc + (dd - cc) * expit(D * aa * (t - bb)), 1e-12, 1 - 1e-12)
            return -np.sum(u * np.log(P) + (1 - u) * np.log1p(-P))

        res = minimize_scalar(nll, bounds=(-6, 6), method="bounded")
        t = float(res.x)
        psi = expit(D * aa * (t - bb))
        P = np.clip(cc + (dd - cc) * psi, 1e-12, 1 - 1e-12)
        info = np.sum((D * aa) ** 2 * (dd - cc) ** 2 * (psi * (1 - psi)) ** 2 / (P * (1 - P)))
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


def _params4(item_params=None, a=None, b=None, c=None, d=None):
    """
    Như _split_params nhưng trả về (a, b, c, d). item_params: DataFrame có thể có cột c, d;
    mảng (J, 2..4). Thiếu c/d hoặc NaN -> mặc định (c = 0, d = 1); trả về None nếu hoàn toàn mặc định.
    """
    if item_params is not None:
        if isinstance(item_params, pd.DataFrame):
            a, b = item_params["a"].to_numpy(float), item_params["b"].to_numpy(float)
            c = item_params["c"].to_numpy(float) if "c" in item_params.columns else None
            d = item_params["d"].to_numpy(float) if "d" in item_params.columns else None
        else:
            arr = np.asarray(item_params, dtype=float)
            a, b = arr[:, 0], arr[:, 1]
            c = arr[:, 2] if arr.shape[1] > 2 else None
            d = arr[:, 3] if arr.shape[1] > 3 else None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if c is not None:
        c = np.nan_to_num(np.asarray(c, float), nan=C_DEFAULT)
        if np.all(c == 0):
            c = None
    if d is not None:
        d = np.nan_to_num(np.asarray(d, float), nan=D_DEFAULT)
        if np.all(d == 1):
            d = None
    return a, b, c, d


def theta_estimate(responses, item_params=None, a=None, b=None, method="MLE",
                   prior_mean=0.0, prior_std=1.0, return_se=False, c=None, d=None):
    """
    Tương thích code cũ: theta_estimate(U, item_params=list(zip(a, b))) -> list theta (MLE).
      method = "MLE": cực đại likelihood (theta bị chặn trong [-6, 6])
      method = "EAP": trung bình posterior, prior N(prior_mean, prior_std^2)
    c, d (hoặc cột c, d trong item_params): tham số 3PL/4PL; mặc định 0 và 1.
    return_se=True -> trả về (theta, se). Dữ liệu thiếu (-1) được bỏ qua đúng cách.
    Lưu ý: MLE không xác định với người đúng/sai hết, hãy lọc trước (như run_irt_series).
    """
    R = np.atleast_2d(np.asarray(responses, dtype=float))
    Um, M = _prep(np.where(R == MISSING, 0.0, R))
    a, b, c, d = _params4(item_params, a, b, c, d)
    if method.upper() == "EAP":
        th, se = _eap(Um, M, a, b, prior_mean, prior_std, c=c, d=d)
    else:
        th, se = _mle(Um, M, a, b, c, d)
    return (th, se) if return_se else [float(t) for t in th]


# ----------------------------------------------------------------------
# Sai số chuẩn của tham số câu hỏi (Hessian số của log-lik biên)
# ----------------------------------------------------------------------
def item_se_observed(U, a, b, K=41, h=1e-2, c=None, d=None, model=2):
    """
    SE của tham số tự do của từng câu (model 1: b; 2: a, b; 3: a, b, c; 4: a, b, c, d)
    từ thông tin quan sát biên, tính bằng sai phân hữu hạn trên log-lik biên, các item khác giữ cố định.
    Trả về DataFrame cột se_a, se_b, se_c, se_d (NaN với tham số không ước lượng riêng).
    """
    Um, M = _prep(U)
    a, b = np.asarray(a, float), np.asarray(b, float)
    J = len(a)
    c = np.zeros(J) if c is None else np.nan_to_num(np.asarray(c, float), nan=C_DEFAULT)
    d = np.ones(J) if d is None else np.nan_to_num(np.asarray(d, float), nan=D_DEFAULT)
    theta, logw = _gh_nodes(K)
    P = np.clip(irt_probability(theta, a, b, c, d), 1e-12, 1 - 1e-12)
    logP, log1mP = np.log(P), np.log1p(-P)
    L = Um @ logP.T + (M - Um) @ log1mP.T + logw
    names = ("a", "b", "c", "d")
    free = [names.index(f) for f in FREE[int(model)]]

    out = []
    for j in range(J):
        uj, wj = Um[:, [j]], (M[:, j] - Um[:, j])[:, None]
        base = L - uj * logP[:, j][None, :] - wj * log1mP[:, j][None, :]
        x0 = np.array([a[j], b[j], c[j], d[j]])

        def f(x):
            p = np.clip(x[2] + (x[3] - x[2]) * expit(D * x[0] * (theta - x[1])), 1e-12, 1 - 1e-12)
            Lj = base + uj * np.log(p)[None, :] + wj * np.log1p(-p)[None, :]
            return logsumexp(Lj, axis=1).sum()

        hs = np.array([min(h, 0.5 * a[j]), h, max(min(h, 0.5 * c[j]), 1e-4), max(min(h, 0.5 * (1 - d[j])), 1e-4)])
        k = len(free)
        f0 = f(x0)

        def fx(*steps):
            x = x0.copy()
            for i, s in steps:
                x[free[i]] += s * hs[free[i]]
            return f(x)

        H = np.zeros((k, k))
        for i in range(k):
            H[i, i] = (fx((i, 1)) - 2 * f0 + fx((i, -1))) / hs[free[i]] ** 2
            for m in range(i):
                H[i, m] = H[m, i] = (fx((i, 1), (m, 1)) - fx((i, 1), (m, -1))
                                     - fx((i, -1), (m, 1)) + fx((i, -1), (m, -1))) / (4 * hs[free[i]] * hs[free[m]])
        se = np.full(4, np.nan)
        try:
            dg = np.diag(np.linalg.inv(-H))
            se[free] = np.where(dg > 0, np.sqrt(np.abs(dg)), np.inf)
        except np.linalg.LinAlgError:
            se[free] = np.inf
        out.append(se)
    return pd.DataFrame(out, columns=["se_a", "se_b", "se_c", "se_d"])


# ----------------------------------------------------------------------
# Kiểm định độ phù hợp item (Yen's Q1, chia nhóm theo phân vị theta)
# ----------------------------------------------------------------------
def chi_square(df, item_param, theta_col="Theta", num_bins=10, n_params=2):
    """
    df         : DataFrame chứa cột theta_col và các cột câu hỏi (0/1/-1)
    item_param : DataFrame, index = tên câu, cột 'a', 'b' (có thể thêm 'c', 'd')
    n_params   : số tham số của mô hình (bậc tự do = số nhóm - n_params)
    """
    n_groups = num_bins
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
        c = float(item_param.loc[item, "c"]) if "c" in item_param.columns and np.isfinite(item_param.loc[item, "c"]) else C_DEFAULT
        d = float(item_param.loc[item, "d"]) if "d" in item_param.columns and np.isfinite(item_param.loc[item, "d"]) else D_DEFAULT

        if len(x) < n_groups * 5:
            rows.append([item, a, b, np.nan, np.nan, np.nan])
            continue

        p = c + (d - c) * expit(D * a * (t - b))
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
        dof = used - n_params
        pval = chi2.sf(stat, dof) if dof > 0 else np.nan
        rows.append([item, a, b, stat, dof, round(pval, 4) if dof > 0 else np.nan])

    return pd.DataFrame(rows, columns=["Item", "a", "b", "Chi2", "df", "p_value"]).set_index("Item")


# ----------------------------------------------------------------------
# Điểm thực quy đổi theo đường đặc trưng đề thi (TCC)
# ----------------------------------------------------------------------
def true_score(theta, raw, data: pd.Series, item_params: pd.DataFrame, scale_max=300):
    """
    Điểm thực = TCC(theta) / n * scale_max. item_params có cột a, b và có thể có c, d
    (thiếu hoặc NaN -> c = 0, d = 1). Muốn tính theo 2PL/3PL thì bỏ cột c/d trước khi gọi.
    """
    items = [c for c in data.index if c in item_params.index]      # toàn bộ câu của đề
    n = len(items)
    if n == 0 or not np.isfinite(raw):
            return np.nan
    if raw <= 0:
            return 0                       # sai hết hoặc bỏ trống hết
    if raw >= n:
            return int(scale_max)          # đúng hết toàn bộ câu tính điểm
    if not np.isfinite(theta):
            return np.nan
    a, b, c, d = _params4(item_params.loc[items])
    tcc = irt_probability(float(theta), a, b, c, d).sum()
    return int(np.clip(np.round(tcc / n * scale_max), 0, scale_max))


# ----------------------------------------------------------------------
# Person-fit (Lz, Infit MSQ, Outfit MSQ) và Item-fit (Infit MSQ, Outfit MSQ)
# ----------------------------------------------------------------------
def _fit_inputs(responses, item_params, theta, a, b, missing_as_wrong, theta_method, c=None, d=None):
    """Chuẩn bị U, M (mặt nạ), a, b, c, d, theta cho các hàm fit. Mặc định câu bỏ trống (-1) = sai."""
    U = np.asarray(responses, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    a, b, c, d = _params4(item_params, a, b, c, d)
    miss = ~np.isfinite(U) | (U == MISSING)
    if missing_as_wrong:
        M = np.ones_like(U)
    else:
        M = (~miss).astype(float)
    U = np.where(miss, 0.0, U)
    if theta is None:
        th, _ = theta_estimate(np.where(M > 0, U, MISSING), a=a, b=b, c=c, d=d,
                               method=theta_method, return_se=True)
    else:
        th = np.asarray(theta, dtype=float).ravel()
    return U, M, a, b, c, d, th


def person_fit(responses, item_params=None, theta=None, a=None, b=None, theta_method="MLE",
               missing_as_wrong=True, eps=1e-6, lz_cut=1.96, msq_range=(0.7, 1.3), c=None, d=None):
    """
    Person-fit cho từng thí sinh: Lz, Infit MSQ, Outfit MSQ.
      Lz     = (l - E[l]) / sqrt(Var[l])      (Drasgow, Levine & Williams, 1985)
      Outfit = mean( (u-P)^2 / (P*Q) )
      Infit  = sum( (u-P)^2 ) / sum( P*Q )
    Cờ: Flag_Lz (Lz < lz_cut), Flag_MSQ (Infit hoặc Outfit ngoài msq_range), Flag (một trong hai).
    responses: ma trận N x J (0/1/-1) hoặc DataFrame; theta=None -> tự ước lượng bằng theta_method.
    Lưu ý: Lz kém tin cậy khi đề ngắn (< ~20 câu) vì sai số của theta ước lượng.
    """
    index = responses.index if isinstance(responses, pd.DataFrame) else None
    U, M, a, b, c, d, th = _fit_inputs(responses, item_params, theta, a, b, missing_as_wrong, theta_method, c, d)

    P = np.clip(irt_probability(th, a, b, c, d), eps, 1 - eps)      # (N, J)
    Q = 1.0 - P
    W = P * Q
    l = np.sum(M * (U * np.log(P) + (1 - U) * np.log(Q)), axis=1)
    El = np.sum(M * (P * np.log(P) + Q * np.log(Q)), axis=1)
    Vl = np.sum(M * (P * Q * np.log(P / Q) ** 2), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lz = np.where(Vl > 0, (l - El) / np.sqrt(Vl), np.nan)
        res2 = M * (U - P) ** 2
        outfit = np.sum(res2 / W, axis=1) / M.sum(axis=1)
        infit = np.sum(res2, axis=1) / np.sum(M * W, axis=1)

    lo, hi = msq_range
    out = pd.DataFrame({"Lz": lz, "Infit_MSQ": infit, "Outfit_MSQ": outfit}, index=index)
    out["Flag_Lz"] = abs(out["Lz"]) < lz_cut
    out["Flag_MSQ"] = (out["Infit_MSQ"] < lo) | (out["Infit_MSQ"] > hi) | \
                      (out["Outfit_MSQ"] < lo) | (out["Outfit_MSQ"] > hi)
    out["Flag"] = out["Flag_Lz"] | out["Flag_MSQ"]
    return out


def item_fit(responses, item_params=None, theta=None, a=None, b=None, theta_method="MLE",
             missing_as_wrong=True, eps=1e-6, msq_range=(0.7, 1.3), poor_range=(0.5, 1.5), c=None, d=None):
    """
    Item-fit cho từng câu: Infit MSQ, Outfit MSQ (kỳ vọng = 1).
      Outfit_j = mean_i( (X-P)^2 / (P*Q) )          Infit_j = sum_i (X-P)^2 / sum_i P*Q
    Nhãn Fit_Label: Poor (max > poor_range[1]) > Overfit (min < poor_range[0])
                    > Good (cả hai trong msq_range) > Fair (còn lại).
    """
    index = responses.columns if isinstance(responses, pd.DataFrame) else None
    U, M, a, b, c, d, th = _fit_inputs(responses, item_params, theta, a, b, missing_as_wrong, theta_method, c, d)

    P = np.clip(irt_probability(th, a, b, c, d), eps, 1 - eps)
    W = P * (1.0 - P)
    res2 = M * (U - P) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        outfit = np.sum(res2 / W, axis=0) / M.sum(axis=0)
        infit = np.sum(res2, axis=0) / np.sum(M * W, axis=0)

    lo, hi = msq_range
    plo, phi = poor_range
    mx, mn = np.fmax(infit, outfit), np.fmin(infit, outfit)
    label = np.where(mx > phi, "Poor",
             np.where(mn < plo, "Overfit",
              np.where((infit >= lo) & (infit <= hi) & (outfit >= lo) & (outfit <= hi), "Good", "Fair")))
    return pd.DataFrame({"Infit_MSQ": infit, "Outfit_MSQ": outfit, "Fit_Label": label}, index=index)


# ----------------------------------------------------------------------
# Pipeline gọn
# ----------------------------------------------------------------------
def n_params(model, n_items):
    """Số tham số câu hỏi tự do của mô hình (1PL: 1 a chung + J b)."""
    return 1 + n_items if int(model) == 1 else int(model) * n_items


def fit_irt(df_items: pd.DataFrame, model=2, K=41, reg=1e-2, verbose=True, theta_method="EAP",
            a_init=None, b_init=None, fit_stats=True, msq_range=(0.7, 1.3), lz_cut=-2.0,
            a_bounds=(0.05, 4.0), b_bounds=None, **mmle_kwargs):
    """
    df_items: DataFrame (N x J) chỉ chứa các cột câu hỏi 0/1/-1.
    model   : 1, 2, 3 hoặc 4 (số tham số). a_bounds, b_bounds: giới hạn ước lượng a, b.
    Trả về (item_params, person_params). item_params luôn có cột a, b, c, d (c = 0, d = 1 nếu không ước lượng)
    và se_a, se_b, se_c, se_d (NaN nếu không ước lượng riêng từng câu).
    """
    model = int(model)
    U = df_items.to_numpy(float)
    U = np.where(U == MISSING, 0.0, U)   # bỏ trống = sai
    a, b, c, d, ll = mmle(U, K=K, reg=reg, verbose=verbose, return_ll=True, return_cd=True, model=model,
                          a_init=a_init, b_init=b_init, a_bounds=a_bounds, b_bounds=b_bounds, **mmle_kwargs)
    se = item_se_observed(U, a, b, K=K, c=c, d=d, model=model)
    items = pd.DataFrame({"a": a, "b": b, "c": c, "d": d,
                          "se_a": se["se_a"].to_numpy(), "se_b": se["se_b"].to_numpy(),
                          "se_c": se["se_c"].to_numpy(), "se_d": se["se_d"].to_numpy()},
                         index=df_items.columns)
    th, psd = theta_estimate(U, a=a, b=b, c=c, d=d, method=theta_method, return_se=True)
    persons = pd.DataFrame({"Theta": th, "SE": psd}, index=df_items.index)

    if fit_stats:
        # person-fit: Lz, Infit_MSQ, Outfit_MSQ, Flag_Lz, Flag_MSQ, Flag
        pf = person_fit(U, a=a, b=b, c=c, d=d, theta=th, missing_as_wrong=False, lz_cut=lz_cut, msq_range=msq_range)
        for col in pf.columns:
            persons[col] = pf[col].to_numpy()
        # item-fit: Infit_MSQ, Outfit_MSQ, Fit_Label
        itf = item_fit(U, a=a, b=b, c=c, d=d, theta=th, missing_as_wrong=False, msq_range=msq_range)
        for col in itf.columns:
            items[col] = itf[col].to_numpy()
        # chi-square (Yen's Q1): Chi2, Chi2_df, p_value
        dfU = pd.DataFrame(U, columns=df_items.columns)
        dfU["Theta"] = th
        chi = chi_square(dfU, items[["a", "b", "c", "d"]], n_params=len(FREE[model]))
        items["Chi2"] = chi["Chi2"].reindex(items.index).to_numpy()
        items["Chi2_df"] = chi["df"].reindex(items.index).to_numpy()
        items["p_value"] = chi["p_value"].reindex(items.index).to_numpy()
        if verbose:
            print(f"Person-fit: {int(persons['Flag'].sum())}/{len(persons)} thí sinh bị gắn cờ "
                  f"(Lz<-2: {int(persons['Flag_Lz'].sum())}, MSQ ngoài 0.7-1.3: {int(persons['Flag_MSQ'].sum())})")
            print("Item-fit:", items["Fit_Label"].value_counts().to_dict())
    if verbose:
        print(f"Marginal LL = {ll:.2f}, AIC = {-2 * ll + 2 * n_params(model, len(a)):.2f}")
    return items, persons


def fit_2pl(df_items: pd.DataFrame, K=41, reg=1e-2, verbose=True, theta_method="EAP",
            a_init=None, b_init=None, fit_stats=True, msq_range=(0.7, 1.3), lz_cut=-2.0, **kwargs):
    """Tương thích code cũ: fit_irt với model=2 (truyền model=... để đổi mô hình)."""
    kwargs.setdefault("model", 2)
    return fit_irt(df_items, K=K, reg=reg, verbose=verbose, theta_method=theta_method, a_init=a_init,
                   b_init=b_init, fit_stats=fit_stats, msq_range=msq_range, lz_cut=lz_cut, **kwargs)
 
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
 