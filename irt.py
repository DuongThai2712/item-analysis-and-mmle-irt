import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import logsumexp
from scipy.optimize import minimize

# Hàm tính độ phân biệt bằng point-biserial correlation
def cal_disc(r):
    r = np.nan_to_num(r, nan=1e-6) 
    return r/np.sqrt(1-r**2)
    
# Tính độ khó
def cal_diff(p) -> float:
    if p <= 0:
        p = 1e-6
    elif p >= 1:
        p = 1 - 1e-6
    
    return np.log((1-p)/p).clip(-6, 6)

def irt_probability(theta, a, b):
    """
    Compute P(1|theta) for 2PL model.
    theta: (K,) or (K,1)
    a, b: (J,)
    Return: (K, J)
    """
    theta = np.atleast_2d(theta).reshape(-1, 1)  # (K,1)
    a = np.asarray(a).reshape(1, -1)  # (1,J)
    b = np.asarray(b).reshape(1, -1)  # (1,J)
    z = 1.702 * a * (theta - b)
    return 1 / (1 + np.exp(-z))

def log_likelihood(U, a_list, b_list, theta_grid, gh_weights, eps=1e-12):
    """Compute the log-likelihood of the data given item parameters using Gauss-Hermite quadrature."""
    N, J = U.shape
    K = len(theta_grid)
    P_kj = irt_probability(theta_grid, a_list, b_list)  # (K, J)

    # đảm bảo không có 0 hoặc 1 tuyệt đối
    P_kj = np.clip(P_kj, eps, 1 - eps)

    logP = np.log(P_kj)
    log1mP = np.log(1.0 - P_kj)

    ll = 0.0
    mask = (U != -1)

    for i in range(N):
        u = U[i, :]
        m = mask[i, :]

        # chỉ tính những câu hợp lệ
        valid = (u != -1)
        u_valid = u[valid]
        logP_valid = logP[:, valid]
        log1mP_valid = log1mP[:, valid]

        term_k = u_valid @ logP_valid.T + (1 - u_valid) @ log1mP_valid.T
        term_k += np.log(gh_weights + eps)

        # nếu toàn bộ invalid → bỏ qua
        if np.all(~m):
            continue  

        # chống -inf toàn bộ
        if np.all(np.isneginf(term_k)):
            continue

        ll += logsumexp(term_k)

    return ll

def mmle(U, a_init, b_init, name="MMLE", max_iter=60, K=81, tol=1e-4,
         reg=1e-2, step_size=0.3, verbose=True):
    N, J = U.shape
    a = np.array(a_init, dtype=float).copy().clip(1e-3, 3.0)
    b = np.array(b_init, dtype=float).copy().clip(-6.0, 6.0)
    b = b - np.mean(b)  # chuẩn hoá b về trung bình 0
    # a = np.ones(J, dtype=float)
    # b = np.zeros(J, dtype=float)

    # Gauss-Hermite nodes
    theta_grid, gh_weights = np.polynomial.hermite.hermgauss(K)
    theta_grid = theta_grid * np.sqrt(2)         # đúng chuẩn GH
    gh_weights = gh_weights / np.sqrt(np.pi)


    prev_ll = -np.inf
    mask = (U != -1)

    if verbose:
        print(f"Start {name}: N={N}, J={J}, K={K}")

    for it in range(1, max_iter + 1):
        # --- E-step ---
        P_kj = irt_probability(theta_grid, a, b)
        P_kj = np.clip(P_kj, 1e-12, 1-1e-12)
        logP = np.log(P_kj)
        log1mP = np.log1p(-P_kj)

        L = np.zeros((N, K))
        for k in range(K):
            L[:, k] = (mask * U) @ logP[k, :].T + (mask * (1 - U)) @ log1mP[k, :].T
            L[:, k] += np.log(gh_weights[k] + 1e-12)

        denom = logsumexp(L, axis=1)
        W = np.exp(L - denom[:, None])

        a_old, b_old = a.copy(), b.copy()

        # --- M-step ---
        theta_k = theta_grid.reshape(K, 1)
        for j in range(J):
            col = U[:, j]
            valid_idx = np.where(col != -1)[0]
            if valid_idx.size == 0:
                continue
            prop = np.mean(col[valid_idx])
            # Thay continue bằng điều chỉnh P để tránh log(0), không bỏ item
            if prop < 0.01:
                prop = 0.01
            elif prop > 0.99:
                prop = 0.99

            u_vec = col[:, None]
            mask_vec = mask[:, j][:, None]
            P_kj_vec = P_kj[:, j].reshape(1, K) # test thử, lỗi

            for _inner in range(2):
                D = (u_vec @ np.ones((1, K))) - P_kj_vec  # (N, K)
                D = D * mask_vec                          # mask theo rows
                term_theta = theta_k - b[j]

                grad_a = np.sum(W * D * (1.702 * term_theta).T) - reg * (a[j] - 1.0)
                grad_b = np.sum(W * D * (-a[j] * 1.702)) - reg * b[j]

                # q_k = P_kj_vec * (1 - P_kj_vec)
                # q_k = (P_kj_vec * (1 - P_kj_vec)).reshape(1, K)   # (1, K)
                # # thêm reg nhỏ trực tiếp vào Hessian
                # hess_aa = -np.sum(W * (q_k.T * (1.702 * term_theta).T ** 2)) - 1e-5
                # hess_ab = -np.sum(W * (q_k.T * 1.702**2 * term_theta.T))
                # hess_bb = -np.sum(W * (q_k.T * (a[j]*1.702)**2)) - 1e-5
                # q_k: (K,)
                q_k = (P_kj_vec.flatten() * (1 - P_kj_vec.flatten()))  # (K,)

                # (1, K) để broadcast với W (N, K)
                q = q_k.reshape(1, K)
                tt = term_theta.flatten().reshape(1, K)

                # Hessian cho a
                hess_aa = -np.sum(W * (q * (1.702 * tt)**2)) - 1e-5

                # Hessian cho cross-term ab
                hess_ab = -np.sum(W * (q * (1.702 * tt) * (1.702 * a[j])))

                # Hessian cho b
                hess_bb = -np.sum(W * (q * (1.702 * a[j])**2)) - 1e-5

                I = -np.array([[hess_aa, hess_ab], [hess_ab, hess_bb]])
                I_reg = I + reg * np.eye(2)
                g = np.array([grad_a, grad_b])

                try:
                    delta = np.linalg.solve(I_reg, g)
                except np.linalg.LinAlgError:
                    delta = 1e-3 * g

                max_step_a = 0.15
                max_step_b = 0.30

                delta[0] = np.clip(delta[0], -max_step_a, max_step_a)
                delta[1] = np.clip(delta[1], -max_step_b, max_step_b)

                a[j] = np.clip(a[j] + step_size * delta[0], 1e-3, 3.0)
                b[j] = np.clip(b[j] + step_size * delta[1], -6.0, 6.0)

        # --- check LL & convergence ---
        new_ll = log_likelihood(U, a, b, theta_grid, gh_weights)

        if new_ll < prev_ll - 1e-6:
            a, b = a_old, b_old
            step_size *= 0.7
            if step_size < 1e-4:
                print(f"Dừng tại Iter {it}: mean(a)={np.mean(a):.4f}, mean(b)={np.mean(b):.4f}, LL={new_ll:.6f}")
                break
            continue

        if (abs(new_ll - prev_ll) < tol) and (np.max(np.abs(a - a_old)) < tol) and (np.max(np.abs(b - b_old)) < tol):
            if verbose:
                print(f"✅ Hội tụ tại Iter {it}.")
            break

        prev_ll = new_ll
        

    return a, b
