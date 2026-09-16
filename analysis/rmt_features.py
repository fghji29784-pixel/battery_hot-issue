# -*- coding: utf-8 -*-
"""
슬라이드의 Random Matrix Theory(RMT) 노이즈 분석을 SDM 원시 시계열에 적용하기 위한 핵심 함수.

아이디어
  긴 시계열 하나를 겹치는 시간창(window) K개로 쪼갠다.
  각 창을 detrend + 정규화하면 "본질이 백색잡음"인 창들끼리는
  서로 상관이 없어야 한다 — 이때 창 간 상관행렬의 고유값 분포는
  Marchenko–Pastur(MP) 법칙을 따른다 (= "random matrix").

  실제 고유값이 MP 상한(λ+)을 벗어나거나, 그 고유벡터가 소수의 창에
  집중(국소화, localized)돼 있으면 → 그 창들 사이에 우연이 아닌
  진짜 시간적 구조(상관된 사건)가 있다는 뜻이다.

  SDM에서 이게 왜 유용한가
    평균 전류 레벨(I_SD)은 정상인데, 미세단락으로 인해 순간적으로
    깜빡이는(telegraph-noise) 결함은 바이지수 피팅(레벨 기반 피처)으로는
    보이지 않는다. 이런 결함은 "특정 시간대 여러 창이 서로 닮아있다"는
    형태로만 드러나므로, 상관행렬 고유값/고유벡터로 잡아낼 수 있다.

사용법 (자체 테스트)
  python3 rmt_features.py
"""
import numpy as np


def sliding_windows(x, win, step):
    """1차원 배열 x를 길이 win, 간격 step인 겹치는 창들로 쪼갠다. shape=(K, win)."""
    x = np.asarray(x, float)
    n = len(x)
    k = (n - win) // step + 1
    if k < 2:
        raise ValueError(f"창 개수가 너무 적습니다 (n={n}, win={win}, step={step} → K={k})")
    idx = np.arange(win)[None, :] + step * np.arange(k)[:, None]
    return x[idx]


def _detrend_zscore(W):
    """각 창(행)에서 선형추세를 빼고 z-score 정규화. 완전 평탄한(분산 0) 창은 그대로 0으로."""
    K, L = W.shape
    t = np.arange(L, dtype=float)
    X = np.vstack([t, np.ones_like(t)]).T          # (L,2)
    pinv = np.linalg.pinv(X)                        # (2,L)
    coeffs = W @ pinv.T                              # (K,2)
    trend = coeffs @ X.T                             # (K,L)
    resid = W - trend
    std = resid.std(axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    return (resid - resid.mean(axis=1, keepdims=True)) / std


def corr_eigen(W):
    """정규화된 창 행렬 W(K,L)의 창-간 상관행렬을 고유분해. 내림차순 정렬."""
    K, L = W.shape
    Wn = _detrend_zscore(W)
    C = (Wn @ Wn.T) / L
    eigvals, eigvecs = np.linalg.eigh(C)             # 오름차순
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order], Wn


def mp_edge(K, L):
    """순수 백색잡음(iid)일 때 기대되는 고유값 상한(λ+). K=창 개수, L=창 길이."""
    q = K / L
    return (1 + np.sqrt(q)) ** 2


def ipr(v):
    """고유벡터의 역참여비(Inverse Participation Ratio). 1/K(완전 비국소)~1(완전 국소)."""
    v = v / np.linalg.norm(v)
    return float(np.sum(v ** 4))


def spectral_entropy(eigvals):
    """고유값 분포의 정규화 엔트로피 [0,1]. 낮을수록 소수 모드에 에너지가 집중됨."""
    p = np.clip(eigvals, 0, None)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    K = len(p)
    h = -np.sum(p * np.log(p + 1e-15))
    return float(h / np.log(K))


def extract_rmt_features(x, win, step, prefix=""):
    """시계열 x 하나에서 RMT 피처 딕셔너리를 뽑는다.

    win, step : 창 길이/간격 (샘플 단위). 노이즈 사건(예: RTN 깜빡임)의
                지속시간과 같은 자릿수로 잡아야 그 사건을 창 하나가
                온전히 담거나, 여러 창에 걸쳐 상관이 생긴다.
    """
    W = sliding_windows(x, win, step)
    K, L = W.shape
    eigvals, eigvecs, _ = corr_eigen(W)
    lam_plus = mp_edge(K, L)
    n_above = int(np.sum(eigvals > lam_plus))
    f = {
        "lambda1": eigvals[0],
        "lambda1_excess": eigvals[0] - lam_plus,
        "lambda1_ratio": eigvals[0] / lam_plus,
        "lambda2": eigvals[1] if K > 1 else np.nan,
        "n_above_mp": n_above,
        "ipr1": ipr(eigvecs[:, 0]),
        "ipr2": ipr(eigvecs[:, 1]) if K > 1 else np.nan,
        "entropy": spectral_entropy(eigvals),
        "K": K, "L": L, "mp_edge": lam_plus,
    }
    return {(prefix + k if prefix else k): v for k, v in f.items()}


def extract_rmt_features_multi(signals, win, step):
    """여러 채널(예: 전류/전압/온도) 딕셔너리 {name: array} 를 한 번에 처리."""
    out = {}
    for name, x in signals.items():
        try:
            out.update(extract_rmt_features(x, win, step, prefix=f"{name}_rmt_"))
        except ValueError:
            continue
    return out


# ── 자체 테스트 ──────────────────────────────────────────────────────────
def _selftest():
    # win >> step, K < L 영역을 유지해야 MP 근사가 성립한다.
    # (창이 시료 개수보다 많아지면(K>L) 상관행렬이 저계수가 되어 MP 가정이 깨진다)
    rng = np.random.default_rng(0)
    n, win, step = 3000, 300, 50

    print("="*72)
    print(" 자체 테스트 1: 순수 백색잡음 — 고유값이 MP 상한 근처에 머물러야 함")
    print("="*72)
    x_white = rng.normal(0, 1.0, n)
    f = extract_rmt_features(x_white, win, step)
    print(f"  K={f['K']} L={f['L']} MP상한(λ+)={f['mp_edge']:.3f}")
    print(f"  λ1={f['lambda1']:.3f}  (λ+ 대비 {f['lambda1_ratio']:.2f}배)"
          f"  n_above_mp={f['n_above_mp']}  ipr1={f['ipr1']:.4f} (기대치≈1/K={1/f['K']:.4f})")

    print("\n" + "="*72)
    print(" 자체 테스트 2: 같은 잡음 + 국소 구간(전체의 15%)에 텔레그래프(RTN) 결함 신호 주입")
    print(" (미세단락으로 순간적으로 깜빡이는 결함을 모사 — 평균 레벨은 거의 안 변함)")
    print("="*72)
    x_defect = x_white.copy()
    lo, hi = 1000, 1450                              # 전체 3000샘플 중 15% 구간
    period = 220
    t_local = np.arange(hi - lo)
    x_defect[lo:hi] += 3.0 * np.sign(np.sin(2*np.pi*t_local/period))
    f2 = extract_rmt_features(x_defect, win, step)
    print(f"  λ1={f2['lambda1']:.3f}  (λ+ 대비 {f2['lambda1_ratio']:.2f}배)"
          f"  n_above_mp={f2['n_above_mp']}  ipr1={f2['ipr1']:.4f}")
    print(f"  참고: 결함 구간 평균값 변화 = {x_defect[lo:hi].mean()-x_white[lo:hi].mean():+.3f}"
          f"  (레벨 기반 피처로는 거의 안 잡힘)")
    print("\n  → 국소 결함이 있으면 λ1이 MP 상한을 뚜렷이 넘고 ipr1이 커져야(국소화) 정상.")
    ok = f2["lambda1_ratio"] > f["lambda1_ratio"] * 1.5 and f2["ipr1"] > f["ipr1"] * 1.3
    print(f"  판정: {'✅ 기대대로 작동' if ok else '⚠ 차이가 뚜렷하지 않음 — 파라미터(win/step/진폭) 조정 필요'}")


if __name__ == "__main__":
    _selftest()
