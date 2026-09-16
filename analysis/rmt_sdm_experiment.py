# -*- coding: utf-8 -*-
"""
RMT(랜덤행렬이론) 노이즈 피처가 SDM 선별에 실제로 도움이 되는지 합성 데이터로 검증.

배경
  슬라이드 자료는 "전압 노이즈 시계열 → 시간창 상관행렬 → 고유값/고유벡터
  국소화(localization) → ML 피처"로 셀 품질을 예측하는 파이프라인을 보여준다.
  이게 SDM(자가방전 측정)에도 통할지 검토해 달라는 요청에 대한 실험 코드.

핵심 가설 (측정 가능한 형태로 재구성)
  기존 방식(바이지수 피팅, sdm_demo.py)은 "평균 자가방전 전류 레벨" 이상을
  기준으로 불량을 잡는다. 반면 미세단락(nascent internal short)처럼
  순간적으로 깜빡이는(전류가 튀었다 가라앉았다 하는) 결함은 평균 레벨을
  거의 바꾸지 않아 레벨 기반 피처로는 못 잡을 수 있다.
  이런 "레벨은 정상, 구조만 이상" 결함을 RMT 피처가 잡아내는지가 관건이다.

실험 설계
  ① 레벨 결함(level defect)  — I_SD 자체가 2~4배 큼 (기존 sdm_demo.py와 동일)
  ② 구조 결함(noise defect)  — I_SD는 정상, 대신 국소 구간에 평균 0인
     텔레그래프성 요동을 주입 (평균 전류·기울기에는 거의 안 보이게 설계)
  두 집단을 독립으로 만들어, "레벨 피처만 쓸 때"와 "레벨+RMT 피처를 같이
  쓸 때" 각각이 ①과 ②를 얼마나 잡는지 lift(선별 배수)로 비교한다.

주의
  전부 합성 데이터다. "이 방법이 원리적으로 통하는 문제 구조인가"를
  보는 용도이며, 실제 SDM 원시 파형(전류/전압/온도, 실제 샘플링레이트)이
  있어야 진짜 검증이 된다. 사용법은 맨 아래 main() 참고.

사용법
  python3 rmt_sdm_experiment.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import hypergeom

from rmt_features import extract_rmt_features

RNG = np.random.default_rng(42)

# ── 합성 데이터 설정 ────────────────────────────────────────────────────
N_TRAY, N_PER_TRAY = 15, 40                 # 600셀 (실행 시간 고려해 축소)
FS_HZ        = 5.0                          # 샘플링 주파수 [Hz] (dt=0.2s)
DURATION_MIN = 10
DT_S         = 1.0 / FS_HZ
N_SAMP       = int(DURATION_MIN * 60 * FS_HZ)

I_SD_MEAN_UA = 120.0
NOISE_UA     = 0.25
LEVEL_DEFECT_FRAC = 0.015                   # ① 레벨 결함 비율
NOISE_DEFECT_FRAC = 0.025                   # ② 구조 결함 비율 (레벨과 독립)
CUTOFFS_MIN  = [3, 5, 8, 10]


def make_cells():
    n = N_TRAY * N_PER_TRAY
    tray = np.repeat(np.arange(N_TRAY), N_PER_TRAY)

    I_sd = RNG.lognormal(np.log(I_SD_MEAN_UA), 0.25, n)
    is_level = RNG.random(n) < LEVEL_DEFECT_FRAC
    I_sd[is_level] *= RNG.uniform(1.8, 4.0, is_level.sum())

    is_noise = RNG.random(n) < NOISE_DEFECT_FRAC

    A1 = RNG.lognormal(np.log(400), 0.4, n)
    tau1 = RNG.lognormal(np.log(180), 0.3, n)
    A2 = RNG.lognormal(np.log(250), 0.5, n)
    tau2 = RNG.lognormal(np.log(4*3600), 0.35, n)

    return pd.DataFrame(dict(cell=np.arange(n), tray=tray, I_sd=I_sd,
                              is_level=is_level, is_noise=is_noise,
                              A1=A1, tau1=tau1, A2=A2, tau2=tau2))


def make_curve(row):
    """셀 하나의 원시 전류 파형. 레벨 결함은 I_sd에, 구조 결함은 국소 요동으로 반영."""
    t = np.arange(N_SAMP) * DT_S
    I = (row.I_sd + row.A1*np.exp(-t/row.tau1) + row.A2*np.exp(-t/row.tau2)
         + RNG.normal(0, NOISE_UA, N_SAMP))

    if row.is_noise:
        # 평균 0인 국소 텔레그래프 요동 — 레벨/기울기 피처에는 거의 안 잡히게 설계.
        span = RNG.integers(int(0.10*N_SAMP), int(0.30*N_SAMP))
        lo = RNG.integers(0, max(N_SAMP - span, 1))
        hi = lo + span
        period = RNG.integers(int(2.0*FS_HZ), int(6.0*FS_HZ))    # 2~6초 주기
        amp = NOISE_UA * RNG.uniform(8, 20)
        tt = np.arange(hi - lo)
        I[lo:hi] += amp * np.sign(np.sin(2*np.pi*tt/period))
    return t, I


# ── 레벨(기존) 피처: sdm_demo.py의 baseline 피처를 축약 재구현 ───────────
def _biexp(t, Iinf, A, tau):
    return Iinf + A*np.exp(-t/tau)


def level_features(t, I):
    f = {}
    n = len(I)
    for j, k in enumerate(np.linspace(n//10, n-1, 6).astype(int)):
        f[f"I_p{j}"] = I[k]
    w = max(n//5, 3)
    f["slope_head"] = np.polyfit(t[:w], I[:w], 1)[0]
    f["slope_tail"] = np.polyfit(t[-w:], I[-w:], 1)[0]
    f["curvature"] = f["slope_head"] - f["slope_tail"]
    f["drop_total"] = I[0] - I[-1]
    f["y_end"] = I[-1]
    f["mean"] = I.mean()
    try:
        p, _ = curve_fit(_biexp, t, I,
                         p0=[I[-1]*0.9, max(I[0]-I[-1], 1.0), max(t[-1], 30.)],
                         bounds=([0., 0., 1.], [I[-1]+1e3, 1e5, 1e6]), maxfev=20000)
        f["fit_Iinf"], f["fit_A"], f["fit_tau"] = p
        f["fit_resid"] = float(np.std(I - _biexp(t, *p)))
    except Exception:
        f["fit_Iinf"] = f["fit_A"] = f["fit_tau"] = f["fit_resid"] = np.nan
    return f


def rmt_feature_row(I, seg_len):
    win = max(60, seg_len // 10)
    step = max(10, win // 6)
    try:
        return extract_rmt_features(I, win, step, prefix="rmt_")
    except ValueError:
        return {}


# ── 선별 성능(lift) 평가 — screen.py와 동일한 방식 ───────────────────────
def lift(score, y, n, ks=(0.02, 0.05, 0.10)):
    out = []
    for q in ks:
        k = max(int(round(n*q)), 1)
        a = set(np.argsort(-y)[:k]); b = set(np.argsort(-score)[:k])
        hit = len(a & b); exp = k*k/n
        p = hypergeom.sf(hit-1, n, k, k) if hit else 1.0
        out.append((q, k, hit, exp, hit/max(exp, 1e-9), p))
    return out


def print_lift(rows, label):
    print(f"  [{label}]")
    print(f"    {'상위':>6}{'셀수':>6}{'일치':>8}{'무작위':>9}{'배수':>7}{'p':>11}")
    for q, k, hit, exp, lf, p in rows:
        print(f"    {q*100:>5.0f}%{k:>6}{f'{hit}/{k}':>8}{exp:>9.2f}{lf:>6.0f}배{p:>11.1e}")


def main():
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = make_cells()
    n = len(df)
    print("="*78)
    print(f" 합성 SDM 실험: {n}셀 / 트레이 {N_TRAY}개 / fs={FS_HZ}Hz"
          f" / 레벨결함 {df.is_level.sum()}개 / 구조결함 {df.is_noise.sum()}개"
          f" (겹침 {int((df.is_level & df.is_noise).sum())}개)")
    print("="*78)

    curves = [make_curve(r) for r in df.itertuples()]

    for m in CUTOFFS_MIN:
        n_samp_cut = int(m * 60 * FS_HZ)
        LV, RM = [], []
        for t, I in curves:
            tt, II = t[:n_samp_cut], I[:n_samp_cut]
            LV.append(level_features(tt, II))
            RM.append(rmt_feature_row(II, n_samp_cut))
        LV = pd.DataFrame(LV).fillna(0.0)
        RM = pd.DataFrame(RM).fillna(0.0)

        g = df["tray"].values
        cv = GroupKFold(n_splits=min(5, N_TRAY))
        clf = HistGradientBoostingClassifier(random_state=0)

        print(f"\n{'-'*78}\n 구간 {m}분  (레벨 피처 {LV.shape[1]}개, RMT 피처 {RM.shape[1]}개)\n{'-'*78}")

        for target_name, y in [("① 레벨결함(is_level)", df.is_level.values.astype(int)),
                                ("② 구조결함(is_noise) — 레벨 피처로는 원래 안 보여야 함", df.is_noise.values.astype(int))]:
            if y.sum() < 3:
                continue
            print(f"\n  ── 타깃: {target_name} (양성 {y.sum()}개) ──")
            p_base = cross_val_predict(clf, LV.values, y, cv=cv, groups=g, method="predict_proba")[:, 1]
            print_lift(lift(p_base, y, n), "레벨 피처만")

            X_both = pd.concat([LV, RM], axis=1).values
            p_both = cross_val_predict(clf, X_both, y, cv=cv, groups=g, method="predict_proba")[:, 1]
            print_lift(lift(p_both, y, n), "레벨+RMT 피처")

    print("\n" + "="*78)
    print(" 해석: ①에서 두 피처셋 성능이 비슷하고, ②에서 'RMT 피처'만 배수가")
    print(" 뚜렷이 오르면 → RMT가 레벨 기반으로 놓치던 결함 유형을 보완한다는 뜻.")
    print("="*78)


if __name__ == "__main__":
    main()
