"""
15분 SDM 곡선으로 3일 ΔOCV를 예측하는 원리 데모  (합성 데이터)

※ 실제 데이터가 아닙니다. "왜 이게 가능한가"를 눈으로 보기 위한 시뮬레이션이며,
   동시에 실제 데이터가 들어왔을 때 쓸 파이프라인의 뼈대입니다.

핵심 논리
---------
SDM과 3일 ΔOCV는 같은 물리량(셀 내부 누설전류 I_sd)을 다르게 잰 것이다.
   3일 ΔOCV = (I_sd × 72h / 용량) × dOCV/dSOC        ... 느리게, 적분해서
   SDM 정착전류 = I_sd                                 ... 빠르게, 직접
따라서 SDM에서 I_sd를 알아내면 3일 ΔOCV를 예측할 수 있다.

문제
----
15분 시점의 전류는 I_sd가 아니다. 과도(이완) 성분이 겹쳐 있다.
   I_meas(t) = I_sd + A1·exp(-t/τ1) + A2·exp(-t/τ2)
τ2가 4시간이면 15분에서 exp(-0.25/4)=0.94, 즉 과도 성분이 94% 남아 있다.

해법
----
곡선의 "모양"으로 수렴할 값을 외삽한다. (전자체온계와 같은 원리)
단, 15분은 τ2에 비해 짧아 단일 지수 피팅이 불안정하므로,
피팅 결과를 포함한 여러 피처를 써서 최종 타깃을 직접 학습한다.
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

RNG = np.random.default_rng(42)

# ── 설정 (실제 값으로 바꿔 쓰는 자리) ──────────────────────────────────
N_TRAY, N_PER_TRAY = 120, 48          # 트레이 120개 × 48셀 = 5,760셀
T_MEAS_MIN   = 15                      # SDM 측정 길이 [분]
DT_S         = 1.0                     # 샘플링 [초]
CAP_AH       = 5.0                     # 셀 용량 [Ah]
DVDSOC_MV    = 5.0                     # dOCV/dSOC [mV per %SOC]
HOURS_STORE  = 72.0                    # 3일
EA_EV        = 0.94                    # 활성화 에너지 [eV] (Keysight 실측)
KB           = 8.617333e-5
NOISE_UA     = 0.25                    # 전류 측정 노이즈 [µA] (Keysight)
OCV_RES_MV   = 0.1                     # ΔOCV 측정 분해능 [mV]
DEFECT_FRAC  = 0.004                   # 불량 비율


def make_cells():
    """셀별 진짜 물성과 트레이 구조를 생성."""
    n = N_TRAY * N_PER_TRAY
    tray = np.repeat(np.arange(N_TRAY), N_PER_TRAY)

    # 진짜 자가방전 전류: 로그정규 + 불량 꼬리
    I_sd = RNG.lognormal(np.log(120), 0.25, n)              # 정상 ~120 µA
    is_def = RNG.random(n) < DEFECT_FRAC
    I_sd[is_def] *= RNG.uniform(1.8, 4.0, is_def.sum())     # 불량은 2~4배

    # 트레이 공통 교란: 온도 오프셋 (아레니우스로 전류에 곱셈 작용)
    tray_dT = RNG.normal(0, 0.6, N_TRAY)[tray]              # 트레이별 ±0.6℃
    cell_dT = RNG.normal(0, 0.25, n)                        # 셀별 잔차
    dT = tray_dT + cell_dT
    arr = np.exp(EA_EV / (KB * 298.15**2) * dT)             # 1℃당 약 13%
    I_sd_eff = I_sd * arr                                   # 측정에 실제로 보이는 값

    # 이완(과도) 성분 — I_sd와 무관하게 셀마다 다름  ← 이게 방해물
    A1  = RNG.lognormal(np.log(400), 0.4, n)                # 짧은 이완 진폭
    tau1= RNG.lognormal(np.log(3*60), 0.3, n)               # τ1 ~ 3분
    A2  = RNG.lognormal(np.log(250), 0.5, n)                # 긴 이완 진폭
    tau2= RNG.lognormal(np.log(4*3600), 0.35, n)            # τ2 ~ 4시간
    # 트레이 공통 이완 (같은 트레이는 이력이 비슷 → 오버행 상태도 비슷)
    A2 *= np.exp(RNG.normal(0, 0.25, N_TRAY))[tray]

    return pd.DataFrame(dict(cell=np.arange(n), tray=tray, I_sd=I_sd,
                             I_sd_eff=I_sd_eff, dT=dT, A1=A1, tau1=tau1,
                             A2=A2, tau2=tau2, is_defect=is_def))


def make_curves(df, minutes=T_MEAS_MIN, tau_scale=1.0):
    """SDM 측정 곡선:  I(t) = I_sd + A1·exp(-t/τ1) + A2·exp(-t/τ2) + noise

    tau_scale : 시정수 배율. τ = (Rout + Rser)·Ceff 이므로 출력저항 Rout를
                낮추면 이 값이 작아진다. Rout 1Ω 기준 1.0, 0.15Ω면 약 0.15.
    """
    t = np.arange(0, minutes*60, DT_S)
    tau1 = df.tau1.values[:, None] * tau_scale
    tau2 = df.tau2.values[:, None] * tau_scale
    I = (df.I_sd_eff.values[:, None]
         + df.A1.values[:, None] * np.exp(-t / tau1)
         + df.A2.values[:, None] * np.exp(-t / tau2))
    I += RNG.normal(0, NOISE_UA, I.shape)
    return t, I


def make_target(df):
    """3일 ΔOCV [mV].  보관 중에도 온도가 작용하므로 I_sd_eff를 사용."""
    dQ_mAh = df.I_sd_eff.values * 1e-3 * HOURS_STORE        # µA×h → mAh
    dSOC   = dQ_mAh / (CAP_AH * 1000) * 100                 # [%]
    dV     = dSOC * DVDSOC_MV                               # [mV]
    dV    += RNG.normal(0, 0.03, len(df))                   # 기타 잡음
    return np.round(dV / OCV_RES_MV) * OCV_RES_MV           # 측정 분해능 양자화


# ── 피처 추출 ────────────────────────────────────────────────────────
def _biexp(t, Iinf, A, tau):
    return Iinf + A * np.exp(-t / tau)

def extract_features(t, I):
    """15분 곡선 하나에서 피처를 뽑는다."""
    rows = []
    for y in I:
        f = {}
        n = len(y)
        # (1) 구간을 10등분한 상대 시점 스냅샷 (윈도우 길이에 무관하게 동작)
        for j, k in enumerate(np.linspace(n//10, n-1, 8).astype(int)):
            f[f"I_p{j}"] = y[k]
        # (2) 기울기 / 비율 / 적분
        w = max(n//5, 3)
        f["slope_head"] = np.polyfit(t[:w], y[:w], 1)[0] * 3600          # µA/h
        f["slope_tail"] = np.polyfit(t[-w:], y[-w:], 1)[0] * 3600        # µA/h
        f["curvature"]  = f["slope_head"] - f["slope_tail"]
        f["ratio_endstart"] = y[-1] / y[max(n//10, 1)]
        f["integral"]   = np.trapezoid(y, t) / t[-1]
        f["drop_total"] = y[0] - y[-1]
        f["y_end"]      = y[-1]
        # (3) 지수 피팅 외삽  ← Keysight가 쓰는 방법
        try:
            p, _ = curve_fit(_biexp, t, y,
                             p0=[y[-1]*0.6, max(y[0]-y[-1], 1.0), max(t[-1], 60.)],
                             bounds=([0., 0., 10.], [y[-1], 1e5, 1e6]),
                             maxfev=20000)
            f["fit_Iinf"], f["fit_A"], f["fit_tau"] = p[0], p[1], abs(p[2])
            f["fit_resid"] = float(np.std(y - _biexp(t, *p)))
        except Exception:
            f["fit_Iinf"] = f["fit_A"] = f["fit_tau"] = f["fit_resid"] = np.nan
        rows.append(f)
    return pd.DataFrame(rows)
