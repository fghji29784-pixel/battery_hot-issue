"""
SDM 곡선에서 시정수 τ를 직접 측정한다.

사용법
  python3 measure_tau.py                 # 자체 테스트(합성 곡선)로 동작 확인
  python3 measure_tau.py curve.csv       # 실제 곡선 (컬럼: time_s, current_uA)

무엇을 알려주나
  1) 단일 지수 피팅 τ            — 정착이 하나의 시정수로 설명되는가
  2) 이중 지수 피팅 τ1, τ2       — 빠른/느린 성분이 따로 있는가
  3) 15분이 τ의 몇 배인가        — 15분 외삽이 가능한 영역인지 판정
  4) Rout를 낮췄을 때 예상 τ     — 단, RC 지배일 때만 유효 (아래 주의)

주의
  정착이 장비 RC가 아니라 셀 자체의 전기화학적 이완(고상 확산, 수 시간)에
  지배되면 Rout를 낮춰도 τ는 거의 줄지 않는다. 이중 지수에서 느린 성분이
  Rout에 반응하지 않으면 그 경우다.
"""
import sys, numpy as np
from scipy.optimize import curve_fit

def _exp1(t, Iinf, A, tau):            return Iinf - A*np.exp(-t/tau)
def _exp2(t, Iinf, A1, t1, A2, t2):    return Iinf - A1*np.exp(-t/t1) - A2*np.exp(-t/t2)

def measure(t, I, rout_now=0.5, rser=0.05, label=""):
    t = np.asarray(t, float); I = np.asarray(I, float)
    t = t - t[0]
    Iend = I[-max(len(I)//50, 5):].mean()       # 꼬리 평균 = 정착값 근사
    span = Iend - I[:max(len(I)//100, 3)].mean()

    print("\n" + "="*72)
    print(f" 시정수 측정  {label}")
    print("="*72)
    print(f"  측정 길이 {t[-1]/3600:.2f} 시간 / 포인트 {len(t):,}개")
    print(f"  시작 {I[0]:.2f} → 정착 근사 {Iend:.2f} µA  (변화폭 {span:.2f} µA)")

    out = {}
    try:
        p1, _ = curve_fit(_exp1, t, I, p0=[Iend, max(span,1), t[-1]/4],
                          bounds=([-np.inf, 0, 1], [np.inf, np.inf, 1e7]), maxfev=40000)
        r1 = np.std(I - _exp1(t, *p1))
        out["tau1"] = p1[2]
        print(f"\n  [단일 지수]  I∞ = {p1[0]:.2f} µA,  τ = {p1[2]/3600:.2f} 시간"
              f"   (잔차 {r1:.3f} µA)")
    except Exception as e:
        print("\n  [단일 지수] 피팅 실패:", e); r1 = np.inf

    try:
        p2, _ = curve_fit(_exp2, t, I,
                          p0=[Iend, max(span,1)*.5, t[-1]/20, max(span,1)*.5, t[-1]/3],
                          bounds=([-np.inf,0,1,0,1], [np.inf,np.inf,1e7,np.inf,1e7]),
                          maxfev=80000)
        r2 = np.std(I - _exp2(t, *p2))
        ta, tb = sorted([p2[2], p2[4]])
        out["tau_fast"], out["tau_slow"] = ta, tb
        print(f"  [이중 지수]  I∞ = {p2[0]:.2f} µA,"
              f"  τ_fast = {ta/60:.1f} 분,  τ_slow = {tb/3600:.2f} 시간"
              f"   (잔차 {r2:.3f} µA)")
        if r2 < r1*0.7:
            print("     → 잔차가 뚜렷이 작음. 시정수가 둘 이상 (빠른 성분 + 느린 성분)")
        else:
            print("     → 단일 지수로도 충분히 설명됨")
    except Exception as e:
        print("  [이중 지수] 피팅 실패:", e)

    tau = out.get("tau_slow", out.get("tau1"))
    if tau:
        ratio = 900/tau
        verdict = ("어려움 — 곡선이 거의 직선이라 수렴값 추정이 불안정"
                   if ratio < 0.25 else
                   "가능 영역 — 다변량 피처와 병행하면 쓸 만함"
                   if ratio < 0.5 else "양호 — 외삽이 안정적")
        print(f"\n  ★ 지배 시정수 {tau/3600:.2f} 시간  →  15분 / τ = {ratio:.2f}   {verdict}")
        print(f"\n  Rout 변경 시 예상 τ  (RC 지배일 경우에만 유효)")
        for R in (0.5, 0.3, 0.2, 0.15, 0.1):
            print(f"     Rout {R:>4.2f}옴 → τ ≈ {tau*(R+rser)/(rout_now+rser)/60:>6.1f} 분"
                  f"   (15분/τ = {900/(tau*(R+rser)/(rout_now+rser)):.2f})")
    return out


def _selftest():
    print("자체 테스트: τ_fast=3분, τ_slow=1.5시간 인 합성 곡선을 넣어 복원되는지 확인")
    t = np.arange(0, 6*3600, 1.0)
    I = _exp2(t, 120.0, 300.0, 180.0, 200.0, 5400.0) + np.random.default_rng(0).normal(0,.25,len(t))
    measure(t, I, label="(합성 곡선 · 정답 τ_slow = 1.50 시간)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        import pandas as pd
        df = pd.read_csv(sys.argv[1])
        tcol = next(c for c in df.columns if "time" in c.lower() or c.lower() in ("t","s"))
        icol = next(c for c in df.columns if "cur" in c.lower() or c.lower() in ("i","ua"))
        measure(df[tcol].values, df[icol].values, label=f"({sys.argv[1]})")
    else:
        _selftest()
