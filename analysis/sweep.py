# -*- coding: utf-8 -*-
"""
최종 산출물 — "몇 분이면 충분한가" 곡선

  python sweep.py "파일.CSV"                 # 기본
  python sweep.py "파일.CSV" --plot          # 그림도 저장

하는 일
  1) 이상 곡선을 '자동으로' 검출해 제외한다 (눈으로 고르지 않는다)
     → 양산에 적용하려면 규칙이 재현 가능해야 하므로
  2) 측정 구간을 1분부터 늘려가며, 최종값 순위를 얼마나 재현하는지 측정
  3) 값 / 비율 피처 / 다변량 세 가지를 비교
  4) 목표 상관에 도달하는 최소 측정시간을 출력 → 장비 대수 산정 근거

원본/셀단위 값은 출력하지 않는다.
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from diagnose import read_any, to_matrix

WINDOWS = [60, 180, 300, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200, 10800]
TARGETS = [0.70, 0.80, 0.90]


def find_outliers(t, I, dd_thresh=0.05, noise_k=6.0):
    """이상 곡선 자동 검출. 눈으로 고르지 않고 재현 가능한 규칙만 사용.

    (1) 낙폭(drawdown): 평활 곡선이 이제까지의 최댓값 대비 얼마나 되돌아갔는가.
        전위고정 SDM 전류는 정착값을 향해 단조 상승하므로, 총 상승폭 대비
        dd_thresh 이상 되돌아가면 이상으로 본다.
    (2) 국소 잡음: 평활 곡선 대비 잔차가 다른 셀들보다 크게 벗어나는 경우.
    """
    n, T = I.shape
    win = max(T // 100 * 2 + 1, 11)
    sm = pd.DataFrame(I.T).rolling(win, center=True, min_periods=1).median().values.T
    rise = np.maximum(sm.max(axis=1) - sm[:, 0], 1e-9)
    runmax = np.maximum.accumulate(sm, axis=1)
    drawdown = (runmax - sm).max(axis=1) / rise           # 총 상승폭 대비 최대 낙폭
    resid = np.std(I - sm, axis=1)
    med, iqr = np.median(resid), np.subtract(*np.percentile(resid, [75, 25]))
    noisy = resid > med + noise_k * (iqr + 1e-12)

    flags = (drawdown > dd_thresh) | noisy
    reasons = {}
    for i in np.where(flags)[0]:
        r = []
        if drawdown[i] > dd_thresh: r.append(f"낙폭 {drawdown[i]*100:.0f}% (기준 {dd_thresh*100:.0f}%)")
        if noisy[i]:                r.append(f"잔차 {resid[i]:.1f} (중앙값 {med:.1f})")
        reasons[i] = ", ".join(r)
    return flags, reasons


def feats(t, I, w):
    """구간 w(초)까지의 곡선에서 피처 추출."""
    j = int(np.argmin(np.abs(t-w)))
    if j < 4: return None
    st, sg = t[:j+1], I[:, :j+1]
    ix = lambda s: int(np.argmin(np.abs(st-s)))
    a, b, c = ix(w*0.2), ix(w*0.5), j
    e = 1e-9
    q = max(j//3, 2)
    return {
        "값":        sg[:, c],
        "비율20":    sg[:, c]/(sg[:, a]+e),
        "비율50":    sg[:, c]/(sg[:, b]+e),
        "값x비율":   sg[:, c]*sg[:, c]/(sg[:, a]+e),
        "적분":      np.trapezoid(sg, st, axis=1)/st[-1],
        "기울기":    np.array([np.polyfit(st[-q:], s[-q:], 1)[0]*3600 for s in sg]),
    }


def main(path, do_plot=False):
    df, enc, sep, hdr = read_any(path)
    t, I, ids, fmt = to_matrix(df)
    o = np.argsort(t); t = t[o]-t[o][0]; I = I[:, o]
    I = I[np.isfinite(I).all(axis=1)]
    n0 = len(I)

    print("="*76); print(" 측정시간 스윕 — 몇 분이면 충분한가"); print("="*76)
    print(f"  입력: 셀 {n0}개 / 길이 {t[-1]/3600:.1f}시간 / 간격 {np.median(np.diff(t)):.0f}초")

    flags, reasons = find_outliers(t, I)
    print(f"\n  [자동 이상 검출] {flags.sum()}개 제외  (전체의 {flags.mean()*100:.1f}%)")
    for i, r in list(reasons.items())[:8]:
        print(f"      셀 #{i:<3} {r}")
    I = I[~flags]; n = len(I)
    print(f"  → 분석 대상 {n}개")

    T = I.shape[1]; Ifin = I[:, -max(T//50,3):].mean(axis=1)

    print("\n" + "-"*76)
    print(" 구간별 최종값 순위 재현도 (Spearman ρ)")
    print("-"*76)
    print(f"  {'측정시간':>8} {'값':>8} {'비율50':>8} {'최선단일':>9} {'다변량CV':>9}   {'최선 피처':<10}")
    print("  " + "-"*62)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    rows = []
    for w in WINDOWS:
        if w > t[-1]*0.9: break
        F = feats(t, I, w)
        if F is None: continue
        rs = {k: spearmanr(v, Ifin).statistic for k, v in F.items()}
        best = max(rs, key=lambda k: rs[k])
        X = np.column_stack(list(F.values()))
        g = np.isfinite(X).all(axis=1)
        try:
            p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV()),
                                  X[g], Ifin[g], cv=LeaveOneOut())
            cv = spearmanr(p, Ifin[g]).statistic
        except Exception:
            cv = np.nan
        lab = f"{w//60}분" if w < 3600 else (f"{w/3600:g}시간")
        rows.append((w, rs["값"], rs["비율50"], rs[best], cv, best))
        print(f"  {lab:>8} {rs['값']:>8.3f} {rs['비율50']:>8.3f} {rs[best]:>9.3f}"
              f" {cv:>9.3f}   {best:<10}")

    print("\n" + "-"*76); print(" 목표 상관에 도달하는 최소 측정시간"); print("-"*76)
    for tgt in TARGETS:
        hit = [(w, b) for w, _, _, b, _, _ in rows if b >= tgt]
        if hit:
            w, b = hit[0]
            lab = f"{w//60}분" if w < 3600 else (f"{w/3600:g}시간")
            print(f"  ρ ≥ {tgt:.2f}  →  {lab:>8}  (ρ = {b:.3f})"
                  f"   전체 {t[-1]/3600:.0f}시간 대비 {t[-1]/w:.0f}배 단축")
        else:
            print(f"  ρ ≥ {tgt:.2f}  →  이 데이터 범위에서는 도달 못 함")

    if do_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w_, v_, r_, b_, c_, _ = zip(*rows)
        m = np.array(w_)/60
        plt.figure(figsize=(7,4.2))
        plt.plot(m, v_, "o-", label="value only")
        plt.plot(m, r_, "s-", label="ratio feature")
        plt.plot(m, b_, "^--", label="best single")
        for y, s in [(.7,"usable"), (.9,"sufficient")]:
            plt.axhline(y, ls=":", c="gray", lw=.8); plt.text(m[-1], y+.005, s, ha="right", fontsize=8)
        plt.xscale("log"); plt.xlabel("measurement window [min]")
        plt.ylabel("Spearman rho vs final"); plt.grid(alpha=.3); plt.legend()
        plt.title("How many minutes are enough?")
        plt.tight_layout(); plt.savefig("sweep.png", dpi=140)
        print("\n  그림 저장: sweep.png")

    print("\n" + "="*76)
    print(" 공유용: 위 표와 최소 측정시간.  셀 단위 값은 없습니다.")
    print("="*76)

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else: main(a[0], "--plot" in sys.argv)
