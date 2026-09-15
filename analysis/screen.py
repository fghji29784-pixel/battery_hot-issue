# -*- coding: utf-8 -*-
"""
선별(screening) 관점 평가 — ΔOCV 값을 맞히는 대신, 나쁜 셀을 골라내는 성능을 잰다.

  python screen.py "데이터.xlsx"
  python screen.py "데이터.xlsx" --drop=layer
  python screen.py "데이터.xlsx" --plot
  python screen.py "데이터.xlsx" --no-cond   # 조건 피처(층·온도·전압) 전부 제외

왜 평가를 바꾸는가
  회귀(R²)는 모든 셀의 값을 맞히려 하지만, 우리 목적은 상위 소수를 골라내는 것이다.
  실제 데이터에서 R² 는 음수인데 상위 0.5% 재현은 무작위 대비 60배가 넘었다.
  즉 지표가 목적과 어긋나 있었다. 여기서는 선별 성능만 본다.

비교 대상
  ① 전류 끝값 순위            — 현행에 가장 가까운 방식
  ② 트레이 정규화 전류 순위     — 트레이 차이만 보정
  ③ 전체 피처 모델            — 전류·기울기·전압·온도 전부 사용
  세 가지를 같은 기준으로 비교해, 모델이 실제로 이득을 주는지 본다.

원본/셀단위 값은 출력하지 않는다.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import hypergeom
import runlog
from predict_xlsx import load, I_PAT, SLOPE_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, measured_upto

KS = [0.002, 0.005, 0.01, 0.02, 0.05]


def lift_table(score, y, n, label):
    """상위 k% 안에 실제 상위 k% 가 몇 개 들어오는가."""
    out = []
    for q in KS:
        k = max(int(round(n*q)), 1)
        a = set(np.argsort(-y)[:k]); b = set(np.argsort(-score)[:k])
        hit = len(a & b); exp = k*k/n
        p = hypergeom.sf(hit-1, n, k, k) if hit else 1.0
        out.append((q, k, hit, exp, hit/max(exp,1e-9), p))
    print(f"\n  [{label}]")
    print(f"    {'상위':>7}{'셀 수':>7}{'일치':>9}{'무작위':>9}{'배수':>8}{'p':>11}")
    print("    " + "-"*51)
    for q,k,hit,exp,lf,p in out:
        print(f"    {q*100:>6.1f}%{k:>7}{f'{hit}/{k}':>9}{exp:>9.2f}{lf:>7.0f}배{p:>11.1e}")
    return out


def main(spec, do_plot=False, drop=(), no_cond=False):
    df, nfile = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c:int(I_PAT.match(c).group(1)))
    scols = sorted([c for c in df.columns if SLOPE_PAT.match(c)], key=lambda c:int(SLOPE_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    dl    = [d.lower() for d in drop]
    conds = [] if no_cond else [c for c in df.columns if COND_PAT.match(c)
             and pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in dl]
    tcol  = [c for c in df.columns if TARGET_PAT.search(c)][-1]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)

    D = df.copy()
    D["_y"] = pd.to_numeric(D[tcol], errors="coerce")
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols+scols+conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=["_y"]+icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    n, ntray = len(D), D["_tray"].nunique()

    print("="*78); print(" 선별 성능 평가"); print("="*78)
    print(f"  {n:,}셀 / 트레이 {ntray}개 / 전류 피처 {len(icols)}개 / 조건 {len(conds)}개")
    if drop: print(f"  제외: {', '.join(drop)}")
    if no_cond:
        print("  [--no-cond] 조건 피처(층·온도·전압·배선저항)를 전부 제외했습니다.")
        print("             전류 형상만으로 얼마나 잡히는지 보기 위한 대조군입니다.")

    # 트레이 정규화
    for c in icols+scols:
        D[c+"_tn"] = D[c] - D.groupby("_tray")[c].transform("median")
    D["_yn"] = D["_y"] - D.groupby("_tray")["_y"].transform("median")

    y_abs, y_nrm = D["_y"].values, D["_yn"].values
    last = icols[-1]

    print("\n" + "-"*78); print(" [A] 절대 ΔOCV 상위를 얼마나 잡는가"); print("-"*78)
    lift_table(D[last].values, y_abs, n, "① 전류 끝값 순위")
    lift_table(D[last+"_tn"].values, y_abs, n, "② 트레이 정규화 전류 순위")

    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = icols+scols+[c+"_tn" for c in icols+scols]+conds
    cv = GroupKFold(n_splits=min(5, max(ntray,2)))
    g = D["_tray"].values
    p_abs = cross_val_predict(HistGradientBoostingRegressor(random_state=0),
                              D[F].values, y_abs, cv=cv, groups=g)
    lift_table(p_abs, y_abs, n, "③ 전체 피처 모델")

    print("\n" + "-"*78); print(" [B] 트레이 정규화 ΔOCV 상위를 얼마나 잡는가"); print("-"*78)
    print("     (트레이 안에서 유난히 나쁜 셀 = 현행 판정이 보는 대상)")
    lift_table(D[last].values, y_nrm, n, "① 전류 끝값 순위")
    lift_table(D[last+"_tn"].values, y_nrm, n, "② 트레이 정규화 전류 순위")
    p_nrm = cross_val_predict(HistGradientBoostingRegressor(random_state=0),
                              D[F].values, y_nrm, cv=cv, groups=g)
    lift_table(p_nrm, y_nrm, n, "③ 전체 피처 모델")

    # ── 구간별 선별 성능 ──────────────────────────────────────────────
    print("\n" + "-"*78)
    print(" [C] ★ 몇 분이면 충분한가 — 선별 기준으로 다시 측정")
    print("-"*78)
    print(f"    {'구간':>6}{'상위0.5% 배수':>14}{'상위1% 배수':>12}{'상위5% 배수':>12}")
    print("    " + "-"*44)
    for m in mins:
        if m not in (5,8,10,12,15,20,25,30): continue
        ci = [c for c,mm in zip(icols,mins) if mm <= m]
        cs = [c for c in scols if int(SLOPE_PAT.match(c).group(1)) <= m]
        cols = ci+cs+[c+"_tn" for c in ci+cs]+conds
        p = cross_val_predict(HistGradientBoostingRegressor(random_state=0),
                              D[cols].values, y_nrm, cv=cv, groups=g)
        lf = []
        for q in (0.005, 0.01, 0.05):
            k = max(int(round(n*q)),1)
            a=set(np.argsort(-y_nrm)[:k]); b=set(np.argsort(-p)[:k])
            lf.append(len(a&b)/max(k*k/n,1e-9))
        print(f"    {m:>4}분{lf[0]:>13.0f}배{lf[1]:>11.0f}배{lf[2]:>11.0f}배")
    print("\n    → 선별 목적에서는 이 표가 '몇 분'의 답입니다")

    print("\n" + "="*78)
    print(" 해석 지침")
    print("="*78)
    print("""  · 모델(③)이 단순 순위(①②)보다 배수가 높아야 모델이 값어치가 있습니다.
  · ①과 ③이 비슷하면, 복잡한 모델 없이 전류 끝값만으로 충분하다는 뜻입니다.
  · [B]가 [A]보다 배수가 크면, 판정을 트레이 내 상대값으로 하는 것이
    옳다는 근거가 됩니다.""")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        dr = []
        for x in sys.argv:
            if x.startswith("--drop="): dr = [t.strip() for t in x.split("=",1)[1].split(",")]
        sv, en = runlog.parse(sys.argv)
        with runlog.saving("screen", a[0], sys.argv, sv, en):
            main(a[0], "--plot" in sys.argv, tuple(dr), "--no-cond" in sys.argv)
