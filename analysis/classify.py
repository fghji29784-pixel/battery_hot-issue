# -*- coding: utf-8 -*-
"""
저전압 불량 판정 — 판정등급(E=불량, A=양품)을 직접 맞히는 분류 평가

  python classify.py "데이터.xlsx"
  python classify.py "데이터.xlsx" --drop=layer
  python classify.py "데이터.xlsx" --ng=E,D          # 불량으로 볼 등급 지정
  python classify.py "데이터.xlsx" --plot
  python classify.py "데이터.xlsx" --no-cond   # 조건 피처(층·온도·전압) 전부 제외

※ 실행 시간 — 셀 수가 많으면 수십 분 걸립니다
  불량이 극소수라 class_weight="balanced" 를 쓰는데, 이때 양성 가중치가
  음성의 수백 배가 되고 부스팅이 그만큼 느려집니다 (실측 60배).
  가중치를 빼거나 early_stopping 을 켜면 빨라지지만 예측이 달라지므로
  (순위 일치 0.09) 그대로 둡니다. 교차검증을 10회 하므로 그 배수만큼입니다.
  빠르게 명부만 보려면 defects.py --no-model 을 쓰십시오.

ΔOCV 회귀가 아니라 실제 판정 라벨을 타깃으로 쓴다.
현장이 내리는 결정과 같은 단위로 평가하기 위함이다.

핵심 지표
  · 상위 k% 를 추가 검사했을 때 불량을 몇 % 잡는가  (검출률)
  · 검출률 90/95/99% 를 달성하려면 몇 %를 검사해야 하는가  ← 운영 판단의 핵심
  · 무작위 대비 몇 배인가 (lift), PR-AUC

비교 대상
  ① 전류 끝값 순위       ② 트레이 정규화 전류 순위       ③ 전체 피처 모델

원본/셀단위 값은 출력하지 않는다.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import runlog
from predict_xlsx import load, I_PAT, SLOPE_PAT, COND_PAT, TRAY_PAT, measured_upto

GRADE_KEY = "판정등급"


def screening_curve(score, y, name):
    """상위 k% 검사 시 검출률과, 목표 검출률 달성에 필요한 검사 비율."""
    n, ng = len(y), int(y.sum())
    order = np.argsort(-score)
    hits  = np.cumsum(y[order])
    print(f"\n  [{name}]   PR-AUC {average_precision_score(y, score):.4f}"
          f"  (기준선 {ng/n:.4f})   ROC-AUC {roc_auc_score(y, score):.4f}")
    print(f"    {'검사 비율':>9}{'검사 셀':>9}{'검출':>10}{'검출률':>9}{'무작위':>8}{'배수':>7}")
    print("    " + "-"*54)
    for q in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(int(round(n*q)), 1)
        h = int(hits[k-1]); exp = ng*q
        print(f"    {q*100:>8.1f}%{k:>9,}{f'{h}/{ng}':>10}{h/ng*100:>8.1f}%"
              f"{exp:>8.1f}{h/max(exp,1e-9):>6.0f}배")
    print(f"    {'목표 검출률':>12} → 필요한 검사 비율")
    for tgt in (0.90, 0.95, 0.99, 1.00):
        need = np.searchsorted(hits, np.ceil(ng*tgt)) + 1
        if need <= n:
            print(f"    {tgt*100:>11.0f}%    {need/n*100:>6.2f}%  ({need:,}셀 검사)")
        else:
            print(f"    {tgt*100:>11.0f}%    도달 불가")
    return average_precision_score(y, score)


def main(spec, do_plot=False, drop=(), ng_codes=("E",), no_cond=False):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c:int(I_PAT.match(c).group(1)))
    scols = sorted([c for c in df.columns if SLOPE_PAT.match(c)], key=lambda c:int(SLOPE_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    dl    = [d.lower() for d in drop]
    conds = [] if no_cond else [c for c in df.columns if COND_PAT.match(c)
             and pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in dl]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if gcol is None:
        print("  !! 판정등급 컬럼을 못 찾았습니다."); return

    D = df.copy()
    D["_g"] = D[gcol].astype(str).str.strip().str.upper()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols+scols+conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)

    print("="*78); print(" 저전압 불량 판정 — 분류 평가"); print("="*78)
    vc = D["_g"].value_counts().sort_index()
    print(f"  판정등급 분포 ({len(D):,}셀)")
    for g, c in vc.items():
        tag = "  ← 불량" if g in ng_codes else ""
        print(f"    {g:<4} {c:>7,}셀  ({c/len(D)*100:>5.2f}%){tag}")
    y = D["_g"].isin(ng_codes).values.astype(int)
    n, ngg, ntray = len(D), int(y.sum()), D["_tray"].nunique()
    print(f"\n  불량 {ngg}개 / 양품 {n-ngg:,}개  (불량률 {ngg/n*100:.3f}%)")
    print(f"  트레이 {ntray}개 / 전류 피처 {len(icols)}개 / 조건 {len(conds)}개")
    if drop: print(f"  제외: {', '.join(drop)}")
    if no_cond:
        print("  [--no-cond] 조건 피처(층·온도·전압·배선저항)를 전부 제외했습니다.")
        print("             전류 형상만으로 얼마나 잡히는지 보기 위한 대조군입니다.")
    if ngg < 5:
        print("\n  !! 불량이 5개 미만이라 평가가 무의미합니다."); return

    for c in icols+scols:
        D[c+"_tn"] = D[c] - D.groupby("_tray")[c].transform("median")

    print("\n" + "-"*78); print(" [A] 방식별 선별 성능"); print("-"*78)
    last = icols[-1]
    screening_curve(D[last].values, y, "① 전류 끝값 순위")
    screening_curve(D[last+"_tn"].values, y, "② 트레이 정규화 전류 순위")

    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = icols+scols+[c+"_tn" for c in icols+scols]+conds
    cv = GroupKFold(n_splits=min(5, max(ntray,2)))
    g = D["_tray"].values
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
                          D[F].values, y, cv=cv, groups=g, method="predict_proba")[:,1]
    ap3 = screening_curve(p, y, "③ 전체 피처 모델")

    # ── 절제 — ③ 의 성능이 전류에서 온 것인가 조건 변수에서 온 것인가 ──
    need = lambda sc: (np.searchsorted(np.cumsum(y[np.argsort(-sc)]), ngg) + 1) / n
    if conds and not no_cond:
        Fc = icols+scols+[c+"_tn" for c in icols+scols]        # 조건 피처 제외
        pc = cross_val_predict(HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
                               D[Fc].values, y, cv=cv, groups=g, method="predict_proba")[:,1]
        apc = average_precision_score(y, pc)
        print("\n" + "-"*78)
        print(" [A-2] ★ 절제 — ③ 의 성능은 전류에서 오는가, 조건 변수에서 오는가")
        print("-"*78)
        print(f"    {'쓴 피처':<20}{'PR-AUC':>9}{'ROC-AUC':>9}{'상위5%':>9}{'상위20%':>9}{'100%검출 검사비율':>18}")
        print("    " + "-"*74)
        for nm_, sc_ in [("전체 (조건 포함)", p), (f"전류만 (조건 {len(conds)}개 제외)", pc)]:
            h = np.cumsum(y[np.argsort(-sc_)])
            print(f"    {nm_:<20}{average_precision_score(y,sc_):>9.4f}{roc_auc_score(y,sc_):>9.4f}"
                  f"{h[max(int(round(n*0.05)),1)-1]:>6}/{ngg}{h[max(int(round(n*0.20)),1)-1]:>6}/{ngg}"
                  f"{need(sc_)*100:>17.2f}%")
        print(f"""
    ★ PR-AUC 가 {'전류만' if apc > ap3 else '전체'} 쪽이 높다 ({max(apc,ap3):.4f} vs {min(apc,ap3):.4f}).
      불량이 희소할 때는 ROC-AUC 보다 PR-AUC 가 믿을 만하다.
      ROC-AUC 는 음성 {n-ngg:,}개를 잘 정렬해도 올라가지만, 우리가 보는 것은
      양성 {ngg}개가 어디 있느냐다.
    ※ 이 표의 차이가 불량 한두 개에서 온다면 결론으로 쓰지 말 것.
      한 개가 {100/max(ngg,1):.1f}%p 다.""")
        if need(pc) > need(p) * 1.3:
            print(f"""
    ☠ '100% 검출에 필요한 검사 비율' 은 조건 피처가 있을 때만 좋다
      ({need(p)*100:.2f}% vs {need(pc)*100:.2f}%). 그 차이는 가장 애매한 불량
      몇 개의 위치에서 나온다. 운영 결론을 이 숫자로 쓰려면, 조건 변수(층·온도·
      전압)가 라인이 바뀌어도 같은 방향으로 작동한다는 근거가 따로 있어야 한다.""")

    print("\n" + "-"*78)
    print(" [B] ★ 몇 분이면 충분한가 — 불량 검출 기준")
    print("-"*78)
    print(f"    {'구간':>6}{'PR-AUC':>9}{'상위1% 검출률':>14}{'상위5% 검출률':>14}")
    print("    " + "-"*44)
    rows = []
    for m in mins:
        if m not in (5,8,10,12,15,20,25,30): continue
        ci = [c for c,mm in zip(icols,mins) if mm <= m]
        cs = [c for c in scols if int(SLOPE_PAT.match(c).group(1)) <= m]
        cols = ci+cs+[c+"_tn" for c in ci+cs]+conds
        pm = cross_val_predict(HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
                               D[cols].values, y, cv=cv, groups=g, method="predict_proba")[:,1]
        hits = np.cumsum(y[np.argsort(-pm)])
        r1 = hits[max(int(round(n*0.01)),1)-1]/ngg
        r5 = hits[max(int(round(n*0.05)),1)-1]/ngg
        ap = average_precision_score(y, pm); rows.append((m, ap, r1, r5))
        print(f"    {m:>4}분{ap:>9.4f}{r1*100:>13.1f}%{r5*100:>13.1f}%")
    if rows:
        bm, bap, _, _ = max(rows, key=lambda r: r[1])
        print(f"\n    ★ PR-AUC 최고 구간: {bm}분 ({bap:.4f})")
        if bm <= rows[0][0] + 2 and bap > rows[-1][1] * 1.2:
            print(f"""      → 가장 짧은 구간이 가장 좋고, 길어질수록 나빠진다 ({bap:.4f} → {rows[-1][1]:.4f}).
        '정착을 기다릴수록 좋아진다' 가 성립하지 않는다는 뜻이다.
        열드리프트가 시간에 따라 쌓이므로 초기 전류가 덜 오염된 신호일 수 있다.
        correct.py 의 시점별 트레이간 분산 비중과 함께 볼 것.""")
        else:
            print("      → 검출률이 포화되는 구간이 최소 필요 측정시간")

    if do_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4.4))
        for sc, nm in [(D[last].values,"current only"),
                       (D[last+"_tn"].values,"tray-normalized"), (p,"full model")]:
            h = np.cumsum(y[np.argsort(-sc)]) / ngg
            plt.plot(np.arange(1,n+1)/n*100, h*100, label=nm)
        plt.plot([0,100],[0,100],"k:",lw=.8,label="random")
        plt.xlim(0,20); plt.xlabel("inspected fraction [%]"); plt.ylabel("defects found [%]")
        plt.legend(); plt.grid(alpha=.3); plt.title(f"Screening curve  (n={n:,}, defects={ngg})")
        plt.tight_layout(); plt.savefig("screening_curve.png", dpi=140)
        print("\n  그림 저장: screening_curve.png")

    print("\n" + "="*78)
    print(""" 읽는 법
  · '검출률 95% 에 필요한 검사 비율' 이 가장 중요한 숫자입니다.
    5% 검사로 95% 를 잡는다면, 나머지 95% 는 3일 보관을 생략할 수 있습니다.
  · ③이 ①보다 나아야 모델이 값어치가 있습니다.
    비슷하면 전류 끝값만으로 충분하다는 뜻이고, 그것도 유효한 결론입니다.""")
    print("="*78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        dr, ng = [], ("E",)
        for x in sys.argv:
            if x.startswith("--drop="): dr = [t.strip() for t in x.split("=",1)[1].split(",")]
            if x.startswith("--ng="):   ng = tuple(t.strip().upper() for t in x.split("=",1)[1].split(","))
        sv, en = runlog.parse(sys.argv)
        with runlog.saving("classify", a[0], sys.argv, sv, en):
            main(a[0], "--plot" in sys.argv, tuple(dr), ng, "--no-cond" in sys.argv)
