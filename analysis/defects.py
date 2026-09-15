# -*- coding: utf-8 -*-
"""불량 셀 카드 — 어느 셀이 안 잡히는지 이름으로 본다.

  python defects.py "데이터.xlsx"
  python defects.py "데이터.xlsx" --at=15 --ng=E
  python defects.py "데이터.xlsx" --no-model   # 모델 점수 생략 (빠르게 명부만)

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).

────────────────────────────────────────────────────────────────────────
 ※ 이 스크립트만 예외다 — 셀 단위 값을 출력한다
────────────────────────────────────────────────────────────────────────
 다른 스크립트는 전부 '원본·셀단위 값은 출력하지 않는다' 를 지킨다.
 공유를 전제로 만들었기 때문이다.

 그런데 불량이 8개뿐이고 그중 한두 개가 계속 안 잡히는 상황에서는,
 '87.5%' 라는 숫자보다 '어느 셀이 왜 안 잡히는가' 가 훨씬 중요하다.
 그 셀의 트레이·층·온도·곡선을 직접 봐야 원인이 나온다.

 ★ 그러므로 이 출력은 내부 검토용이다. 발표 자료에 그대로 넣지 말 것.
   트레이 ID 와 셀 ID 가 그대로 찍힌다.

────────────────────────────────────────────────────────────────────────
 무엇을 보는가
────────────────────────────────────────────────────────────────────────
 [1] 불량 셀 명부 — 트레이·셀ID·층·ΔOCV·트레이 내 순위
 [2] 점수별로 몇 번째인가 — 원시 전류 / 트레이 정규화 / 물리보정 / 모델
     각 점수에서 '이 셀을 잡으려면 몇 % 를 검사해야 하는가' 를 낸다
 [3] ★ 가장 안 잡히는 셀은 다른 불량과 무엇이 다른가
 [4] 트레이별 불량 분포 — 한 트레이에 몰려 있는가
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import runlog
from predict_xlsx import (load, I_PAT, SLOPE_PAT, COND_PAT, TARGET_PAT,
                          TRAY_PAT, CELL_PAT, measured_upto)
from correct import between_tray_share

GRADE_KEY = "판정등급"
DUMMY_PAT = re.compile(r"^dummy[_\s]*l\d+$", re.I)   # layer 와 중복이라 표에서 뺀다


def pct_rank(score, i):
    """점수 내림차순에서 i 번 셀의 백분위. 낮을수록 먼저 검사된다."""
    return (np.sum(score > score[i]) + 1) / len(score) * 100


def main(spec, at=None, ng_codes=("E",), no_model=False):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    scols = sorted([c for c in df.columns if SLOPE_PAT.match(c)], key=lambda c: int(SLOPE_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    conds = [c for c in df.columns if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    tgts  = [c for c in df.columns if TARGET_PAT.search(c)]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    cell  = next((c for c in df.columns if CELL_PAT.match(str(c).strip())), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols or gcol is None:
        print("  !! 전류 컬럼 또는 판정등급 컬럼을 못 찾았습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    D["_cell"] = D[cell].astype(str) if cell else pd.Series(np.arange(len(D))).astype(str)
    for c in icols + scols + conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    n = len(D); g = D["_tray"].values
    if n < 30:
        print(f"\n  !! {mins[-1]}분까지 측정된 셀이 {n}개뿐입니다."); return

    at = at or (15 if 15 in mins else mins[-1])
    acol = f"i_{at}min" if f"i_{at}min" in D.columns else icols[-1]
    D["_g"] = D[gcol].astype(str).str.strip().str.upper()
    y = D["_g"].isin(ng_codes).values
    idx = np.where(y)[0]

    print("=" * 78); print(" 불량 셀 카드 — 어느 셀이 안 잡히는가"); print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 평가 시점 {at}분")
    print(f"  불량({'/'.join(ng_codes)}) {len(idx)}개")
    print("  ★ 이 출력은 내부 검토용입니다. 셀ID·트레이ID 가 그대로 나옵니다.")
    if len(idx) == 0:
        print("\n  !! 해당 등급의 셀이 없습니다."); return

    # ── 점수 만들기 ──────────────────────────────────────────────
    tn = lambda v: v - pd.Series(v).groupby(g).transform("median").values
    scores = [("원시 전류", D[acol].values), ("트레이 정규화", tn(D[acol].values))]

    # 물리보정 P2 (correct.py 와 같은 방식, 온도 컬럼이 있을 때만)
    if "t_init" in D.columns and "t_final" in D.columns:
        D["_dTdt"] = (D["t_final"] - D["t_init"]) / float(mins[-1])
        D["_Tmean"] = (D["t_final"] + D["t_init"]) / 2.0
        cov = [c for c in ["_dTdt", "_Tmean", "v_init", "rwiring"] if c in D.columns]
        cov = [c for c in cov if D[c].notna().sum() > 0 and D[c].nunique() > 2]
        T = D.groupby("_tray")[cov + [acol]].median()
        T = T[np.isfinite(T.values).all(1)]
        if len(T) >= 4 and cov:
            from sklearn.linear_model import RidgeCV
            from correct import safe_z
            Zt, mu, sd, dead = safe_z(T[cov].values.astype(float))
            keep = ~dead
            if keep.any():
                cov = [c for c, k in zip(cov, keep) if k]
                Zt, mu, sd = Zt[:, keep], mu[keep], sd[keep]
                b = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Zt, T[acol].values.astype(float))
                Xc = D.groupby("_tray")[cov].transform("median").values.astype(float)
                ref = np.array([0.0 if c == "_dTdt" else 25.0 if c == "_Tmean"
                                else float(np.nanmedian(D[c])) for c in cov])
                Zc = np.nan_to_num((Xc - ref) / sd, nan=0.0, posinf=0.0, neginf=0.0)
                scores.append(("물리보정 P2", D[acol].values - (Zc @ b.coef_)))

    # 전류만 모델 (조건 피처 제외 — 부록 12 의 결론)
    # 교차검증이라 셀 수가 많으면 몇 분 걸린다. 명부만 빨리 보려면 --no-model.
    try:
        if no_model: raise RuntimeError("--no-model")
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.ensemble import HistGradientBoostingClassifier
        for c in icols + scols:
            D[c + "_tn"] = D[c] - D.groupby("_tray")[c].transform("median")
        F = icols + scols + [c + "_tn" for c in icols + scols]
        cv = GroupKFold(n_splits=min(5, max(D["_tray"].nunique(), 2)))
        scores.append(("전류만 모델", cross_val_predict(
            HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
            D[F].values, y.astype(int), cv=cv, groups=g, method="predict_proba")[:, 1]))
    except Exception as e:
        print(f"  (모델 점수 생략: {e})")

    # ── [1] 명부 ─────────────────────────────────────────────────
    print("\n" + "-" * 78); print(" [1] 불량 셀 명부"); print("-" * 78)
    yv = pd.to_numeric(D[tgts[-1]], errors="coerce").values if tgts else None
    lay = pd.to_numeric(D["layer"], errors="coerce").values if "layer" in D.columns else None
    hdr = f"    {'#':>2} {'트레이':<16}{'셀ID':<14}{'층':>4}"
    if yv is not None: hdr += f"{'ΔOCV':>9}{'트레이내 순위':>14}"
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for k, i in enumerate(idx, 1):
        line = f"    {k:>2} {D['_tray'][i]:<16}{D['_cell'][i]:<14}" \
               f"{(int(lay[i]) if lay is not None and np.isfinite(lay[i]) else 0):>4}"
        if yv is not None:
            same = np.where(g == g[i])[0]
            r = int(np.sum(yv[same] > yv[i]) + 1)
            line += f"{yv[i]:>9.4f}{f'{r}/{len(same)}':>14}"
        print(line)

    # ── [2] 점수별 위치 ──────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 각 점수에서 이 셀을 잡으려면 몇 % 를 검사해야 하는가")
    print("-" * 78)
    hdr = f"    {'#':>2} {'셀ID':<14}" + "".join(f"{nm:>15}" for nm, _ in scores)
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    P = np.zeros((len(idx), len(scores)))
    for k, i in enumerate(idx):
        row = f"    {k+1:>2} {D['_cell'][i]:<14}"
        for j, (_, s) in enumerate(scores):
            P[k, j] = pct_rank(s, i)
            row += f"{P[k, j]:>14.2f}%"
        print(row)
    print(f"    {'':>2} {'--- 최악 ---':<14}" + "".join(f"{P[:, j].max():>14.2f}%" for j in range(len(scores))))
    print("""
    → 각 칸은 '전수의 몇 % 를 검사하면 이 셀이 들어오는가' 다. 작을수록 좋다.
      맨 아래 '최악' 줄이 그 점수로 불량을 전부 잡는 데 필요한 검사 비율이다.""")

    # ── [3] 가장 안 잡히는 셀 ────────────────────────────────────
    best_j = int(np.argmin(P.max(0)))
    worst_k = int(np.argmax(P[:, best_j]))
    wi = idx[worst_k]
    print("\n" + "-" * 78)
    print(f" [3] ★ 가장 안 잡히는 셀  —  {D['_tray'][wi]} / {D['_cell'][wi]}")
    print("-" * 78)
    print(f"    기준 점수: {scores[best_j][0]} (최악값이 가장 작은 점수)")
    print(f"    이 셀은 {P[worst_k, best_j]:.2f}% 지점에 있다."
          f"  나머지 {len(idx)-1}개는 {np.delete(P[:, best_j], worst_k).max():.2f}% 안에 들어온다.")

    others = np.array([i for i in idx if i != wi])
    good = np.where(~y)[0]
    print(f"\n    이 셀이 다른 불량 {len(others)}개와 무엇이 다른가")
    print(f"    {'항목':<16}{'이 셀':>12}{'다른 불량 중앙':>16}{'양품 중앙':>12}")
    print("    " + "-" * 56)
    show = [(f"i_{m}min", f"전류 {m}분") for m in mins if m in (mins[0], 10, at, mins[-1])]
    show += [(c, c) for c in conds if c in D.columns and not DUMMY_PAT.match(c)]
    for c, nm in show:
        if c not in D.columns: continue
        v = D[c].values
        print(f"    {nm:<16}{v[wi]:>12.4g}{np.nanmedian(v[others]):>16.4g}{np.nanmedian(v[good]):>12.4g}")
    tnv = tn(D[acol].values)
    print(f"    {'트레이대비 전류':<16}{tnv[wi]:>12.4g}{np.nanmedian(tnv[others]):>16.4g}{np.nanmedian(tnv[good]):>12.4g}")
    if yv is not None:
        print(f"    {'ΔOCV':<16}{yv[wi]:>12.4g}{np.nanmedian(yv[others]):>16.4g}{np.nanmedian(yv[good]):>12.4g}")
    print("""
    → '이 셀' 이 '다른 불량' 이 아니라 '양품' 쪽에 가까운 항목이 원인 후보다.
      전류가 양품과 같으면, SDM 이 이 셀의 이상을 못 본 것이다.
      ΔOCV 만 크고 전류는 정상이라면 불량 기구가 다를 수 있다
      (자가방전이 아니라 접촉·계측·라벨링 문제 포함).
      ★ 이 셀의 원본 곡선을 직접 볼 것. 그것이 다음 단계다.""")

    # ── [4] 트레이 분포 ──────────────────────────────────────────
    print("\n" + "-" * 78); print(" [4] 불량이 트레이에 몰려 있는가"); print("-" * 78)
    vc = pd.Series(g[idx]).value_counts()
    for t_, c_ in vc.items():
        sz = int((g == t_).sum())
        print(f"    {t_:<16}{c_}개 / {sz}셀"
              + ("   ← 한 트레이에 2개 이상" if c_ >= 2 else ""))
    print(f"    트레이 {len(vc)}개에 불량 {len(idx)}개가 흩어져 있다.")
    if (vc >= 2).any():
        print("""      → 같은 트레이에 몰려 있으면 트레이 상대평가가 불리하다.
        그 트레이의 중앙값 자체가 올라가 서로를 가려주기 때문이다.""")

    print("\n" + "=" * 78)
    print(""" 다음에 할 일
   1. [3] 의 셀 ID 로 원본 SDM 곡선을 찾아 눈으로 볼 것
   2. 그 셀의 공정 이력(층·적재 위치·투입 시각)을 확인할 것
   3. 전류가 양품과 같다면, 그 셀은 SDM 으로 잡을 수 있는 불량이 아니다.
      그렇다면 '검출률 87.5%' 가 아니라 'SDM 으로 잡을 수 있는 것은 7/8' 이
      맞는 표현이고, 남은 1개는 다른 검사로 넘겨야 한다.""")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        at, ng = None, ("E",)
        for x in sys.argv:
            if x.startswith("--at="): at = int(x.split("=")[1])
            if x.startswith("--ng="): ng = tuple(t.strip().upper() for t in x.split("=", 1)[1].split(","))
        sv, en = runlog.parse(sys.argv)
        with runlog.saving("defects", a[0], sys.argv, sv, en):
            main(a[0], at, ng, "--no-model" in sys.argv)
