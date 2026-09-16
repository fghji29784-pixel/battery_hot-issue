# -*- coding: utf-8 -*-
"""안 잡히는 셀을 살릴 수 있는가 — 트레이 내 공간 보정과 곡선 형상

  python rescue.py "데이터.xlsx"
  python rescue.py "데이터.xlsx" --grid=12x12 --order=col
  python rescue.py "데이터.xlsx" --at=15

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).

────────────────────────────────────────────────────────────────────────
 왜 만들었는가 — 트레이 히트맵이 알려준 것
────────────────────────────────────────────────────────────────────────
 CFDD011456 의 히트맵에서 두 가지가 보였다.

  ① 전류에 뚜렷한 공간 구배가 있다
     A행 쪽은 +4 uA 대, L행 쪽은 음수(-6 uA 까지). 셀 고유 특성이 아니라
     트레이 안의 '자리' 가 전류를 결정하고 있다.

  ② 그 구배가 온도 구배와 정확히 반대다
     dT(=T_start-T_final) 히트맵은 A행 쪽이 0.03~0.05, L행 쪽이 0.11~0.16.
     즉 L행 셀이 더 많이 식고, 그만큼 전류가 낮게 읽힌다.
     rho(dT/dt, I) = +0.485 (부록 9) 를 트레이 하나 안에서 그림으로 본 것이다.

 그리고 셀 번호가 곧 자리다. 140 -> L08 (열-우선 12x12).
 즉 우리는 이미 모든 셀의 트레이 내 물리적 위치를 알고 있다.

 → 온도를 재서 보정하는 대신, '자리' 로 보정할 수 있을지 모른다.
   온도 측정은 잡음이 크지만(셀당 ±0.05 K 요동) 자리는 정확하다.

 ★ 단, 전제가 있다 — 그 구배가 트레이마다 같아야 한다
   트레이 하나의 히트맵만 보고 '자리로 보정하면 된다' 고 말할 수는 없다.
   트레이마다 챔버 위치·투입 방향·대기시간이 다르면 구배도 다르다.
   구배의 부호가 트레이마다 뒤집히면, 전체 공통 보정은 오히려 해가 된다.
   → [2-b] 에서 이 전제를 먼저 검정한다. 전제가 깨지면 트레이별 보정을 쓴다.
     [3] 에서 '전체 공통' 과 '트레이별' 을 나란히 놓고 비교한다.

────────────────────────────────────────────────────────────────────────
 또 하나 — 곡선 형상
────────────────────────────────────────────────────────────────────────
 셀 140 의 전류 곡선은 U자였다. 0 -> -1.8 uA(15분 부근) -> -0.2 uA(30분).
 다른 셀은 단조 증가인데 이 셀만 내려갔다 올라온다.

 열드리프트로 설명된다. 온도가 떨어지는 동안은 음의 인공전류가 실리고,
 온도가 안정되면 그 항이 사라지면서 진짜 자가방전이 드러난다.
 그렇다면 '값' 이 아니라 '회복 기울기' 가 신호일 수 있다.

 값 피처(i_XXmin)로는 이 셀이 양품보다도 낮게 나온다. 형상 피처는 다르다.
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import runlog
from predict_xlsx import (load, I_PAT, COND_PAT, TARGET_PAT, TRAY_PAT,
                          CELL_PAT, measured_upto)
from correct import between_tray_share, within_tray_rho, topk_recall, safe_z

GRADE_KEY = "판정등급"


def grid_pos(num, rows, cols, order):
    """셀 번호를 (행 인덱스, 열 인덱스) 로 되돌린다.

    order='col' : A1,A2,..,A12,B1,..  (열-우선. 140 -> L08)
    order='row' : A1,B1,..,L1,A2,..   (행-우선)
    """
    k = np.asarray(num, float) - 1
    bad = ~np.isfinite(k) | (k < 0) | (k >= rows * cols)
    k = np.where(bad, 0, k).astype(int)
    r, c = (k // cols, k % cols) if order == "col" else (k % rows, k // rows)
    return np.where(bad, -1, r), np.where(bad, -1, c)


def main(spec, at=None, rows=None, cols=None, order="col"):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    conds = [c for c in df.columns if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    tgts  = [c for c in df.columns if TARGET_PAT.search(c)]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    cellc = next((c for c in df.columns if CELL_PAT.match(str(c).strip())), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols or cellc is None:
        print("  !! 전류 컬럼 또는 셀 번호 컬럼을 못 찾았습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    D["_num"] = pd.to_numeric(D[cellc], errors="coerce")
    for c in icols + conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols + ["_num"]).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    n = len(D); g = D["_tray"].values
    if n < 30:
        print(f"\n  !! {mins[-1]}분까지 측정된 셀이 {n}개뿐입니다."); return
    at = at or (15 if 15 in mins else mins[-1])
    acol = f"i_{at}min" if f"i_{at}min" in D.columns else icols[-1]
    y = D[gcol].astype(str).str.strip().str.upper().eq("E").values.astype(int) if gcol else np.zeros(n, int)
    yv = pd.to_numeric(D[tgts[-1]], errors="coerce").values if tgts else None

    print("=" * 78); print(" 안 잡히는 셀을 살릴 수 있는가 — 공간 보정과 곡선 형상"); print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 평가 시점 {at}분"
          + (f" / 불량(E) {int(y.sum())}개" if y.sum() else ""))

    # ── [1] 격자 복원 ────────────────────────────────────────────
    sz = int(D.groupby("_tray").size().median())
    # 격자 크기는 '셀 수' 가 아니라 '셀 번호의 최대값' 으로 잡아야 한다.
    # 결측이 있으면 셀 수가 격자보다 작고, 그 수가 소수면 엉뚱한 격자가 나온다.
    # (142 셀 -> 2 x 71 이 되지만 실제 격자는 12 x 12 = 144 이다)
    mx = int(np.nanmax(D.groupby("_tray")["_num"].max().values))
    if rows is None or cols is None:
        cand = [(abs(r - mx / r), r, mx // r) for r in range(1, mx + 1) if mx % r == 0]
        _, rows, cols = min(cand) if cand else (0, 1, mx)
    print("\n" + "-" * 78); print(" [1] 셀 번호에서 트레이 내 자리 복원"); print("-" * 78)
    print(f"    트레이당 셀 수 중앙 {sz},  셀 번호 최대 {mx}")
    print(f"    →  격자 {rows} x {cols}  (순서: {order}-우선)")
    if rows < 3 or cols < 3:
        print(f"""    ☠ 격자가 {rows} x {cols} 로 한쪽이 지나치게 얇다. 셀 번호 최대값 {mx} 이
      소수이거나 그에 가까워서 생기는 일이다. 실제 격자를 --grid 로 지정할 것.
      (예: --grid=12x12).  아래 [2] 이후 수치는 격자가 맞아야 의미가 있다.""")
    elif abs(rows - cols) > max(rows, cols) / 2:
        print(f"    ※ 격자가 {rows} x {cols} 로 한쪽이 길다. 실제와 다르면 --grid 로 지정할 것.")
    R, C = grid_pos(D["_num"].values, rows, cols, order)
    D["_r"], D["_c"] = R, C
    ok = (R >= 0).mean()
    print(f"    번호가 격자 범위 안인 셀 {ok*100:.1f}%"
          + ("" if ok > 0.98 else "   ← 격자 크기나 순서가 틀렸을 수 있다. --grid / --order 로 지정할 것"))
    ex = D.index[D["_num"] == 140]
    if len(ex):
        i = ex[0]
        print(f"    검산: 셀 번호 140 → {chr(65+int(D['_r'][i]))}{int(D['_c'][i])+1:02d}"
              "   (히트맵 제목이 L08 이면 맞다)")

    # ── [2] 공간 구배 ────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 자리가 전류를 얼마나 설명하는가  (트레이 기준선을 뺀 뒤)")
    print("-" * 78)
    tn = lambda v: v - pd.Series(v).groupby(g).transform("median").values
    tnI = tn(D[acol].values)
    tot = float(np.nanvar(tnI))

    def share(key):
        m = pd.Series(tnI).groupby(D[key].values).transform("mean").values
        return float(np.nanvar(m) / tot) if tot > 0 else float("nan")

    print(f"    {'구분':<22}{'분산 설명 비중':>14}")
    print("    " + "-" * 36)
    print(f"    {'행 (A~)':<22}{share('_r')*100:>13.1f}%")
    print(f"    {'열 (1~)':<22}{share('_c')*100:>13.1f}%")
    D["_rc"] = D["_r"] * 1000 + D["_c"]
    print(f"    {'자리 (행x열)':<22}{share('_rc')*100:>13.1f}%")
    if "layer" in D.columns:
        print(f"    {'층 (참고)':<22}{share('layer')*100:>13.1f}%")

    prof = pd.Series(tnI).groupby(D["_r"].values).median()
    print(f"\n    행별 전류 (트레이 중앙값 대비)")
    for r_, v_ in prof.items():
        if r_ < 0: continue
        bar = "█" * int(min(abs(v_) / (prof.abs().max() + 1e-300) * 24, 24))
        print(f"      {chr(65+int(r_))}  {v_:>12.4g}  {'' if v_>=0 else '-'}{bar}")
    if "t_init" in D.columns and "t_final" in D.columns:
        dT = tn((D["t_init"] - D["t_final"]).values)
        pr2 = pd.Series(dT).groupby(D["_r"].values).median()
        from scipy.stats import spearmanr
        rr = spearmanr(prof.values, pr2.reindex(prof.index).values).statistic
        print(f"\n    행별 전류 와 행별 냉각량(T_start-T_final) 의 순위상관 = {rr:+.3f}")
        print("      → 음수면 '많이 식은 행일수록 전류가 낮다' 는 뜻이다. 열드리프트와 부합.")

    # ── [2-b] 그 구배가 트레이마다 같은가 ─────────────────────────
    print("\n" + "-" * 78)
    print(" [2-b] ★ 그 구배가 트레이마다 같은가  —  자리 보정의 전제")
    print("-" * 78)
    trays = list(pd.unique(g))
    B = np.full((len(trays), 2), np.nan)          # 트레이별 (행 기울기, 열 기울기)
    for k, t in enumerate(trays):
        i = np.where((g == t) & (R >= 0))[0]
        if len(i) < 20: continue
        A = np.c_[np.ones(len(i)), R[i], C[i]].astype(float)
        try:
            coef, *_ = np.linalg.lstsq(A, tnI[i], rcond=None)
            B[k] = coef[1:]
        except np.linalg.LinAlgError:
            pass
    fin = np.isfinite(B).all(1)
    print(f"    트레이마다 전류에 평면을 맞춘다:  I = a + b x 행 + c x 열")
    print(f"    맞춘 트레이 {int(fin.sum())} / {len(trays)}개")
    # 전체 행 프로파일이 직선으로 설명되는가. 안 되면 평면 모형 자체가 부적합하다.
    gp0 = pd.Series(tnI).groupby(R).mean()
    gp0 = gp0[gp0.index >= 0]
    if len(gp0) >= 4:
        xr = gp0.index.values.astype(float); yr = gp0.values
        pr = np.polyval(np.polyfit(xr, yr, 1), xr)
        r2p = 1 - np.sum((pr - yr) ** 2) / max(np.sum((yr - yr.mean()) ** 2), 1e-300)
        print(f"    전체 행 프로파일을 직선이 설명하는 정도  R2 = {r2p:.3f}")
        if r2p < 0.3:
            print(f"""    ☠ 직선으로 설명이 안 된다. 프로파일이 단조 구배가 아니라는 뜻이다
      (양 끝이 낮고 가운데가 높은 U자 등). 그러면 아래 '기울기 부호 일치율' 은
      의미가 없다 — 직선 기울기가 0 근처라 부호를 잡음이 정한다.
      ★ 이 경우 판정은 기울기가 아니라 '프로파일 상관' 으로 해야 한다.\n""")
        else:
            print()
    else:
        print()
    print(f"    {'기울기':<10}{'중앙':>13}{'5~95%':>26}{'같은 부호':>11}")
    print("    " + "-" * 60)
    for j, nm in ((0, "행 방향 b"), (1, "열 방향 c")):
        v = B[fin, j]
        if len(v) == 0: continue
        same = max(np.mean(v > 0), np.mean(v < 0))
        print(f"    {nm:<10}{np.median(v):>13.3g}"
              f"{f'{np.percentile(v,5):.3g} ~ {np.percentile(v,95):.3g}':>26}"
              f"{same*100:>10.0f}%")
    b_same = max(np.mean(B[fin, 0] > 0), np.mean(B[fin, 0] < 0)) if fin.any() else float("nan")

    # 행 프로파일의 공통 성분 대 트레이별 성분
    rowg = pd.Series(tnI).groupby(R).transform("mean").values
    rowt = pd.Series(tnI).groupby([pd.Series(g), pd.Series(R)]).transform("mean").values
    tot = float(np.nanvar(tnI))
    v_com = float(np.nanvar(rowg)) / tot if tot > 0 else np.nan
    v_dev = float(np.nanvar(rowt - rowg)) / tot if tot > 0 else np.nan
    print(f"\n    행 프로파일의 분산 분해 (트레이 기준선을 뺀 전류 기준)")
    print(f"      모든 트레이에 공통인 구배    {v_com*100:>6.1f}%")
    print(f"      트레이마다 다른 부분        {v_dev*100:>6.1f}%")

    from scipy.stats import spearmanr as _sp
    cors = []
    gp = pd.Series(tnI).groupby(R).mean()
    for t in trays:
        i = np.where((g == t) & (R >= 0))[0]
        if len(i) < 20: continue
        tp = pd.Series(tnI[i]).groupby(R[i]).mean()
        common = tp.index.intersection(gp.index)
        if len(common) < 4: continue
        r_ = _sp(tp[common].values, gp[common].values).statistic
        if np.isfinite(r_): cors.append(r_)
    if cors:
        cors = np.array(cors)
        print(f"\n    트레이별 행 프로파일 과 전체 평균 프로파일의 순위상관")
        print(f"      중앙 {np.median(cors):.3f}   하위10% {np.percentile(cors,10):.3f}"
              f"   음수인 트레이 {int((cors<0).sum())} / {len(cors)}개")

    # 판정은 프로파일 상관으로 한다. 기울기 부호는 프로파일이 단조일 때만 뜻이 있다.
    if len(cors):
        neg = int((cors < 0).sum()); med = float(np.median(cors))
        if med > 0.4 and neg <= max(1, len(cors) // 10):
            print(f"""
    ★ 전제가 성립하는 쪽이다. 트레이별 프로파일이 전체 평균과 중앙 {med:.2f} 로
      같은 방향이고, 반대인 트레이는 {neg}/{len(cors)}개뿐이다.
      → 전체 공통 보정(②)을 먼저 볼 것. [3] 에서 ②와 ②' 를 비교한다.""")
        elif med > 0.2:
            print(f"""
    △ 전제가 반쯤 성립한다. 프로파일 상관 중앙 {med:.2f}, 반대인 트레이 {neg}/{len(cors)}개.
      → ② 와 ②' 중 어느 쪽이 나은지는 [3] 의 실측으로만 정해진다.
        둘 다 보고 고를 것.""")
        else:
            print(f"""
    ☠ 전제가 약하다. 프로파일 상관 중앙이 {med:.2f} 로 낮고 반대인 트레이가
      {neg}/{len(cors)}개다. 전체 공통 보정은 방향이 반대인 트레이를 망친다.
      → 트레이별 보정(②'/②'')을 쓸 것.""")
    print(f"      (참고) 공통 성분 {v_com*100:.1f}% 대 트레이별 {v_dev*100:.1f}%")
    if len(cors) and v_dev > v_com:
        print(f"""
    ★★ 두 지표를 합쳐 읽을 것
      프로파일 상관이 높은데(중앙 {float(np.median(cors)):.2f}) 트레이별 분산이
      공통보다 크다({v_dev*100:.1f}% > {v_com*100:.1f}%).
      모순이 아니다 — '모양은 트레이마다 같은데 크기가 다르다' 는 뜻이다.
      → 그러면 전체 공통 보정(②)도 트레이별 자유 보정(②'')도 최선이 아니다.
        모양은 전체에서 한 번 추정하고 크기만 트레이마다 맞추면 된다.
        [3] 의 ②* 가 그것이다. 트레이당 파라미터가 하나뿐이라
        진짜 불량을 지울 위험이 ②'' 보다 훨씬 작다.""")
    print("""
    ※ 트레이별 보정에는 대가가 있다. 그 트레이 안의 진짜 공간 불량
      (예: 특정 자리만 실제로 나쁜 경우)까지 같이 지워진다.
      [4] 의 행 분포와 함께 읽을 것.""")

    # ── [3] 점수 비교 ────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [3] 공간 보정과 곡선 형상이 검출을 바꾸는가")
    print("-" * 78)
    # ② 전체 공통 보정 — 모든 트레이가 같은 구배를 갖는다고 본다
    pos = pd.Series(tnI).groupby(D["_rc"].values).transform("median").values
    sp = tnI - pos

    # ②' 트레이별 보정 — 트레이마다 평면을 따로 맞춰 뺀다
    #    구배가 트레이마다 다를 때 쓴다. 대신 그 트레이의 진짜 공간 불량도 지운다.
    sp_t = tnI.copy()
    for k, t in enumerate(trays):
        i = np.where((g == t) & (R >= 0))[0]
        if len(i) < 20 or not np.isfinite(B[k]).all(): continue
        pl = B[k, 0] * R[i] + B[k, 1] * C[i]
        sp_t[i] = tnI[i] - (pl - np.mean(pl))

    # ②'' 트레이별 '프로파일' 보정 — 평면이 아니라 행별 중앙값을 뺀다.
    #    프로파일이 U자처럼 단조가 아니면 평면으로는 못 따라간다.
    #   ※ 한 묶음에 셀이 몇 개 없으면 중앙값이 그 셀 자신이 되어 점수가 0 이 된다.
    #     불량이 그렇게 숨을 수 있으므로, 5개 미만인 묶음은 보정하지 않는다.
    key = [pd.Series(g), pd.Series(R)]
    prof_t = pd.Series(tnI).groupby(key).transform("median").values
    cnt_t = pd.Series(tnI).groupby(key).transform("count").values
    sp_p = np.where(cnt_t >= 5, tnI - np.nan_to_num(prof_t, nan=0.0), tnI)
    n_small = int(np.sum(cnt_t < 5))

    # ②* 공통 모양 x 트레이별 크기 — 트레이당 파라미터 하나
    #   [2-b] 가 '모양은 트레이마다 같은데 크기는 다르다' 를 가리켰다
    #   (프로파일 상관 중앙 0.62 인데 트레이별 분산이 공통의 두 배).
    #   그러면 모양은 전체에서 한 번 추정하고, 크기만 트레이마다 맞추면 된다.
    #   구배가 약한 트레이는 alpha 가 작게 나와 보정이 저절로 약해진다.
    gprof = pd.Series(tnI).groupby(R).median()
    Pcell = pd.Series(R).map(gprof).values.astype(float)
    Pcell = np.nan_to_num(Pcell, nan=0.0)
    sp_a = tnI.copy(); alphas = {}
    for t in trays:
        i = np.where((g == t) & (R >= 0))[0]
        if len(i) < 20: continue
        x = Pcell[i]; d = float(x @ x)
        if d <= 0: continue
        a_ = float((tnI[i] @ x) / d)
        alphas[t] = a_
        sp_a[i] = tnI[i] - a_ * x

    # 곡선 형상: 후반 기울기와 최저점 대비 회복량
    V = D[icols].values
    lateA = next((j for j, m in enumerate(mins) if m >= at - 10), 0)
    lateB = next((j for j, m in enumerate(mins) if m >= at), len(mins) - 1)
    span = max(mins[lateB] - mins[lateA], 1)
    late = (V[:, lateB] - V[:, lateA]) / span
    upto = np.array([j for j, m in enumerate(mins) if m <= at])
    rec = V[:, upto[-1]] - V[:, upto].min(1)

    cand = [("① 트레이 정규화 전류", tnI),
            ("② 자리 보정 (전체 공통)", sp),
            ("②' 자리 보정 (트레이별 평면)", sp_t),
            ("②'' 자리 보정 (트레이별 행프로파일)", sp_p),
            ("②* 공통 모양 x 트레이별 크기", sp_a),
            (f"③ 후반 기울기 ({mins[lateA]}~{mins[lateB]}분)", late),
            ("④ 후반 기울기 + 자리보정", late - pd.Series(late).groupby(D["_rc"].values).transform("median").values),
            ("⑤ 최저점 대비 회복량", rec),
            ("⑥ 자리보정 전류 + 회복량", safe_z(np.c_[sp, rec])[0].sum(1))]
    hdr = f"    {'점수':<28}{'트레이간':>9}"
    if yv is not None: hdr += f"{'트레이내rho':>12}"
    if y.sum(): hdr += f"{'상위1%':>8}{'상위5%':>8}{'최악셀 검사율':>14}"
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    worst = {}
    for nm, s in cand:
        line = f"    {nm:<28}{between_tray_share(s, g)*100:>8.1f}%"
        if yv is not None: line += f"{within_tray_rho(s, yv, g)[0]:>12.3f}"
        if y.sum():
            t = topk_recall(s, y)
            need = max((np.sum(s > s[i]) + 1) / n for i in np.where(y == 1)[0])
            worst[nm] = need
            line += f"{t[0.01]*100:>7.0f}%{t[0.05]*100:>7.0f}%{need*100:>13.2f}%"
        print(line)
    print("""
    → '최악셀 검사율' 이 이 표의 목적이다. 불량을 전부 잡는 데 필요한 검사 비율이며,
      안 잡히는 한 셀이 어디 있는지가 그대로 드러난다.""")

    # ── [4] 불량 셀의 자리 ───────────────────────────────────────
    if y.sum():
        print("\n" + "-" * 78); print(" [4] 불량 셀은 트레이 어디에 있는가"); print("-" * 78)
        idx = np.where(y == 1)[0]
        print(f"    {'셀':>6}{'자리':>7}{'트레이':>18}{'트레이대비 전류':>16}{'후반기울기':>12}")
        print("    " + "-" * 59)
        for i in idx:
            print(f"    {int(D['_num'][i]):>6}{chr(65+int(D['_r'][i]))+str(int(D['_c'][i])+1).zfill(2):>7}"
                  f"{D['_tray'][i]:>18}{tnI[i]:>16.4g}{late[i]:>12.4g}")
        rc = pd.Series([chr(65+int(r)) for r in D["_r"][idx]]).value_counts()
        print(f"\n    행 분포: " + ", ".join(f"{k}{v}개" for k, v in rc.items()))
        allr = pd.Series([chr(65+int(r)) for r in D["_r"]]).value_counts(normalize=True)
        top = rc.index[0]
        print(f"      {top}행에 {rc.iloc[0]}/{len(idx)}개. 전체에서 {top}행 비중은 {allr[top]*100:.1f}%.")
        print("""      → 특정 행에 몰려 있으면 그 자리가 불량을 만드는지(진짜),
        아니면 그 자리가 측정을 왜곡해 불량으로 보이게 하는지(가짜) 갈라야 한다.
        [2] 의 냉각량 상관이 그 단서다.""")

    # ── [5] 시점별 안정성 ────────────────────────────────────────
    if y.sum() and len(mins) > 3:
        print("\n" + "-" * 78)
        print(" [5] 그 결과가 시점을 바꿔도 버티는가  (최악셀 검사율)")
        print("-" * 78)
        need = lambda sc: max((np.sum(sc > sc[i]) + 1) / n for i in np.where(y == 1)[0])
        h3 = "②'' 트레이별행"
        print(f"    {'구간':>6}{'① 트레이정규화':>15}{'② 전체공통':>12}{h3:>16}{'②* 공통x크기':>14}")
        print("    " + "-" * 57)
        for m in mins:
            if m not in (5, 8, 10, 12, 15, 20, 25, 30) or m > mins[-1]: continue
            cm = f"i_{m}min"
            if cm not in D.columns: continue
            v = D[cm].values.astype(float)
            t0 = v - pd.Series(v).groupby(g).transform("median").values
            s2 = t0 - pd.Series(t0).groupby(D["_rc"].values).transform("median").values
            pt = pd.Series(t0).groupby(key).transform("median").values
            ct = pd.Series(t0).groupby(key).transform("count").values
            s3 = np.where(ct >= 5, t0 - np.nan_to_num(pt, nan=0.0), t0)
            gpm = pd.Series(t0).groupby(R).median()
            Pc = np.nan_to_num(pd.Series(R).map(gpm).values.astype(float), nan=0.0)
            s4 = t0.copy()
            for t_ in trays:
                i_ = np.where((g == t_) & (R >= 0))[0]
                if len(i_) < 20: continue
                x_ = Pc[i_]; d_ = float(x_ @ x_)
                if d_ > 0: s4[i_] = t0[i_] - float((t0[i_] @ x_) / d_) * x_
            print(f"    {m:>4}분{need(t0)*100:>14.2f}%{need(s2)*100:>11.2f}%"
                  f"{need(s3)*100:>15.2f}%{need(s4)*100:>13.2f}%")
        print("""
    → 한 시점에서만 좋고 다른 시점에서 무너지면 우연일 가능성이 크다.
      여러 시점에서 일관되게 낮아야 믿을 수 있는 결과다.""")
        if n_small:
            print(f"    ※ (트레이,행) 묶음 중 셀 5개 미만이라 보정하지 않은 셀 {n_small}개")

    print("\n" + "=" * 78)
    print(""" 이 스크립트가 시험하는 것
   1. 셀 번호로 트레이 내 자리를 복원할 수 있다 (히트맵이 근거).
   2. 자리로 보정하면 온도를 재지 않고도 열드리프트를 지울 수 있는가.
      단 [2-b] 의 전제 검정을 먼저 통과해야 한다. 구배가 트레이마다
      다르면 전체 공통 보정(②)은 해가 되고 트레이별 보정(②')을 써야 한다.
   3. 값이 아니라 곡선 형상(회복 기울기)이 신호일 수 있다.
      열드리프트에 눌린 셀은 값은 낮아도 회복은 빠르다.
 [3] 의 '최악셀 검사율' 이 ① 보다 뚜렷이 낮아지면 그 방법이 답이다.""")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        at, rw, cl, od = None, None, None, "col"
        for x in sys.argv:
            if x.startswith("--at="): at = int(x.split("=")[1])
            if x.startswith("--order="): od = x.split("=")[1].strip().lower()[:3]
            if x.startswith("--grid="):
                m = re.match(r"(\d+)\s*[xX*]\s*(\d+)", x.split("=")[1])
                if m: rw, cl = int(m.group(1)), int(m.group(2))
        sv, en = runlog.parse(sys.argv)
        with runlog.saving("rescue", a[0], sys.argv, sv, en):
            main(a[0], at, rw, cl, od)
