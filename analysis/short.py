# -*- coding: utf-8 -*-
"""짧은 창에서 더 짜내기 — 5분 선별을 가능하게 하는 것들

  python short.py "데이터.xlsx"
  python short.py "데이터.xlsx" --at=5
  python short.py "데이터.xlsx" --grid=12x12

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).

────────────────────────────────────────────────────────────────────────
 왜 5분이 어려운가
────────────────────────────────────────────────────────────────────────
 자가방전 정착률 = 1 - exp(-t/tau),  tau = 7.45시간 (부록 1 실측)
    5분  1.112%      10분  2.212%      15분  3.300%
 → 5분의 신호는 10분의 정확히 절반이다.
   반면 열드리프트는 5분에서도 이미 크다 (셀 140 의 최저점이 11~15분).
   즉 5분은 '신호는 작고 교란은 큰' 구간이다.

 그러면 방법은 셋뿐이다.
   ① 신호를 키운다      — 장비 설정 (출력저항)
   ② 교란을 줄인다      — 공정 (투입 전 열평형, 적재 방식)
   ③ 있는 데이터를 더 짠다 — 분석.  이 스크립트가 하는 것.

────────────────────────────────────────────────────────────────────────
 ③ 을 두 축으로 나눈다
────────────────────────────────────────────────────────────────────────
 [축 1] 누구와 비교할 것인가 (이웃 정의)
   지금은 '같은 트레이 같은 행 12셀' 의 중앙값을 뺀다.
   이웃을 어떻게 잡느냐에 따라 교란 제거 정도가 달라진다.
     트레이 전체 / 행 / 열 / 행+열 / 인접 8셀 / 인접 24셀
   이웃이 가까울수록 교란이 비슷해 잘 상쇄되고,
   이웃이 많을수록 중앙값이 안정돼 잡음이 준다. 둘은 상충한다.

 [축 2] 창 안의 몇 점을 쓸 것인가
   지금은 끝점 하나만 쓴다. 5분이면 1분 평균 5점이 있는데 1점만 쓰는 셈이다.
     끝점 / 전 시점 평균 / 기울기 / 누적 전하
   여러 점을 합치면 잡음이 줄어든다. 5분처럼 신호가 작을수록 이득이 크다.

 ★ 모든 셀이 같은 시간을 측정하고 같은 계산을 받는다. 동일 과정 제약과 맞는다.

  --target="컬럼명"   3일 ΔOCV 컬럼을 직접 지정 (DOCV 처럼 표기가 다를 때)
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import runlog
from predict_xlsx import load, I_PAT, TARGET_PAT, TRAY_PAT, CELL_PAT, measured_upto, find_targets, parse_target
from correct import target_report, within_tray_rho, topk_recall
from rescue import grid_pos

GRADE_KEY = "판정등급"


def neighbor_med(v, g, R, C, rows, cols, kind):
    """이웃 중앙값. kind 에 따라 이웃 정의가 달라진다."""
    s = pd.Series(v)
    if kind == "tray":  return s.groupby(g).transform("median").values
    if kind == "row":   return s.groupby([pd.Series(g), pd.Series(R)]).transform("median").values
    if kind == "col":   return s.groupby([pd.Series(g), pd.Series(C)]).transform("median").values
    if kind == "cross":                                   # 행 중앙값과 열 중앙값의 평균
        a = s.groupby([pd.Series(g), pd.Series(R)]).transform("median").values
        b = s.groupby([pd.Series(g), pd.Series(C)]).transform("median").values
        return (a + b) / 2.0
    d = 1 if kind == "n8" else 2                          # 인접 8셀 / 24셀
    out = np.full(len(v), np.nan)
    for t in pd.unique(g):
        i = np.where(g == t)[0]
        G = np.full((rows, cols), np.nan)
        ok = i[(R[i] >= 0) & (C[i] >= 0)]
        G[R[ok], C[ok]] = v[ok]
        st = [np.roll(np.roll(G, dr, 0), dc, 1)
              for dr in range(-d, d + 1) for dc in range(-d, d + 1)
              if not (dr == 0 and dc == 0)]
        M = np.nanmedian(np.stack(st), axis=0)
        out[ok] = M[R[ok], C[ok]]
    return np.nan_to_num(out, nan=0.0)


def main(spec, at=None, rows=None, cols=None, order="col"):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    tgts  = find_targets(df)
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    cellc = next((c for c in df.columns if CELL_PAT.match(str(c).strip())), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if not icols or cellc is None or gcol is None:
        print("  !! 전류·셀번호·판정등급 컬럼이 모두 필요합니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    D["_num"] = pd.to_numeric(D[cellc], errors="coerce")
    for c in icols: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols + ["_num"]).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    D = D[np.isfinite(D[icols].values).all(1)].reset_index(drop=True)
    n = len(D); g = D["_tray"].values
    y = D[gcol].astype(str).str.strip().str.upper().eq("E").values.astype(int)
    yv = pd.to_numeric(D[tgts[-1]], errors="coerce").values if tgts else None
    if tgts: target_report(yv, D["_tray"].values, str(tgts[-1]))
    if n < 30 or y.sum() < 3:
        print(f"\n  !! 셀 {n}개 / 불량 {int(y.sum())}개. 평가가 어렵습니다."); return

    mx = int(np.nanmax(D.groupby("_tray")["_num"].max().values))
    if rows is None or cols is None:
        cand_g = [(abs(r - mx / r), r, mx // r) for r in range(1, mx + 1) if mx % r == 0]
        _, rows, cols = min(cand_g) if cand_g else (0, 1, mx)
    R, C = grid_pos(D["_num"].values, rows, cols, order)
    at = at or 5
    ai = int(np.argmin(np.abs(np.array(mins) - at)))
    need = lambda sc: max((np.sum(sc > sc[i]) + 1) / n for i in np.where(y == 1)[0])

    print("=" * 78); print(" 짧은 창에서 더 짜내기 — 5분 선별을 가능하게 하는 것들"); print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 격자 {rows} x {cols} / 불량 {int(y.sum())}개")
    print(f"  대상 창: {mins[0]}~{mins[ai]}분  ({ai + 1}개 시점)")
    if ai + 1 < 3:
        n10 = sum(1 for m in mins if m <= 10)
        print(f"""
  ☠ 이것이 짧은 창의 첫 번째 제약이다 — 쓸 수 있는 점이 {ai + 1}개뿐이다
    데이터가 {mins[0]}분부터 시작하므로 0~{mins[0]}분 구간이 아예 없다.
    {mins[ai]}분 선별은 지금 '셀당 숫자 하나' 로 하는 셈이고, 그 한 점의
    잡음을 줄일 방법이 없다. 10분 창은 {n10}개 점을 평균할 수 있어
    잡음이 약 {n10 ** 0.5:.1f}배 준다.
    → {mins[ai]}분이 10분보다 나쁜 것은 신호 크기(정착률 절반) 만의 문제가 아니라
      쓸 수 있는 점의 개수 때문이기도 하다.
    ★ 0~{mins[0]}분 구간 데이터를 받아오는 것이 {mins[ai]}분 선별의 전제다.
      장비는 이미 재고 있다. 저장·전달만 안 되고 있을 가능성이 크다.
      아래 [축 2] 비교는 그 데이터가 생겼을 때 무엇을 쓸지 미리 보는 것이다.""")

    # ── 축 2: 창 안의 값을 하나로 줄이는 방법들 ──────────────────
    Xw = D[icols[:ai + 1]].values.astype(float)
    tw = np.asarray(mins[:ai + 1], float)
    reducers = [("끝점", Xw[:, -1])]
    if ai >= 1: reducers.append(("전 시점 평균", Xw.mean(1)))
    if ai >= 2:
        L = tw - tw.mean()
        reducers.append(("기울기", (Xw - Xw.mean(1, keepdims=True)) @ L / (L @ L)))
        Q = np.concatenate([np.zeros((n, 1)),
                            np.cumsum((Xw[:, 1:] + Xw[:, :-1]) / 2 * np.diff(tw * 60), 1)], 1)
        reducers.append(("누적 전하", Q[:, -1]))

    KINDS = [("트레이 전체", "tray"), ("행 12셀", "row"), ("열 12셀", "col"),
             ("행+열", "cross"), ("인접 8셀", "n8"), ("인접 24셀", "n24")]

    print("\n" + "-" * 78)
    print(f" [1] 이웃 정의 x 시점 결합  —  최악셀 검사율 (낮을수록 좋다)")
    print("-" * 78)
    hdr = f"    {'이웃':<14}" + "".join(f"{nm:>14}" for nm, _ in reducers)
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    best = (1e9, None)
    for knm, kind in KINDS:
        line = f"    {knm:<14}"
        for rnm, val in reducers:
            sc = val - neighbor_med(val, g, R, C, rows, cols, kind)
            v = need(sc)
            line += f"{v * 100:>13.2f}%"
            if v < best[0]: best = (v, f"{knm} x {rnm}")
        print(line)
    print(f"""
    ★ 가장 낮은 칸: {best[1]}  ({best[0] * 100:.2f}%,  검사 {int(round(n * best[0])):,}셀)
      현행(트레이 전체 x 끝점) 대비 {need(Xw[:, -1] - neighbor_med(Xw[:, -1], g, R, C, rows, cols, 'tray')) / max(best[0], 1e-12):.1f}배 개선.

    읽는 법
      · 이웃이 가까울수록(인접 8셀) 교란이 비슷해 잘 상쇄되지만 중앙값이 불안정하다.
      · 이웃이 많을수록(트레이 전체) 중앙값은 안정되지만 교란이 덜 상쇄된다.
        그 사이 어딘가가 최적이고, 이 표가 그 지점을 찾아준다.
      · 시점 결합은 잡음을 줄인다. 창이 짧아 신호가 작을수록 이득이 크다.""")

    # ── [2] 시점별 ───────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 최적 조합을 시점별로 — 몇 분까지 줄일 수 있는가")
    print("-" * 78)
    bk = next(k for knm, k in KINDS if knm == best[1].split(" x ")[0])
    brn = best[1].split(" x ")[1]
    print(f"    조합: {best[1]}")
    print(f"    {'구간':>6}{'최악셀 검사율':>14}{'검사 셀':>10}{'과검':>9}{'상위1%':>8}{'상위5%':>8}")
    print("    " + "-" * 55)
    ngg = int(y.sum())
    for j, m in enumerate(mins):
        if m not in (3, 5, 8, 10, 12, 15, 20, 30) or m > mins[-1]: continue
        Xm = D[icols[:j + 1]].values.astype(float)
        tm = np.asarray(mins[:j + 1], float)
        if brn == "끝점": val = Xm[:, -1]
        elif brn == "전 시점 평균": val = Xm.mean(1)
        elif brn == "기울기":
            if j < 2: continue
            Lm = tm - tm.mean(); val = (Xm - Xm.mean(1, keepdims=True)) @ Lm / (Lm @ Lm)
        else:
            if j < 2: continue
            Qm = np.concatenate([np.zeros((n, 1)),
                                 np.cumsum((Xm[:, 1:] + Xm[:, :-1]) / 2 * np.diff(tm * 60), 1)], 1)
            val = Qm[:, -1]
        sc = val - neighbor_med(val, g, R, C, rows, cols, bk)
        v = need(sc); kk = int(round(n * v)); t_ = topk_recall(sc, y)
        print(f"    {m:>4}분{v * 100:>13.2f}%{kk:>9,}셀{kk - ngg:>8,}셀"
              f"{t_[0.01] * 100:>7.0f}%{t_[0.05] * 100:>7.0f}%")
    print("""
    → '검사 셀' 이 현행 공정의 적출 셀 수와 견줄 만해지는 가장 짧은 구간이
      실제로 쓸 수 있는 측정시간이다.""")

    print("\n" + "=" * 78)
    print(""" 이 표로 답할 수 없는 것 — 분석 밖의 지렛대
   ① 출력저항 Rout 를 낮춘다 (Keysight 논문)
      tau = (Rout + Rser) x Ceff. 0.5 -> 0.2 옴이면 5분 신호가 정확히 2배가 되어
      10분과 같아진다. 물리적으로 가장 근본적인 해법이다.
      단 Rout 를 낮추면 전압 잡음이 1/Rout 로 실리는데, 그 잡음은 같은 트레이
      셀들이 공유하므로 이웃 중앙값 빼기로 상당 부분 상쇄된다. 궁합이 좋다.
      ★ 현재 설정값 확인이 먼저다.
   ② 1 Hz 원시 샘플 확보
      지금은 1분 평균이라 5분 창에 5점뿐이다. 1 Hz 면 300점이고,
      평균화만으로 잡음이 약 7.7배 준다. 추가 계측도 공정 변경도 없다.
      장비가 이미 재고 있는 것을 안 받아온 것뿐이다.
   ③ 투입 전 열평형 대기 표준화
      교란의 원인이 'dT/dt' 이므로 투입 전에 식히면 5분 창의 교란이 준다.
      단 대기시간이 리드타임에 더해지면 의미가 없다. 다른 공정과 병렬일 때만.""")
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
        parse_target(sys.argv)
        with runlog.saving("short", a[0], sys.argv, sv, en):
            main(a[0], at, rw, cl, od)
