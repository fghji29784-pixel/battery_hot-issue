# -*- coding: utf-8 -*-
"""
적응형 측정시간 — 트레이마다 '순위가 굳는 시점'에서 멈춘다.

  python adaptive.py "데이터.xlsx"
  python adaptive.py "데이터.xlsx" --theta=0.98 --tmin=5
  python adaptive.py "데이터.xlsx" --band=0.10     # 2단 선별의 재검 대역폭
  python adaptive.py "데이터.xlsx" --floor=0.95    # 조기종료 허용 하한 (최악 트레이 기준)

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).
────────────────────────────────────────────────────────────────────────
 문제 설정
────────────────────────────────────────────────────────────────────────
 지금은 모든 트레이를 똑같이 15분(또는 30분) 측정한다.
 그런데 트레이마다 상태가 다르다. 온도가 이미 안정된 트레이는 5분이면 순위가
 굳고, 막 투입돼 식고 있는 트레이는 20분이 지나도 순위가 흔들린다.
 똑같이 주는 것은 앞쪽 트레이에는 낭비고 뒤쪽 트레이에는 부족하다.

 판정이 트레이 상대평가이므로, 필요한 것은 전류의 '값' 이 아니라
 '트레이 안에서의 순위' 다. 그러면 멈춰야 할 시점의 정의가 분명해진다.

     순위가 더 이상 바뀌지 않으면 멈춘다.

 이 규칙은 정답(3일 ΔOCV)을 몰라도 측정 중에 판정할 수 있다.
 트레이 안에서 t분 순위와 t-2분 순위의 상관만 보면 되기 때문이다.

────────────────────────────────────────────────────────────────────────
 두 가지 절감 경로
────────────────────────────────────────────────────────────────────────
 ① 트레이 단위 조기 종료 — 트레이 전체를 일찍 뺀다. 설비 점유시간이 줄어든다.
 ② 2단 선별 — 1단 짧게 전수, 애매한 대역만 2단으로 연장.
    채널을 재배치할 수 있을 때만 유효하다. 절감 단위는 '셀-분'.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import runlog
from predict_xlsx import load, I_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, measured_upto

GRADE_KEY = "판정등급"


def rank_in_tray(v, tray):
    return pd.Series(v).groupby(pd.Series(tray).values).rank().values


def main(spec, theta=0.98, tmin=5, band=0.10, floor=0.90):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins  = [int(I_PAT.match(c).group(1)) for c in icols]
    tgts  = [c for c in df.columns if TARGET_PAT.search(c)]
    tray  = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    gcol  = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    conds = [c for c in df.columns if COND_PAT.match(c) and pd.api.types.is_numeric_dtype(df[c])]
    if len(icols) < 4:
        print("  !! 시점별 전류 컬럼이 4개 미만이라 스윕할 수 없습니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    for c in icols + conds: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    if len(D) < 30:
        print(f"\n  !! {mins[-1]}분까지 측정된 셀이 {len(D)}개뿐입니다. 분석을 건너뜁니다.")
        print("     모든 시점의 전류가 같으면 '측정 조기 종료' 로 판정되어 전부 걸러집니다.")
        print("     입력 엑셀의 i_XXmin 컬럼이 시점마다 다른 값인지 확인하십시오.")
        return
    n = len(D); g = D["_tray"].values
    trays = list(pd.unique(g))
    if not [t for t in trays if (g == t).sum() >= 10]:
        print(f"\n  !! 10셀 이상인 트레이가 없습니다 (트레이 {len(trays)}개).")
        print("     트레이 안에서 순위를 매기는 분석이므로 트레이당 셀 수가 필요합니다.")
        return
    y = pd.to_numeric(D[tgts[-1]], errors="coerce").values if tgts else None
    ng = D[gcol].astype(str).str.strip().str.upper().isin(["E"]).values.astype(int) \
        if gcol else np.zeros(n, int)
    R = {m: rank_in_tray(D[c].values, g) for c, m in zip(icols, mins)}
    FLOOR = floor
    last = mins[-1]

    print("=" * 78); print(" 적응형 측정시간 — 트레이별로 멈추는 시점을 정한다"); print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {len(trays)}개 / 시점 {mins[0]}~{last}분 {len(mins)}개")
    print(f"  정지 기준 theta={theta}  최소 측정 {tmin}분"
          + (f"   불량(E) {int(ng.sum())}개" if ng.sum() else ""))

    # ── [1] 순위 안정화 곡선 ──────────────────────────────────────
    print("\n" + "-" * 78)
    print(f" [1] 순위 안정화 — t분 순위가 {last}분 순위를 얼마나 재현하는가 (트레이 안에서)")
    print("-" * 78)
    print(f"    {'구간':>6}{'중앙 rho':>10}{'하위10% 트레이':>15}{'rho>=0.98 트레이':>17}")
    print("    " + "-" * 48)
    per_m = {}
    for m in mins:
        rs = []
        for t in trays:
            i = np.where(g == t)[0]
            if len(i) < 10: continue
            r = spearmanr(R[m][i], R[last][i]).statistic
            if np.isfinite(r): rs.append(r)
        per_m[m] = np.array(rs)
        if not rs: continue
        if m in (mins[0], 8, 10, 12, 15, 20, 25, last):
            print(f"    {m:>4}분{np.median(rs):>10.3f}{np.percentile(rs, 10):>15.3f}"
                  f"{np.mean(np.array(rs) >= 0.98) * 100:>16.0f}%")
    print("""
    → 중앙값만 보면 일찍 포화해 보인다. 중요한 건 '하위 10% 트레이' 다.
      전수 검사에서 실패는 평균이 아니라 최악의 트레이에서 난다.
      고정 시간으로 정하려면 하위 10% 가 기준을 넘는 시점이어야 한다.""")

    # ── [2] 트레이별 조기 종료 ────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [2] 트레이별 조기 종료 — 무엇이 '안 바뀌면' 멈출 것인가")
    print("-" * 78)
    qtop = 0.05

    def top_set(m, i):
        k = max(int(round(len(i) * qtop)), 3)
        return set(i[np.argsort(-R[m][i])[:k]])

    def stop_times(crit, th):
        """crit='rank' 전체 순위 상관, crit='top' 적출 후보 집합의 일치도.

        둘 다 정답(3일 ΔOCV)을 쓰지 않는다. 측정 중에 계산할 수 있다.
        """
        st = {}
        for t in trays:
            i = np.where(g == t)[0]; s_ = last
            if len(i) >= 10:
                hit = 0
                for j, m in enumerate(mins):
                    if m < tmin or j < 2: continue
                    if crit == "rank":
                        v = spearmanr(R[m][i], R[mins[j - 2]][i]).statistic
                    else:
                        a, b = top_set(m, i), top_set(mins[j - 2], i)
                        v = len(a & b) / max(len(a | b), 1)
                    hit = hit + 1 if (np.isfinite(v) and v >= th) else 0
                    if hit >= 2: s_ = m; break
            st[t] = s_
        return st

    cells = D.groupby("_tray").size().reindex(trays).values.astype(float)

    def summarize(st):
        sv = np.array([st[t] for t in trays], float)
        mt = float(np.average(sv, weights=cells))
        fd = []
        for t in trays:
            i = np.where(g == t)[0]
            if len(i) >= 10:
                fd.append(spearmanr(R[st[t]][i], R[last][i]).statistic)
        return sv, mt, np.array(fd, float)

    print(f"    {'기준':<10}{'theta':>7}{'평균 측정시간':>13}{'최종순위 재현(중앙/최저)':>24}"
          + (f"{'상위5% E':>10}" if ng.sum() else ""))
    print("    " + "-" * (54 + (10 if ng.sum() else 0)))
    from correct import within_tray_rho, topk_recall
    cands = []
    for crit, nm in (("rank", "전체순위"), ("top", "적출후보")):
        for th in (0.90, 0.95, 0.98, 0.99):
            st = stop_times(crit, th)
            sv, mt, fd = summarize(st)
            adp = np.concatenate([[R[st[t]][k] for k in np.where(g == t)[0]] for t in trays])
            ordr = np.concatenate([np.where(g == t)[0] for t in trays])
            sc = np.empty(n); sc[ordr] = adp
            sc = sc - pd.Series(sc).groupby(g).transform("mean").values
            line = f"    {nm:<10}{th:>7.2f}{mt:>12.1f}분{np.nanmedian(fd):>14.3f}{np.nanmin(fd):>10.3f}"
            if ng.sum(): line += f"{topk_recall(sc, ng)[0.05] * 100:>9.0f}%"
            print(line)
            cands.append((nm, th, st, sv, mt, fd, sc))
    print("""
    → theta 를 미리 정하지 말고 이 표에서 고른다.
      '최종순위 재현 최저' 가 무너지지 않는 선에서 평균 측정시간이 가장 짧은 칸이다.
      '적출후보' 기준은 중간 순위의 잡음에 둔감하다. 판정은 상위만 쓰기 때문이다.""")

    # 최종순위 재현의 '최저 트레이' 가 floor 위에 있는 칸 중 가장 짧은 것을 고른다.
    ok = [c for c in cands if np.isfinite(np.nanmin(c[5])) and np.nanmin(c[5]) >= FLOOR]
    pick = min(ok, key=lambda c: c[4]) if ok else max(cands, key=lambda c: np.nanmin(c[5]))
    nm_p, th_p, st, sv, mean_t, fd, sc_adp = pick
    print(f"\n    [권고 설정] {nm_p} 기준 theta={th_p:.2f}"
          f"   (최저 재현 {np.nanmin(fd):.3f} >= 기준 {FLOOR:.2f})"
          if ok else f"\n    [권고 설정] 기준 {FLOOR:.2f} 를 만족하는 칸이 없습니다."
                     f" 가장 안전한 칸: {nm_p} theta={th_p:.2f}")
    print(f"    정지 시점 분포   중앙 {np.median(sv):.0f}분"
          f"   5~95% {np.percentile(sv, 5):.0f}~{np.percentile(sv, 95):.0f}분"
          f"   최대 {sv.max():.0f}분")
    for m in sorted(set(sv)):
        k = int((sv == m).sum())
        print(f"      {m:>4.0f}분에 정지: {k:>3}트레이 ({k / len(trays) * 100:>4.0f}%)")
    print(f"\n    셀 가중 평균 측정시간 {mean_t:.1f}분")
    for ref in (15, last):
        if ref > mean_t:
            print(f"      고정 {ref}분 대비 {(1 - mean_t / ref) * 100:>4.1f}% 단축"
                  f"  (설비 점유 {ref / max(mean_t, 1e-9):.2f}배 처리량)")

    if y is not None or ng.sum():
        print("\n    고정 시간과 같은 기준으로 비교")
        print(f"    {'방식':<22}{'평균 측정시간':>13}"
              + (f"{'트레이내 rho':>13}" if y is not None else "")
              + (f"{'상위1% E':>10}{'상위5% E':>10}" if ng.sum() else ""))
        print("    " + "-" * (35 + (13 if y is not None else 0) + (20 if ng.sum() else 0)))
        opts = [(f"고정 {m}분", float(m),
                 R[m] - pd.Series(R[m]).groupby(g).transform("mean").values)
                for m in (mins[0], 10, 15, last) if m in R]
        opts.append((f"적응형 ({nm_p} {th_p:.2f})", mean_t, sc_adp))
        for nm, tt, s_ in opts:
            line = f"    {nm:<22}{tt:>12.1f}분"
            if y is not None: line += f"{within_tray_rho(s_, y, g)[0]:>13.3f}"
            if ng.sum():
                r = topk_recall(s_, ng); line += f"{r[0.01] * 100:>9.0f}%{r[0.05] * 100:>9.0f}%"
            print(line)
    stop = st
    fid = {t: v for t, v in zip([t for t in trays if (g == t).sum() >= 10], fd)}
    fid = {t: fid.get(t, np.nan) for t in trays}

    # ── [3] 2단 선별 ──────────────────────────────────────────────
    print("\n" + "-" * 78)
    print(f" [3] 2단 선별 — 1단 {tmin}분 전수, 상·하위가 분명한 셀은 그때 내보낸다")
    print("-" * 78)
    print(f"    {'1단':>5}{'2단 연장 대상':>13}{'평균 셀-분':>12}{'고정15분 대비':>14}"
          + (f"{'E 유지':>9}" if ng.sum() else ""))
    print("    " + "-" * (44 + (9 if ng.sum() else 0)))
    for t1 in [m for m in (5, 8, 10, 12) if m in R]:
        z1 = R[t1] - pd.Series(R[t1]).groupby(g).transform("mean").values
        nz = pd.Series(z1).groupby(g).transform(
            lambda v: v / v.std() if v.std() > 0 else v * 0.0).values
        nz = np.nan_to_num(nz, nan=0.0, posinf=0.0, neginf=0.0)
        thr = np.nanquantile(nz, 1 - band)
        ext = nz >= thr
        cm = t1 + (last - t1) * ext.mean()
        keep = (ng[ext] == 1).sum() / max(ng.sum(), 1)
        line = f"    {t1:>4}분{ext.mean() * 100:>12.1f}%{cm:>11.1f}분{(1 - cm / 15) * 100:>13.1f}%"
        if ng.sum(): line += f"{keep * 100:>8.0f}%"
        print(line)
    print(f"""
    → 1단에서 상위 {band * 100:.0f}% 만 2단으로 연장한다. 나머지는 그 시점에 확정.
      'E 유지' 는 진짜 불량이 2단 대상에 남아 있는 비율이다. 100%가 아니면
      1단을 늘리거나 대역을 넓혀야 한다 (--band).
      ★ 이 방식은 채널을 재배치할 수 있을 때만 시간이 실제로 준다.
        재배치가 안 되면 [2] 의 트레이 단위 조기 종료를 쓸 것.""")

    # ── [4] 측정 건전성 ───────────────────────────────────────────
    print("\n" + "-" * 78)
    print(" [4] 측정 건전성 — 그 트레이의 결과를 믿어도 되는가")
    print("-" * 78)
    H = pd.DataFrame(index=trays)
    H["음수비율"] = D.groupby("_tray")[icols[-1]].apply(lambda v: float(np.mean(v < 0))).reindex(trays)
    if "t_init" in D.columns and "t_final" in D.columns:
        H["dTdt"] = (D.groupby("_tray")["t_final"].median()
                     - D.groupby("_tray")["t_init"].median()).reindex(trays) / float(last)
    H["순위안정"] = [fid[t] for t in trays]
    H["정지시점"] = [stop[t] for t in trays]
    bad = pd.Series(False, index=trays)
    if "dTdt" in H: bad |= H["dTdt"].abs() > H["dTdt"].abs().quantile(0.90)
    bad |= H["순위안정"] < 0.95
    bad |= H["음수비율"] > 0.5
    print(f"    {'지표':<12}{'중앙':>10}{'90분위':>10}{'최악':>10}")
    print("    " + "-" * 42)
    for c in H.columns:
        print(f"    {c:<12}{H[c].median():>10.3f}{H[c].quantile(0.9):>10.3f}{H[c].max():>10.3f}")
    print(f"\n    재측정·연장 권고 트레이: {int(bad.sum())}개 / {len(trays)}개"
          f"  ({bad.mean() * 100:.0f}%)")
    print("""
    판정 규칙 (제안)
      · |dT/dt| 상위 10%        → 대기 후 재측정. 열평형 전에 넣은 트레이다.
      · 순위 안정도 < 0.95      → 측정 연장. 아직 순위가 흔들린다.
      · 음수 비율 > 50%         → 열드리프트가 신호를 덮었다. 결과를 쓰지 말 것.
    → 지금은 모든 트레이의 결과를 같은 신뢰도로 쓴다. 트레이마다 신뢰도가
      다르다는 것을 수치로 말할 수 있으면, 그것만으로도 오판이 줄어든다.""")

    print("\n" + "=" * 78)
    print(""" 이 스크립트가 주는 새 결과
   · '몇 분이 필요한가' 의 답이 트레이마다 다르다는 것을 수치로 보인다.
   · 정답 없이, 측정 중에 멈출 시점을 정하는 규칙을 준다.
   · 같은 검출 성능에서 평균 측정시간이 얼마나 주는지 계산한다.
   · 믿으면 안 되는 트레이를 골라낸다.""")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        th, tm, bd, fl = 0.98, 5, 0.10, 0.90
        for x in sys.argv:
            if x.startswith("--theta="): th = float(x.split("=")[1])
            if x.startswith("--tmin="):  tm = int(x.split("=")[1])
            if x.startswith("--band="):  bd = float(x.split("=")[1])
            if x.startswith("--floor="): fl = float(x.split("=")[1])
        sv, en = runlog.parse(sys.argv)
        with runlog.saving("adaptive", a[0], sys.argv, sv, en):
            main(a[0], th, tm, bd, fl)
