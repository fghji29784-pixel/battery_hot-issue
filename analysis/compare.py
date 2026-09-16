# -*- coding: utf-8 -*-
"""보정 방식 정면 비교 — 조건·위치 보정 vs 행 프로파일, 5분 vs 15분

  python compare.py "데이터.xlsx"
  python compare.py "데이터.xlsx" --at=5,10,15      # 볼 시점

결과는 analysis/results/ 에 자동 저장된다 (--save=경로 / --no-save).

  --target="컬럼명"   3일 ΔOCV 컬럼을 직접 지정 (DOCV 처럼 표기가 다를 때)
  --drop-tray=A,B     특정 트레이 제외
  --drop-empty-target ΔOCV 가 20개 미만인 트레이 자동 제외 (=N 으로 기준 변경)
  --grid=12x12        격자를 직접 지정

────────────────────────────────────────────────────────────────────────
 왜 필요한가
────────────────────────────────────────────────────────────────────────
 '15분에서 전류·전압·온도·위치를 보정하고 트레이별 상대평가로 과검 없이
 선별했다' 는 결과가 따로 있다. 그렇다면 5분에서도 같은 보정이 통하는가?

 두 가지를 갈라야 답이 나온다.

 (1) 무엇을 맞혔다는 것인가
     · 3일 ΔOCV 와의 일치   → 상관(rho) 문제
     · 판정등급 E 의 검출    → 선별 문제
     이 둘은 같이 움직이지 않는다. 이 저장소의 기존 결과에서는
     조건 변수를 넣자 rho 가 0.005 → 0.723 (145배) 올라가는 동안
     불량 검출은 50%/62% → 38%/50% 로 떨어졌다.
     그러므로 '과검 없이' 가 어느 잣대인지 모르면 비교가 성립하지 않는다.
     → 이 스크립트는 두 잣대를 나란히 찍는다.

 (2) 그 성능이 같은 데이터에서 맞춘 것인가
     조건 4종 + 위치까지 넣어 계수를 맞추면 변수는 수십 개가 되는데
     불량은 4~8개뿐이다. 맞춘 데이터에서 그대로 재면 과검 0 은 어렵지 않고,
     새 로트에서는 재현되지 않는다.
     → 이 스크립트는 같은 보정을 in-sample 과 out-of-fold(트레이 단위
       GroupKFold) 두 번 재서 나란히 찍는다. 둘의 차이가 곧 과적합량이다.

 ※ 트레이별로 계수를 맞추는 보정은 원리상 out-of-fold 검증이 불가능하다.
   그 트레이를 빼면 그 트레이의 계수를 만들 수 없기 때문이다.
   그런 보정은 '검증된 성능' 을 말할 수 없다는 점을 같이 적는다.

────────────────────────────────────────────────────────────────────────
 5분이 15분보다 불리한 이유 (물리)
────────────────────────────────────────────────────────────────────────
   신호는 작고   tau = 7.45시간. 5분은 tau 의 1.1%, 15분은 3.3% 구간이라
                 자가방전 신호 자체가 약 3배 작다.
   교란은 크다   열드리프트는 dT/dt 에 비례하는데 투입 직후가 가장 가파르다.
                 5분 창은 교란의 꼭대기에 걸리고, 15분이면 일부 감쇠한다.
   → 5분은 신호·잡음 양쪽에서 불리하다. 보정이 지울 것은 많고 남길 것은 적다.
     다만 교란이 클수록 더 체계적이기도 하므로, 보정이 유리해질 여지도 있다.
     어느 쪽이 이기는지는 재 봐야 안다. 그것이 이 스크립트다.
"""
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import runlog
from predict_xlsx import (load, I_PAT, COND_PAT, TARGET_PAT, TRAY_PAT, CELL_PAT,
                          measured_upto, find_targets, parse_target, drop_trays)
from correct import derive_conds, target_report, within_tray_rho, safe_z
from rescue import grid_pos
from judge import group_z, worst_rank

GRADE_KEY = "판정등급"
TMIN_PAT = re.compile(r"^t[_\s]*(\d+)\s*min$", re.I)


def resid(Xf, ycur, fit_idx=None, pred_idx=None):
    """ycur 를 Xf 로 회귀하고 잔차를 돌려준다. 절편 포함, 최소제곱.

    fit_idx 로 계수를 맞추고 pred_idx 에서 잔차를 낸다.
    둘 다 None 이면 전체로 맞추고 전체에서 잔차를 낸다(in-sample).
    """
    n = len(ycur)
    fit_idx = np.arange(n) if fit_idx is None else fit_idx
    pred_idx = np.arange(n) if pred_idx is None else pred_idx
    A = np.c_[np.ones(len(Xf)), Xf]
    Af, yf = A[fit_idx], ycur[fit_idx]
    ok = np.isfinite(Af).all(1) & np.isfinite(yf)
    if ok.sum() < Af.shape[1] + 2:
        return ycur[pred_idx] - np.nanmedian(ycur[fit_idx])
    try:
        beta, *_ = np.linalg.lstsq(Af[ok], yf[ok], rcond=None)
    except np.linalg.LinAlgError:
        return ycur[pred_idx] - np.nanmedian(ycur[fit_idx])
    pred = A[pred_idx] @ beta
    return ycur[pred_idx] - pred


def oof_resid(Xf, ycur, tray):
    """트레이 단위로 하나씩 빼면서 잔차를 낸다 (GroupKFold, leave-one-tray-out).

    보정 계수를 그 셀이 속하지 않은 트레이들로만 맞춘다.
    새 로트에 적용했을 때의 성능에 해당한다.
    """
    out = np.full(len(ycur), np.nan)
    tv = pd.Series(tray).values
    for t in pd.unique(tv):
        te = np.where(tv == t)[0]
        tr = np.where(tv != t)[0]
        if len(tr) < Xf.shape[1] + 5: out[te] = ycur[te]; continue
        out[te] = resid(Xf, ycur, tr, te)
    return out


def tray_center(v, g):
    """트레이 중앙값을 빼서 트레이 상대값으로 만든다."""
    return v - pd.Series(v).groupby(pd.Series(g)).transform("median").values


def row_profile(v, g, r):
    """같은 트레이 같은 행의 중앙값을 뺀다 (이 저장소의 ②'')."""
    key = [pd.Series(g), pd.Series(r)]
    prof = pd.Series(v).groupby(key).transform("median").values
    cnt = pd.Series(v).groupby(key).transform("count").values
    return np.where(cnt >= 5, v - np.nan_to_num(prof, nan=0.0), v)


def report(s, y, yv, g, n):
    """한 점수를 두 잣대로 잰다. (최악셀 적출수, 과검, rho, μ+3σ 검출)"""
    s = np.nan_to_num(np.asarray(s, float), nan=-1e18, posinf=-1e18, neginf=-1e18)
    if np.nanstd(s) <= 0: return None
    cnt = worst_rank(s, y)
    over = cnt - int(y.sum())
    rho = within_tray_rho(s, yv, g)[0] if yv is not None else float("nan")
    z = group_z(s, g, "sigma")
    f3 = np.isfinite(z) & (z >= 3.0)
    return cnt, over, rho, int(f3[y == 1].sum()), int(f3.sum())


def main(spec, ats=None, rows=None, cols=None, order="col"):
    df, _ = load(spec)
    icols = sorted([c for c in df.columns if I_PAT.match(c)], key=lambda c: int(I_PAT.match(c).group(1)))
    mins = [int(I_PAT.match(c).group(1)) for c in icols]
    tgts = find_targets(df)
    tray = next((c for c in df.columns if TRAY_PAT.search(c)), None)
    cell = next((c for c in df.columns if CELL_PAT.match(str(c).strip())), None)
    gcol = next((c for c in df.columns if GRADE_KEY in str(c)), None)
    if len(icols) < 2 or not gcol:
        print("  !! 전류 컬럼(i_XXmin)과 판정등급이 필요합니다."); return

    D = df.copy()
    D["_tray"] = D[tray].astype(str) if tray else "ALL"
    D = drop_trays(D)
    D["_num"] = pd.to_numeric(D[cell], errors="coerce") if cell else np.nan
    for c in icols: D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=icols).reset_index(drop=True)
    D["_upto"] = measured_upto(D[icols].values, mins)
    D = D[D["_upto"] >= mins[-1]].reset_index(drop=True)
    made = derive_conds(D, "t") + derive_conds(D, "v")
    if made: print(f"  시계열에서 만든 조건 변수 {len(made)}개: {', '.join(made)}")
    n = len(D)
    if n < 40: print(f"\n  !! 셀이 {n}개뿐입니다."); return
    y = D[gcol].astype(str).str.strip().str.upper().eq("E").values.astype(int)
    if y.sum() == 0: print("\n  !! 불량(E) 이 없습니다."); return
    g = D["_tray"].values
    yv = pd.to_numeric(D[tgts[-1]], errors="coerce").values if tgts else None
    if tgts: target_report(yv, g, str(tgts[-1]))

    # 자리
    have_pos = D["_num"].notna().any()
    if have_pos:
        mx = int(np.nanmax(D.groupby("_tray")["_num"].max().values))
        rw, cl = rows, cols
        if rw is None or cl is None:
            cand = [(abs(r - mx / r), r, mx // r) for r in range(1, mx + 1) if mx % r == 0]
            _, rw, cl = min(cand) if cand else (0, 1, mx)
        D["_r"], D["_c"] = grid_pos(D["_num"].values, rw, cl, order)
    else:
        rw = cl = 0; D["_r"] = D["_c"] = 0

    # 온도 시계열이 있으면 시점별 dT/dt 를 쓴다
    tmins = sorted([c for c in D.columns if TMIN_PAT.match(c)], key=lambda c: int(TMIN_PAT.match(c).group(1)))
    X = D[icols].values.astype(float)

    print("=" * 78)
    print(" 보정 방식 정면 비교 — 조건·위치 보정 vs 행 프로파일")
    print("=" * 78)
    print(f"  {n:,}셀 / 트레이 {D['_tray'].nunique()}개 / 격자 {rw} x {cl} / 불량(E) {int(y.sum())}개")
    print(f"  ※ '적출' 은 불량을 전부 잡는 데 필요한 셀 수, '과검' 은 그중 양품이다.")
    print(f"    과검 0 은 불량 {int(y.sum())}개가 정확히 1~{int(y.sum())}위일 때만 나온다.")

    ats = ats or [a for a in (5, 10, 15) if a <= mins[-1]] or [mins[-1]]
    for at in ats:
        ai = int(np.argmin(np.abs(np.array(mins) - at)))
        cur = X[:, ai]

        # 조건 변수 — 다른 분 방식이 쓰는 것들
        feats, names = [], []
        if len(tmins) >= 3:
            tm = np.array([int(TMIN_PAT.match(c).group(1)) for c in tmins], float)
            keep = tm <= mins[ai] + 1e-9
            if keep.sum() >= 3:
                TT = D[[c for c, k in zip(tmins, keep) if k]].values.astype(float)
                tc = tm[keep] - tm[keep].mean()
                feats += [(TT - TT.mean(1, keepdims=True)) @ tc / np.sum(tc ** 2), TT.mean(1)]
                names += ["dT/dt", "T_mean"]
        if not names:
            for c in ("delta_t", "t_init"):
                if c in D.columns: feats.append(pd.to_numeric(D[c], errors="coerce").values); names.append(c)
        for c in ("delta_v", "v_init"):
            if c in D.columns: feats.append(pd.to_numeric(D[c], errors="coerce").values); names.append(c)
        if not feats:
            print(f"\n  !! {at}분: 조건 변수가 없어 보정 비교를 못 합니다."); continue
        Xc = np.column_stack(feats)
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        # 위치는 더미로 (행·열 각각)
        if have_pos and rw > 2:
            dr = pd.get_dummies(pd.Series(D["_r"].values), drop_first=True).values.astype(float)
            dc = pd.get_dummies(pd.Series(D["_c"].values), drop_first=True).values.astype(float)
            Xcp = np.c_[Xc, dr, dc]
        else:
            Xcp = Xc

        rp = D["_r"].values
        # 0~at 창의 기울기 (이 저장소의 5분 방식)
        tw = np.asarray(mins[:ai + 1], float)
        sl = (X[:, :ai + 1] - X[:, :ai + 1].mean(1, keepdims=True)) @ (tw - tw.mean()) / np.sum((tw - tw.mean()) ** 2) \
            if ai >= 2 else np.full(n, np.nan)

        cands = [
            ("원시 전류 끝값", cur, ""),
            ("트레이 정규화", tray_center(cur, g), ""),
            (f"조건보정({len(names)}종) → 트레이", tray_center(resid(Xc, cur), g), "fit"),
            ("조건+위치 보정 → 트레이", tray_center(resid(Xcp, cur), g), "fit"),
            ("행 프로파일 (②'')", row_profile(cur, g, rp), ""),
        ]
        if ai >= 2:
            cands.append(("행 프로파일 × 기울기", row_profile(sl, g, rp), ""))

        print("\n" + "-" * 78)
        print(f" [{at}분] 같은 데이터, 같은 잣대")
        print("-" * 78)
        print(f"    쓴 조건 변수: {', '.join(names)}"
              + (f" + 행·열 더미" if Xcp.shape[1] > Xc.shape[1] else ""))
        print(f"    {'점수':<28}{'적출':>8}{'과검':>8}{'수율손실':>10}{'ΔOCV rho':>10}{'μ+3σ 검출':>12}")
        print("    " + "-" * 76)
        for nm, s, kind in cands:
            r = report(s, y, yv, g, n)
            if r is None:
                print(f"    {nm:<28}   분산이 0 이라 건너뜀"); continue
            cnt, over, rho, hit3, flag3 = r
            print(f"    {nm:<28}{cnt:>7}셀{over:>7}셀{over / n * 100:>9.2f}%"
                  f"{rho:>10.3f}{f'{hit3}/{int(y.sum())}':>11}")

        # ── 과적합 검사 ─────────────────────────────────────────
        print(f"\n    ── 맞춘 데이터에서 잰 것인가 (트레이 단위 leave-one-out) ──")
        print(f"    {'점수':<28}{'in-sample':>12}{'out-of-fold':>14}{'차이':>9}")
        print("    " + "-" * 66)
        for nm, Xuse in [("조건보정 → 트레이", Xc), ("조건+위치 보정 → 트레이", Xcp)]:
            a_in = report(tray_center(resid(Xuse, cur), g), y, yv, g, n)
            a_oof = report(tray_center(oof_resid(Xuse, cur, g), g), y, yv, g, n)
            if not a_in or not a_oof: continue
            d = a_oof[0] - a_in[0]
            print(f"    {nm:<28}{a_in[0]:>11}셀{a_oof[0]:>13}셀{d:>+8}셀")
        print(f"""    → 두 값이 크게 벌어지면 그 성능은 맞춘 데이터 안에서만 나온 것이다.
      변수는 {Xcp.shape[1]}개인데 불량은 {int(y.sum())}개뿐이라 벌어지기 쉽다.
      ※ 트레이별로 계수를 맞추는 보정은 이 검증 자체가 불가능하다.
        그 트레이를 빼면 그 트레이의 계수를 만들 수 없기 때문이다.""")

    print("\n" + "=" * 78)
    print(" 읽는 법")
    print("   · 'ΔOCV rho' 와 '적출' 이 같이 좋아지지 않으면, 두 방식은 서로 다른")
    print("     것을 맞히고 있는 것이다. 어느 쪽이 목적인지 먼저 합의해야 한다.")
    print("   · 'μ+3σ 검출' 은 현행 공정과 같은 판정 규칙을 이 점수에 적용한 결과다.")
    print("     여기서 전량 검출이 되면 '트레이별 상대평가로 잡힌다' 가 성립한다.")
    print("   · in-sample 과 out-of-fold 가 벌어지면 새 로트에서 재현되지 않는다.")
    print("     보정 변수를 늘릴수록 벌어진다. 이것이 '과검 0' 의 흔한 정체다.")
    print("=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a: print(__doc__)
    else:
        ats, rw, cl, od = None, None, None, "col"
        for x in sys.argv:
            if x.startswith("--at="):
                ats = [int(t) for t in re.findall(r"\d+", x.split("=", 1)[1])]
            if x.startswith("--order="): od = x.split("=")[1].strip().lower()[:3]
            if x.startswith("--grid="):
                m = re.match(r"(\d+)\s*[xX*]\s*(\d+)", x.split("=")[1])
                if m: rw, cl = int(m.group(1)), int(m.group(2))
        sv, en = runlog.parse(sys.argv)
        parse_target(sys.argv)
        with runlog.saving("compare", a[0], sys.argv, sv, en):
            main(a[0], ats, rw, cl, od)
