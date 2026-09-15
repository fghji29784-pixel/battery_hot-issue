# -*- coding: utf-8 -*-
"""
SDM 데이터 진단 — 사내에서 실행. 원본/셀단위 값은 출력되지 않고 네트워크도 쓰지 않음.

사용법
  1) 먼저 파일 구조부터 확인          python diagnose.py --inspect "파일.CSV"
  2) 구조가 맞으면 진단 실행           python diagnose.py "파일.CSV"
  3) 3일 ΔOCV 파일이 있으면 (선택)     python diagnose.py "파일.CSV" "타깃.CSV"

  ※ 3일 ΔOCV가 없어도 됩니다. 핵심 결과인 [B] 순위보존 표는 곡선만으로 나옵니다.

입력 형식 (자동 판별)
  wide : 한 컬럼이 시간, 나머지 각 컬럼이 셀별 전류
  long : 셀ID / 시간 / 전류 컬럼이 따로 있는 형태
"""
import sys, io, os, re
import numpy as np, pandas as pd

ENCODINGS  = ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin-1"]
TIME_PAT   = re.compile(r"(time|시간|sec|second|t\[|^t$|elapsed)", re.I)
CUR_PAT    = re.compile(r"(current|전류|isd|i_sd|ua|µa|μa|amp|i\[)", re.I)
CELL_PAT   = re.compile(r"(cell|셀|channel|ch|slot|serial|barcode|id)", re.I)
OCV_PAT    = re.compile(r"(ocv|docv|docv|delta|전압|drop|mv)", re.I)


def read_any(path):
    """인코딩·구분자·머리말 자동 판별해서 DataFrame으로."""
    raw = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                raw = f.read(); used = enc; break
        except (UnicodeDecodeError, LookupError):
            continue
    if raw is None:
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            raw = f.read(); used = "latin-1(대체)"

    lines = raw.splitlines()
    # 구분자 추정
    head = "\n".join(lines[:40])
    sep = max([",", ";", "\t", "|"], key=lambda s: head.count(s))
    # 헤더 행 찾기: 구분자 개수가 안정적으로 최대가 되는 첫 행
    counts = [l.count(sep) for l in lines[:60]]
    if not counts or max(counts) == 0:
        raise ValueError(f"구분자를 찾지 못했습니다. (추정 '{sep}')")
    target = max(counts)
    hdr = next(i for i, c in enumerate(counts) if c == target)

    df = pd.read_csv(io.StringIO(raw), sep=sep, skiprows=hdr, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df, used, sep, hdr


def inspect(path):
    print("="*74); print(f" 파일 구조 확인: {os.path.basename(path)}"); print("="*74)
    try:
        df, enc, sep, hdr = read_any(path)
    except Exception as e:
        print("  읽기 실패:", e)
        print("\n  파일 앞부분 원문 10줄:")
        with open(path, "rb") as f:
            for i, b in enumerate(f):
                if i >= 10: break
                print("   ", b[:160])
        return None
    print(f"  인코딩 {enc} / 구분자 '{sep}' / 헤더 {hdr}번째 줄 / 크기 {df.shape[0]:,}행 x {df.shape[1]}열")
    print(f"\n  컬럼 목록 (최대 40개):")
    for i, c in enumerate(df.columns[:40]):
        kind = df[c].dtype
        nn = df[c].notna().sum()
        tag = []
        if TIME_PAT.search(c): tag.append("시간?")
        if CUR_PAT.search(c):  tag.append("전류?")
        if CELL_PAT.search(c): tag.append("셀ID?")
        if OCV_PAT.search(c):  tag.append("전압?")
        print(f"    [{i:>2}] {c[:38]:<38} {str(kind):<10} 유효 {nn:>7,}  {' '.join(tag)}")
    if df.shape[1] > 40: print(f"    ... 외 {df.shape[1]-40}개")
    print(f"\n  첫 3행 (앞 6열만):")
    print(df.iloc[:3, :6].to_string(index=False))
    print("\n" + "="*74)
    print(" 이 목록만 알려주시면 형식을 맞춰 드립니다. (값이 아니라 컬럼 이름만)")
    print("="*74)
    return df


def to_matrix(df):
    """DataFrame → (t, I[n_cell, n_time], ids).  wide/long 자동 판별."""
    cols = list(df.columns)
    tcol = next((c for c in cols if TIME_PAT.search(c)), None)
    ccol = next((c for c in cols if CELL_PAT.search(c) and c != tcol), None)
    icol = next((c for c in cols if CUR_PAT.search(c) and c not in (tcol, ccol)), None)

    if tcol and ccol and icol and df[ccol].nunique() > 1:                 # long
        W = df.pivot_table(index=tcol, columns=ccol, values=icol, aggfunc="mean")
        return W.index.values.astype(float), W.values.T.astype(float), [str(c) for c in W.columns], "long"

    num = df.select_dtypes("number")
    if tcol is None:                                                      # 시간 컬럼 추정
        inc = [c for c in num.columns if num[c].is_monotonic_increasing and num[c].nunique() > 10]
        tcol = inc[0] if inc else num.columns[0]
    val = [c for c in num.columns if c != tcol]
    if len(val) < 2:
        raise ValueError("셀 컬럼을 2개 이상 찾지 못했습니다. --inspect 로 구조를 확인해 주세요.")
    return num[tcol].values.astype(float), num[val].values.T.astype(float), [str(c) for c in val], "wide"


def analyse(t, I, ids, fmt, targets=None):
    from scipy.stats import spearmanr, pearsonr
    from scipy.optimize import curve_fit
    order = np.argsort(t); t = t[order]; I = I[:, order]
    ok = np.isfinite(I).all(axis=1)
    if (~ok).any(): print(f"  (결측 있는 셀 {(~ok).sum()}개 제외)")
    I = I[ok]; ids = [c for c, k in zip(ids, ok) if k]
    t = t - t[0]; n, T = I.shape

    print("="*74); print(" SDM 데이터 진단"); print("="*74)
    print(f"  형식 {fmt} / 셀 {n}개 / 시점 {T:,}개 / 길이 {t[-1]/3600:.2f}시간"
          f" / 간격 {np.median(np.diff(t)):.1f}초")
    Ifin = I[:, -max(T//50, 3):].mean(axis=1)
    print(f"  최종 전류: 중앙값 {np.median(Ifin):.1f} µA,"
          f" 5~95% {np.percentile(Ifin,5):.0f}~{np.percentile(Ifin,95):.0f},"
          f" 최대 {Ifin.max():.0f} µA")

    print("\n" + "-"*74); print(" [A] 정착 시정수 (중앙값 곡선)"); print("-"*74)
    med = np.median(I, axis=0)
    f2 = lambda x, a, b, c, d, e: a - b*np.exp(-x/c) - d*np.exp(-x/e)
    try:
        p, _ = curve_fit(f2, t, med, p0=[med[-1]*1.15, med[-1]*.5, t[-1]/20, med[-1]*.5, t[-1]/3],
                         bounds=([0,0,1,0,1], [np.inf]*5), maxfev=100000)
        tf, ts = sorted([p[2], p[4]])
        print(f"  I∞ {p[0]:.1f} µA   τ_fast {tf/60:.1f}분   τ_slow {ts/3600:.2f}시간")
        print(f"  측정 종료 정착률 {(1-np.exp(-t[-1]/ts))*100:.1f}%"
              f"   /  15분 정착률 {(1-np.exp(-900/ts))*100:.1f}%")
    except Exception as e:
        print("  피팅 실패:", e)

    print("\n" + "-"*74)
    print(" [B] ★ 순위 보존 — 이 표가 핵심입니다")
    print("-"*74)
    print(f"  {'t':>8} {'Spearman ρ':>12} {'Pearson r':>11} {'상위10% 재현':>13}")
    print("  " + "-"*48)
    k = max(n//10, 1); topf = set(np.argsort(-Ifin)[:k])
    for ct in [60, 180, 300, 600, 900, 1200, 1800, 3600, 7200, 10800]:
        if ct > t[-1]: continue
        j = int(np.argmin(np.abs(t-ct))); v = I[:, j]
        rec = len(set(np.argsort(-v)[:k]) & topf)/k
        lab = f"{ct//60}분" if ct < 3600 else f"{ct//3600}시간"
        star = "   ← 15분" if ct == 900 else ""
        print(f"  {lab:>8} {spearmanr(v,Ifin).statistic:>12.3f} {pearsonr(v,Ifin)[0]:>11.3f}"
              f" {rec*100:>11.0f}%{star}")
    print("""
  판정:  ρ > 0.9   그 시점이면 최종 순위를 거의 재현. 충분
         ρ 0.7~0.9 쓸 만함. 다변량 피처로 보완하면 실용적
         ρ < 0.7   부족. 더 긴 측정 필요""")

    j9 = int(np.argmin(np.abs(t-900)))
    if j9 > 5:
        print("\n" + "-"*74); print(" [C] 15분 구간 피처별 최종값 상관"); print("-"*74)
        st, sg = t[:j9+1], I[:, :j9+1]; q = max(j9//3, 2)
        for nm, v in {
            "값 I(900s)":  sg[:, -1],
            "기울기(후반)": np.array([np.polyfit(st[-q:], s[-q:],1)[0]*3600 for s in sg]),
            "적분":        np.trapezoid(sg, st, axis=1)/st[-1],
            "곡률":        np.array([np.polyfit(st[:q], s[:q],1)[0]-np.polyfit(st[-q:], s[-q:],1)[0] for s in sg]),
        }.items():
            print(f"    {nm:<14} ρ = {spearmanr(v, Ifin).statistic:+.3f}")

    if targets is not None:
        print("\n" + "-"*74); print(" [D] 3일 ΔOCV 상관"); print("-"*74)
        m = pd.DataFrame({"cid": ids, "I900": I[:, j9], "Ifin": Ifin}).merge(targets, on="cid")
        if len(m) < 3:
            print("  셀 ID 매칭 실패 — 건너뜁니다")
        else:
            print(f"  매칭 {len(m)}개, ΔOCV 중앙값 {m.docv.median():.3f} mV")
            for nm, v in [("15분 전류", m.I900), ("최종 전류", m.Ifin)]:
                print(f"    {nm} → ΔOCV   ρ = {spearmanr(v, m.docv).statistic:+.3f}")

    print("\n" + "="*74)
    print(" 공유용: [A] 2줄 + [B] 표 + [C] 4줄.  셀 단위 값은 없습니다.")
    print("="*74)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--inspect" in sys.argv or "-i" in sys.argv:
        inspect(args[0]) if args else print(__doc__)
    elif not args:
        print(__doc__)
    else:
        try:
            df, enc, sep, hdr = read_any(args[0])
            print(f"[읽기] 인코딩 {enc} / 구분자 '{sep}' / 헤더 {hdr}행 / {df.shape[0]:,}x{df.shape[1]}\n")
            t, I, ids, fmt = to_matrix(df)
        except Exception as e:
            print("!! 자동 판별 실패:", e)
            print("!! 아래 구조를 확인하고 알려주세요:\n")
            inspect(args[0]); sys.exit(1)
        tg = None
        if len(args) > 1:
            d2, *_ = read_any(args[1])
            cc = next((c for c in d2.columns if CELL_PAT.search(c)), d2.columns[0])
            oc = next((c for c in d2.columns if OCV_PAT.search(c)), d2.columns[-1])
            tg = d2[[cc, oc]].rename(columns={cc: "cid", oc: "docv"})
            tg["cid"] = tg.cid.astype(str)
        analyse(t, I, ids, fmt, tg)
