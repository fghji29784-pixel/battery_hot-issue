# -*- coding: utf-8 -*-
"""트레이별 원시 곡선 파일들을 하나로 모은다 — 전류·전압·온도

  python ingest.py "폴더"  --inspect        # ★ 먼저 이것부터. 구조만 확인
  python ingest.py "폴더"  -o 모은것.xlsx    # 표준 스키마(i_XXmin)로 합치기
  python ingest.py "폴더"  -o 모은것.xlsx --step=0.5   # 0.5분 간격으로
  python ingest.py "폴더"  --fine=원본해상도.csv       # 원해상도 long 도 함께
  python ingest.py "폴더"  -o 모은것.xlsx --interp      # 평균 대신 보간 (권장 안 함)

 ※ 단위는 원본 그대로 둔다 (전류 uA, 전압 mV, 온도 섭씨).
   본체 엑셀의 전류는 A 단위라 1e6 배 차이가 난다. 두 출처의 전류 값을
   같은 표에 섞지 말 것. 분석은 전부 순위·상대값 기반이라 단위 자체는
   결과에 영향이 없다.

────────────────────────────────────────────────────────────────────────
 왜 필요한가
────────────────────────────────────────────────────────────────────────
 지금 쓰는 본체 엑셀은 i_5min ~ i_30min 의 1분 평균 26점이다.
 그래서 두 가지가 막혀 있었다.

   · 0~5분 구간이 없다 → 5분 창에 점이 1개뿐이다 (부록 23)
   · 1분 평균이라 모양 차이를 볼 수 없다 → 모양 분해가 기각됐다 (부록 18)
   · 온도가 t_init / t_final 두 점뿐이다 → dT/dt 를 30분 평균으로만 안다

 트레이별 원시 파일에는 시간에 따른 전류·전압·온도가 다 있다.
 그것을 셀 단위로 펴서 기존 스크립트가 그대로 먹을 수 있는 형태로 만든다.

────────────────────────────────────────────────────────────────────────
 하는 일
────────────────────────────────────────────────────────────────────────
 [1] 파일마다 구조를 자동 판별한다
     · wide  : TIME + 셀별 컬럼 (I(01), V(01), T(01) … 또는 Current_1 …)
     · long  : cell / time / current / voltage / temp
     · 트레이 ID 는 컬럼에서 찾고, 없으면 파일명에서 뽑는다
 [2] 시간축을 공통 격자에 맞춘다 (기본 1분, --step 으로 조정)
 [3] 셀 하나가 한 행인 표로 펴서 저장한다
     i_0min, i_1min, …  /  v_0min, …  /  t_0min, …
     기존 스크립트의 컬럼 규칙(i_XXmin)을 그대로 따르므로
     rescue.py · short.py · decompose.py 에 바로 넣을 수 있다.
 [4] 본체 엑셀과 합칠 수 있게 tray_id · cell_no 를 만든다

 ★ --inspect 를 먼저 돌릴 것. 파일 구조를 모르는 채로 합치면 안 된다.
"""
import sys, os, re, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from diagnose import read_any

# 실제 컬럼명은 time_s, Time(sec), 경과시간, TIME[min] 처럼 접미어가 붙는다.
# 완전일치로 잡으면 long 파일의 시간축을 놓친다. 앞부분 일치로 잡되,
# temp / tray 가 't' 로 시작하는 것에 걸리지 않게 짧은 형태는 완전일치로 둔다.
TIME_PAT = re.compile(r"^(time|시간|경과|elapsed|stamp)|^(t|sec|second|min|minute)s?$", re.I)
CUR_PAT  = re.compile(r"(current|전류|^i[\s_(\[]|\bI\b|amp|\bua\b|\bma\b)", re.I)
VOL_PAT  = re.compile(r"(volt|전압|^v[\s_(\[]|\bV\b|\bmv\b)", re.I)
TMP_PAT  = re.compile(r"(temp|온도|^t[\s_(\[]|degc|℃)", re.I)
CELL_PAT = re.compile(r"(cell|셀|channel|ch|slot|serial|barcode|번호|no\b)", re.I)
TRAY_PAT = re.compile(r"(tray|트레이|rack|lot|랏|batch)", re.I)
IDX_PAT  = re.compile(r"(\d+)")


def tray_from_name(path):
    """파일명에서 트레이 ID 를 뽑는다. 영문+숫자 덩어리 중 가장 긴 것."""
    stem = os.path.splitext(os.path.basename(path))[0]
    cands = re.findall(r"[A-Za-z]{2,}\d{3,}|\d{6,}", stem)
    return max(cands, key=len) if cands else stem


def split_wide(df):
    """wide 파일에서 시간 컬럼과 전류/전압/온도 블록을 가른다."""
    cols = [str(c) for c in df.columns]
    tcol = next((c for c in cols if TIME_PAT.match(c.strip())), None)
    if tcol is None:
        num = df.select_dtypes("number")
        inc = [c for c in num.columns if num[c].is_monotonic_increasing and num[c].nunique() > 5]
        tcol = inc[0] if inc else (num.columns[0] if len(num.columns) else None)
    rest = [c for c in cols if c != tcol]
    blocks = {"i": [], "v": [], "t": []}
    for c in rest:
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        if   VOL_PAT.search(c): blocks["v"].append(c)
        elif TMP_PAT.search(c): blocks["t"].append(c)
        elif CUR_PAT.search(c): blocks["i"].append(c)
    if not blocks["i"] and rest:                    # 접두어가 없으면 전부 전류로 본다
        blocks["i"] = [c for c in rest if pd.api.types.is_numeric_dtype(df[c])]
    return tcol, blocks


def cell_key(name):
    """컬럼명에서 셀 번호를 뽑는다. I(02) → 2, Current_7 → 7."""
    m = IDX_PAT.findall(str(name))
    return int(m[-1]) if m else None


def load_one(path):
    """파일 하나 → (시간[분], {종류: DataFrame(셀번호 x 시간)}, 형식, 트레이)."""
    df, enc, sep, hdr = read_any(path)
    cols = [str(c) for c in df.columns]
    tray = next((c for c in cols if TRAY_PAT.search(c)), None)
    tray_id = str(df[tray].dropna().iloc[0]) if tray and df[tray].notna().any() else tray_from_name(path)

    ccol = next((c for c in cols if CELL_PAT.search(c) and not TIME_PAT.match(c.strip())), None)
    tcol = next((c for c in cols if TIME_PAT.match(c.strip())), None)
    icol = next((c for c in cols if CUR_PAT.search(c) and c not in (tcol, ccol, tray)), None)
    if ccol and tcol and icol and df[ccol].nunique() > 2:                    # long
        out = {}
        for k, pat in (("i", CUR_PAT), ("v", VOL_PAT), ("t", TMP_PAT)):
            col = next((c for c in cols if pat.search(c) and c not in (tcol, ccol, tray)), None)
            if col is None: continue
            W = df.pivot_table(index=ccol, columns=tcol, values=col, aggfunc="mean")
            out[k] = W
        tv = sorted(df[tcol].dropna().unique().astype(float))
        return np.array(tv), out, "long", tray_id, enc

    tcol, blocks = split_wide(df)                                            # wide
    if tcol is None: raise ValueError("시간 컬럼을 찾지 못했습니다.")
    tv = pd.to_numeric(df[tcol], errors="coerce").values.astype(float)
    out = {}
    for k, cs in blocks.items():
        if not cs: continue
        W = pd.DataFrame(df[cs].values.T, index=[cell_key(c) or j for j, c in enumerate(cs)],
                         columns=tv)
        out[k] = W
    return tv, out, "wide", tray_id, enc


def bin_mean(x, y, grid, step, how="mean"):
    """격자 점마다 그 구간 안의 원본 값을 평균한다.

    1 Hz 원본을 1분 격자로 옮길 때, 그 시각의 값 하나만 쓰면(보간) 잡음이
    그대로 남는다. 구간 안 60점을 평균하면 잡음이 sqrt(60) = 7.7배 준다.
    1 Hz 를 받아오는 이득의 대부분이 여기서 나온다.
    원본이 이미 성기면(격자보다 간격이 크면) 자동으로 보간으로 넘어간다.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if how == "interp" or len(x) < 2 or np.median(np.diff(x)) >= step * 0.75:
        return np.interp(grid, x, y, left=np.nan, right=np.nan)
    edges = np.concatenate([grid - step / 2.0, [grid[-1] + step / 2.0]])
    idx = np.clip(np.searchsorted(edges, x, "right") - 1, 0, len(grid) - 1)
    ok = (x >= edges[0]) & (x <= edges[-1])
    ssum = np.bincount(idx[ok], y[ok], minlength=len(grid))
    cnt = np.bincount(idx[ok], minlength=len(grid)).astype(float)
    out = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    if np.isnan(out).any():                       # 빈 칸은 보간으로 메운다
        m = ~np.isnan(out)
        if m.sum() >= 2: out[~m] = np.interp(grid[~m], grid[m], out[m])
    return out


def to_minutes(tv):
    """시간축 단위를 추정해 분으로 바꾼다."""
    tv = np.asarray(tv, float); span = np.nanmax(tv) - np.nanmin(tv)
    if span > 3600: return tv / 60.0, "초(추정)"          # 1시간 넘으면 초
    if span > 100:  return tv / 60.0, "초(추정)"
    return tv, "분(추정)"


def main(spec, out=None, step=1.0, fine=None, inspect=False, limit=None, how="mean"):
    files = sorted(sum([glob.glob(os.path.join(spec, e)) for e in
                        ("*.csv", "*.CSV", "*.txt", "*.xlsx", "*.xls")], [])) \
        if os.path.isdir(spec) else sorted(glob.glob(spec))
    if not files:
        print(f"  !! 파일을 찾지 못했습니다: {spec}"); return
    print("=" * 78); print(" 트레이별 원시 곡선 수집"); print("=" * 78)
    print(f"  파일 {len(files)}개  ({os.path.dirname(files[0]) or '.'})")

    # ── [1] 구조 확인 ────────────────────────────────────────────
    print("\n" + "-" * 78); print(" [1] 파일 구조 자동 판별"); print("-" * 78)
    n_show = min(3, len(files))
    for f in files[:n_show]:
        try:
            tv, blocks, fmt, tray, enc = load_one(f)
            tm, unit = to_minutes(tv)
            ncell = max((b.shape[0] for b in blocks.values()), default=0)
            print(f"    {os.path.basename(f)[:44]:<46}")
            print(f"      형식 {fmt} / 인코딩 {enc} / 트레이 '{tray}'")
            print(f"      시간 {np.nanmin(tm):.2f} ~ {np.nanmax(tm):.2f}분 {unit}, {len(tm)}점"
                  f"  (간격 중앙 {np.median(np.diff(np.sort(tm))) * 60:.1f}초)")
            print(f"      셀 {ncell}개 / 담긴 종류: "
                  + ", ".join({"i": "전류", "v": "전압", "t": "온도"}[k] for k in blocks))
        except Exception as e:
            print(f"    {os.path.basename(f)[:44]:<46} !! 읽기 실패: {type(e).__name__}: {e}")
    if len(files) > n_show: print(f"    … 나머지 {len(files) - n_show}개는 생략")

    if inspect:
        print("""
    ★ 확인할 것
      · 시간이 0 부터 시작하는가 — 0~5분 구간이 있어야 5분 선별이 산다
      · 간격이 몇 초인가 — 촘촘할수록 잡음이 줄고 모양 분해가 가능해진다
      · 온도가 셀별로 있는가 — 있으면 dT/dt 를 시점마다 구할 수 있다
      · 트레이 ID 가 본체 엑셀의 tray_id 와 같은 형식인가
      · 셀 번호가 본체 엑셀의 cell_no 와 맞는가 (자리 복원에 필요)
    문제 없으면 -o 로 합치십시오.""")
        print("=" * 78); return

    # ── [2] 공통 격자로 모으기 ───────────────────────────────────
    print("\n" + "-" * 78)
    print(f" [2] 공통 시간 격자로 정렬  (간격 {step}분)")
    print("-" * 78)
    rows, fine_rows, bad, tmax = [], [], 0, []
    for f in files if limit is None else files[:limit]:
        try:
            tv, blocks, fmt, tray, _ = load_one(f)
        except Exception:
            bad += 1; continue
        tm, _ = to_minutes(tv)
        ok = np.isfinite(tm)
        if ok.sum() < 3: bad += 1; continue
        tmax.append(np.nanmax(tm[ok]))
        grid = np.arange(0, np.nanmax(tm[ok]) + 1e-9, step)
        ids = sorted(set().union(*[set(b.index) for b in blocks.values()]))
        rec = {c: {"tray_id": tray, "cell_no": c} for c in ids}
        for k, W in blocks.items():
            cv = np.asarray(W.columns, float)
            cm, _ = to_minutes(cv)
            o = np.argsort(cm)
            for c in W.index:
                y = np.asarray(W.loc[c].values, float)[o]
                m = np.isfinite(y)
                if m.sum() < 2: continue
                yi = bin_mean(cm[o][m], y[m], grid, step, how)
                for gi, gv in zip(grid, yi):
                    rec[c][f"{k}_{gi:g}min"] = gv
                if fine is not None and k == "i":
                    for tt, yy in zip(cm[o][m], y[m]):
                        fine_rows.append((tray, c, tt, yy))
        rows.extend(rec.values())
    if not rows:
        print("  !! 읽어낸 셀이 없습니다. --inspect 로 구조를 확인해 주세요."); return
    D = pd.DataFrame(rows)
    print(f"    구간 처리: {'구간 평균 (잡음 감소)' if how == 'mean' else '보간 (--interp)'}")
    print(f"    합친 셀 {len(D):,}개 / 트레이 {D['tray_id'].nunique()}개"
          + (f"  (읽기 실패 {bad}개)" if bad else ""))
    print(f"    측정 길이 중앙 {np.median(tmax):.1f}분")
    got = [k for k in ("i", "v", "t") if any(c.startswith(k + "_") for c in D.columns)]
    print(f"    담긴 종류: " + ", ".join({"i": "전류", "v": "전압", "t": "온도"}[k] for k in got))
    t0 = [c for c in D.columns if re.match(r"^i_0min$", c)]
    print(f"    0분 시점 {'있음' if t0 else '없음'}"
          + ("   ← 5분 창에 여러 점이 생긴다" if t0 else "   ← 여전히 시작 시점부터다"))

    if out:
        (D.to_excel(out, index=False) if str(out).lower().endswith((".xlsx", ".xlsm"))
         else D.to_csv(out, index=False, encoding="utf-8-sig"))
        print(f"\n  저장: {out}   {D.shape[0]:,}행 x {D.shape[1]}열")
        print("""
  다음
    · 본체 엑셀의 판정등급·ΔOCV 를 tray_id + cell_no 로 붙이십시오.
      (엑셀에서 VLOOKUP 이든, pandas merge 든)
    · 그 다음 기존 스크립트를 그대로 돌릴 수 있습니다.
        python analysis/short.py  "합친것.xlsx" --at=5
        python analysis/rescue.py "합친것.xlsx"
        python analysis/decompose.py "합친것.xlsx"
    ※ 컬럼 규칙(i_XXmin)을 그대로 따르므로 별도 수정이 필요 없습니다.""")
    if fine and fine_rows:
        F = pd.DataFrame(fine_rows, columns=["tray_id", "cell_no", "time_min", "current"])
        F.to_csv(fine, index=False, encoding="utf-8-sig")
        print(f"  원해상도 저장: {fine}   {len(F):,}행  (잡음 구조 분석용)")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if not a: print(__doc__)
    else:
        out = fine = None; step = 1.0; lim = None
        for i, x in enumerate(sys.argv):
            if x == "-o" and i + 1 < len(sys.argv): out = sys.argv[i + 1]
            if x.startswith("--out="):  out = x.split("=", 1)[1]
            if x.startswith("--fine="): fine = x.split("=", 1)[1]
            if x.startswith("--step="): step = float(x.split("=")[1])
            if x.startswith("--limit="): lim = int(x.split("=")[1])
        outs = {out, fine} - {None}
        main(a[0], out, step, fine, "--inspect" in sys.argv, lim,
             "interp" if "--interp" in sys.argv else "mean")
