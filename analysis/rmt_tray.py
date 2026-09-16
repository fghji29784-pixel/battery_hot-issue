# -*- coding: utf-8 -*-
"""
실제 SDM 원시 파일 포맷(TIME, I(01..144), V(01..144), T(01..144)) 전용 RMT 분석.

이 포맷은 슬라이드 원본 방법(여러 채널의 상관행렬)에 훨씬 더 직접적으로 들어맞는다.
한 셀의 시계열을 여러 창으로 쪼개는 대신, **같은 시각에 트레이의 144개 셀이
동시에 보인 전류(또는 전압/온도)를 채널로 삼아 144×144 상관행렬**을 만든다.
(금융의 RMT — N개 종목의 수익률 상관행렬 — 과 정확히 같은 구조. N=144)

왜 이게 유리한가
  기존 방식(screen.py/predict_xlsx.py)의 "트레이 정규화"는 각 시점에서
  144개 값의 중앙값을 빼는 것뿐이다 — 이건 트레이에 공통으로 걸린 상수
  오프셋만 제거한다. 그런데 실제 트레이 공통 교란(챔버 온도 흔들림,
  전원/접지 잡음 등)은 시간에 따라 변하고, 셀마다 그 교란에 반응하는
  민감도(loading)도 다르다. RMT는 시계열 "모양"까지 써서 이 공통 교란을
  분리해내므로, 단순 중앙값 차감보다 더 정확하게 셀 고유 신호를 남길 수 있다.

  또한 상관행렬 고유값이 Marchenko-Pastur(MP) 상한을 넘는 개수(n_above_mp)로
  "트레이 안에 공통 교란이 몇 개의 독립적인 패턴으로 존재하는지"를 정량화할
  수 있다 — 0개면 트레이 정규화 자체가 불필요하다는 뜻이고, 1개면 지금처럼
  단일 오프셋으로 충분할 수도(시간에 따라 변하지 않는다면) 있고, 2개 이상이면
  기존 중앙값 차감으로는 못 잡는 구조가 있다는 뜻이다.

사용법
  python3 rmt_tray.py 원시파일.txt --signal I --cutoff-s 300,600,900
  python3 rmt_tray.py 원시파일.txt --signal I --target delta_ocv.csv --cutoff-s 300,600,900
  python3 rmt_tray.py 원시파일.txt --signal I --vs-temp --target delta_ocv.csv   # 온도채널 직접보정과 비교

  target 파일: 컬럼에 채널 번호(1~144, 또는 01~144)와 delta_ocv 를 포함하는
  csv/xlsx. 있으면 raw / RMT보정 각각의 순위가 delta_ocv 순위와 얼마나
  일치하는지(Spearman + lift)까지 계산한다.

자체 테스트 (실제 파일 없이 원리 검증)
  python3 rmt_tray.py
"""
import re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hypergeom

CH_PAT = {
    "I": re.compile(r"^I\s*\(\s*0*(\d+)\s*\)$", re.I),
    "V": re.compile(r"^V\s*\(\s*0*(\d+)\s*\)$", re.I),
    "T": re.compile(r"^T\s*\(\s*0*(\d+)\s*\)$", re.I),
}
TIME_PAT = re.compile(r"^time$", re.I)


def _looks_like_header_token(tok):
    return bool(TIME_PAT.match(tok) or any(p.match(tok) for p in CH_PAT.values()))


def _read_text_any_encoding(path):
    """BOM 바이트를 직접 봐서 인코딩을 정한다. UTF-16(LE/BE)은 latin-1 같은
    '뭐든 받아주는' 인코딩으로 잘못 읽으면 예외 없이 조용히 깨진 텍스트가
    나오므로(널바이트가 낀 형태), 추측 순회가 아니라 BOM으로 먼저 확정한다.
    실측 장비 TXT는 cp949/euc-kr(윈도우 한글)이거나, 엑셀에서 저장한
    'Unicode 텍스트'(UTF-16)인 경우도 흔하다.
    """
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        candidates = ("utf-16",)
    elif head[:3] == b"\xef\xbb\xbf":
        candidates = ("utf-8-sig",)
    else:
        candidates = ("utf-8", "cp949", "euc-kr")
    candidates = candidates + tuple(e for e in ("utf-16", "utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1")
                                     if e not in candidates)
    last_err = None
    for enc in candidates:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
    raise ValueError(f"'{path}' 인코딩을 판별하지 못했습니다: {last_err}")


def _parse_by_token_stream(text):
    """구분자가 무엇이든, 헤더·데이터가 몇 줄에 걸쳐 접혀 있든 상관없이 복원한다.

    줄바꿈을 무시하고 파일 전체를 공백 기준 토큰 스트림으로 본 뒤,
    앞에서부터 TIME/I(..)/V(..)/T(..) 패턴에 맞는 토큰이 계속되는 구간을
    헤더로 인식하고(처음으로 패턴에 안 맞는 토큰이 나오면 헤더 끝),
    그 뒤 토큰을 헤더 길이만큼씩 잘라 데이터 행으로 되돌린다.
    실제 장비가 한 줄에 다 못 쓰고 여러 줄로 줄바꿈해 출력하는 포맷도
    이 방식이면 줄 경계와 무관하게 그대로 복원된다.
    """
    toks = text.split()
    ncol = 0
    for tok in toks:
        if _looks_like_header_token(tok):
            ncol += 1
        else:
            break
    if ncol < 4:
        raise ValueError("헤더(TIME, I(..), V(..), T(..)) 패턴을 토큰 스트림 앞부분에서 찾지 못했습니다. "
                          "컬럼 이름 표기가 예상과 다를 수 있습니다 (예: I(01) 대신 다른 표기).")
    header, body = toks[:ncol], toks[ncol:]
    nrow, rem = divmod(len(body), ncol)
    if nrow < 1:
        raise ValueError(f"헤더는 {ncol}개 인식했는데 그 뒤 데이터 토큰이 {len(body)}개뿐이라 행을 하나도 못 만듭니다.")
    if rem:
        body = body[:nrow * ncol]          # 마지막 불완전한 행(파일 끝 잘림 등)은 버림
    arr = np.array(body, dtype=object).reshape(nrow, ncol)
    return pd.DataFrame(arr, columns=header)


def _looks_structurally_sane(df):
    """I/V/T 채널 개수가 서로 같고, 컬럼이 TIME+I+V+T 딱 그만큼인지 확인.

    행이 여러 줄에 걸쳐 접힌 파일을 '한 줄=한 레코드'로 잘못 읽으면
    컬럼이 일부만 잡히거나(예: I(01)~I(09)만) TIME 파싱에 NaN이 쏟아지므로,
    이런 구조적 모순을 걸러 fast-path 오탐을 막는다.
    """
    cols = list(df.columns)
    if not any(TIME_PAT.match(c) for c in cols):
        return False
    counts = {s: sum(1 for c in cols if p.match(c)) for s, p in CH_PAT.items()}
    if any(v == 0 for v in counts.values()) or len(set(counts.values())) != 1:
        return False
    if 1 + sum(counts.values()) != len(cols):
        return False
    tcol = next(c for c in cols if TIME_PAT.match(c))
    tnum = pd.to_numeric(df[tcol], errors="coerce")
    if tnum.isna().mean() > 0.05:
        return False
    return True


def load_tray_txt(path):
    """탭/콤마/여러 칸 공백, 그리고 헤더·데이터가 여러 줄로 접힌 포맷까지 대응."""
    text, enc = _read_text_any_encoding(path)

    # 1차: 한 줄 = 한 레코드인 정상적인 표 형태를 빠르게 시도
    from io import StringIO
    for sep in ("\t", ",", r"\s+"):
        try:
            df = pd.read_csv(StringIO(text), sep=sep, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            if df.shape[1] > 3 and _looks_structurally_sane(df):
                return df
        except Exception:
            continue

    # 2차: 줄 경계를 무시하고 토큰 스트림으로 재구성 (여러 줄로 접힌 헤더/데이터 대응)
    try:
        df = _parse_by_token_stream(text)
        if not _looks_structurally_sane(df):
            raise ValueError(f"재구성은 됐지만 I/V/T 채널 개수가 서로 다르거나 TIME이 숫자로 안 읽힙니다 "
                              f"(컬럼 {df.shape[1]}개, 행 {df.shape[0]}개) — 줄바꿈 폭 추정이 실패했을 수 있습니다.")
        print(f"  [참고] 표 형태로 바로 안 읽혀서 줄바꿈을 무시하고 토큰 단위로 재구성했습니다."
              f" (인코딩={enc}, 컬럼 {df.shape[1]}개, 행 {df.shape[0]}개)")
        return df
    except Exception as e:
        head = text[:200].replace("\n", "\\n")
        raise ValueError(f"'{path}' 구조를 인식하지 못했습니다 (인코딩={enc}). "
                          f"파일 시작 200자: {head!r}\n원인: {e}")


def to_matrix(df, signal="I"):
    """df에서 지정 신호의 (T, N) 행렬과 채널 번호, 시간축을 뽑는다."""
    pat = CH_PAT[signal]
    cols = [(int(pat.match(c).group(1)), c) for c in df.columns if pat.match(c)]
    if not cols:
        raise ValueError(f"'{signal}' 채널 컬럼을 못 찾았습니다. 예: I(01)")
    cols.sort()
    ch_ids = np.array([c for c, _ in cols])
    X = df[[c for _, c in cols]].apply(pd.to_numeric, errors="coerce").values
    tcol = next((c for c in df.columns if TIME_PAT.match(c)), df.columns[0])
    t = pd.to_numeric(df[tcol], errors="coerce").values
    return t, X, ch_ids


# ── RMT 핵심 ────────────────────────────────────────────────────────────
def mp_edge(N, T):
    q = N / T
    return (1 + np.sqrt(q)) ** 2


def cross_channel_eigen(X):
    """X:(T,N). 채널별로 중심화+표준화한 뒤 채널-간 상관행렬을 고유분해."""
    T, N = X.shape
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-12] = 1.0
    Xc = X - mu
    Xs = np.nan_to_num(Xc / sd)
    C = (Xs.T @ Xs) / T
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order], Xs, mu, sd


def remove_common_modes(X, k):
    """상위 k개 공통모드(트레이 공통 교란)를 채널별 회귀로 제거.

    처음에는 "표준화 공간에서 고유벡터 성분을 빼고 원 스케일로 되돌리는"
    방식으로 구현했었으나, 이는 각 채널의 표준편차(sd)가 공통교란 자체에
    의해 부풀려져 있을 때(공통교란이 지배적일 때) 되돌리는 과정에서 그
    채널의 고유 잡음까지 함께 증폭시켜 버리는 결함이 있었다 (합성 데이터로
    검증 중 발견 — MSE는 개선되어도 순위상관은 오히려 나빠졌음).

    올바른 방법은 공통 인자 시계열 F(t)를 뽑아낸 뒤, 채널별로
    X_i(t) = α_i + F(t)·β_i + resid_i(t) 를 최소자승 회귀로 적합하고
    (α_i + resid_i(t))만 남기는 것 — 즉 각 채널이 그 공통 인자에 실제로
    얼마나(β_i) 반응했는지를 그 채널 자신의 물리 단위로 직접 추정한다.
    """
    T, N = X.shape
    eigvals, eigvecs, Xs, mu, sd = cross_channel_eigen(X)
    if k <= 0:
        return X.copy(), eigvals, eigvecs

    F = Xs @ eigvecs[:, :k]                        # (T,k) 공통 인자 시계열
    Fc = F - F.mean(axis=0, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    beta = np.linalg.pinv(Fc) @ Xc                  # (k,N) 채널별 회귀계수
    fitted_common = Fc @ beta                        # (T,N)
    X_clean = X - fitted_common                       # 채널 고유 평균·변동은 그대로 유지
    return X_clean, eigvals, eigvecs


def choose_k(eigvals, N, T):
    lam = mp_edge(N, T)
    return int(np.sum(eigvals > lam)), lam


def regress_against_paired_channel(X, Z):
    """채널별로 X_i(t) = a_i + b_i·Z_i(t) + resid_i(t) 회귀.

    RMT처럼 '정체불명의 공통 인자'를 상관행렬에서 추정하는 게 아니라,
    이미 원인으로 의심되는 채널(예: 그 셀 자신의 온도)이 있을 때
    그 채널로 직접 회귀하는 더 단순하고 해석 가능한 보정법.
    18_BT2152...md의 Ea≈0.94eV 아레니우스 관계처럼, 원인이 물리적으로
    특정된 경우엔 RMT보다 이 방법을 먼저 시도하는 게 맞다 — 온도 채널이
    이미 있다면 PCA로 그걸 다시 '발견'할 필요가 없다.
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    b = (Zc*Xc).sum(0) / np.clip((Zc**2).sum(0), 1e-12, None)
    a = X.mean(0) - b*Z.mean(0)
    resid = X - (a[None, :] + b[None, :]*Z)
    return a[None, :] + resid, b


# ── 평가(선별 관점) ──────────────────────────────────────────────────────
def lift(score, y, ks=(0.02, 0.05, 0.10)):
    n = len(y)
    out = []
    for q in ks:
        k = max(int(round(n*q)), 1)
        a = set(np.argsort(-y)[:k]); b = set(np.argsort(-score)[:k])
        hit = len(a & b); exp = k*k/n
        p = hypergeom.sf(hit-1, n, k, k) if hit else 1.0
        out.append((q, k, hit, exp, hit/max(exp, 1e-9), p))
    return out


def print_lift(rows, label):
    print(f"    [{label}]")
    for q, k, hit, exp, lf, p in rows:
        print(f"      상위{q*100:>5.0f}%  {hit}/{k}  (무작위 {exp:.1f})  {lf:>5.1f}배   p={p:.1e}")


def evaluate_against_target(endpoints, ch_ids, target_df):
    """endpoints: {'raw':arr, 'rmt':arr, ...} (채널 순서 = ch_ids). target_df: 채널,delta_ocv."""
    tcol = next((c for c in target_df.columns if re.search(r"ch|channel|cell", c, re.I)), target_df.columns[0])
    ycol = next((c for c in target_df.columns if re.search(r"delta.*ocv|ocv.*delta|dOCV", c, re.I)), None)
    if ycol is None:
        raise ValueError("target 파일에서 delta_ocv 컬럼을 못 찾았습니다.")
    tmap = dict(zip(pd.to_numeric(target_df[tcol], errors="coerce"), target_df[ycol]))
    y = np.array([tmap.get(int(c), np.nan) for c in ch_ids], dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 5:
        raise ValueError("target과 채널 번호가 매칭되지 않습니다 (채널 ID 형식을 확인하세요).")
    print(f"\n    타깃 매칭 {ok.sum()}/{len(ch_ids)}채널")
    for name, sc in endpoints.items():
        rho = spearmanr(sc[ok], y[ok]).statistic
        print(f"\n    ρ(순위상관, {name} vs delta_ocv) = {rho:+.3f}")
        print_lift(lift(sc[ok], y[ok]), name)


# ── CLI ──────────────────────────────────────────────────────────────────
def main(path, signal="I", cutoffs_s=(180, 300, 600, 900), target_path=None, vs_temp=False):
    df = load_tray_txt(path)
    t, X, ch_ids = to_matrix(df, signal)
    T_mat = None
    if vs_temp:
        if signal == "T":
            print("  --vs-temp 는 signal=I 일 때만 의미가 있습니다 (무시).")
        else:
            _, T_mat, t_ids = to_matrix(df, "T")
            assert np.array_equal(ch_ids, t_ids), "I/T 채널 번호가 서로 다릅니다"
    print("="*78)
    print(f" {path}  —  신호 {signal}  /  채널 {len(ch_ids)}개  /  전체 시료 {len(t)}개")
    print("="*78)

    target_df = None
    if target_path:
        target_df = pd.read_excel(target_path) if str(target_path).lower().endswith(("xlsx","xls")) \
                    else pd.read_csv(target_path)

    for cut in cutoffs_s:
        m = t <= cut
        if m.sum() < 10:
            continue
        Xc = X[m]
        T, N = Xc.shape
        n_above, lam_plus = 0, mp_edge(N, T)
        eigvals, eigvecs, _, _, _ = cross_channel_eigen(Xc)
        n_above = int(np.sum(eigvals > lam_plus))

        print(f"\n{'-'*78}\n 구간 ≤{cut}s (T={T}, N={N}, MP상한={lam_plus:.3f})")
        print(f" 상위 5개 고유값: {np.round(eigvals[:5], 3).tolist()}")
        print(f" MP 상한을 넘는 '공통모드' 개수: {n_above}")
        print(f"{'-'*78}")

        k = max(n_above, 1)                       # 최소 1개(1st PC=시장모드)는 제거해 비교
        X_clean, _, _ = remove_common_modes(Xc, k)
        raw_end = Xc[-1]
        rmt_end = X_clean[-1]

        variants = {"raw(원본 끝값)": raw_end,
                    f"RMT보정(공통모드 {k}개 제거)": rmt_end}
        if T_mat is not None:
            X_tcorr, beta_t = regress_against_paired_channel(Xc, T_mat[m])
            variants[f"온도채널 직접회귀보정 (β 평균={beta_t.mean():.2f} µA/℃)"] = X_tcorr[-1]

        if target_df is not None:
            evaluate_against_target(variants, ch_ids, target_df)
        else:
            print(f"    (--target 없음 — 순위상관은 delta_ocv 파일을 주시면 계산합니다)")
            print(f"    raw 끝값 표준편차={raw_end.std():.3f}   RMT보정 후 표준편차={rmt_end.std():.3f}")


# ── 자체 테스트: 합성 144채널 트레이로 원리 검증 ─────────────────────────
def _make_synthetic_tray(n_ch=144, T=600, dt=1.0, defect_frac=0.02, common_scale=400.0, seed=0):
    """common_scale : 트레이 공통교란의 세기. 작으면(수십) 무시할 수준, 크면(수백)
    개별 셀 신호(I_sd 차이)를 압도할 만큼 강함 — 실측에서 어느 쪽인지는
    사용자 데이터로 직접 확인해야 한다 (아래 main()의 '공통모드 개수/크기' 출력 참고)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * dt
    I_sd = rng.lognormal(np.log(120), 0.25, n_ch)
    is_defect = rng.random(n_ch) < defect_frac
    I_sd[is_defect] *= rng.uniform(1.8, 4.0, is_defect.sum())

    A1 = rng.lognormal(np.log(400), 0.4, n_ch)
    tau1 = rng.lognormal(np.log(180), 0.3, n_ch)
    base = I_sd[None, :] + A1[None, :] * np.exp(-t[:, None] / tau1[None, :])

    # 시간에 따라 변하는 트레이 공통 교란 (예: 챔버 온도 흔들림) + 채널별 민감도
    steps = rng.normal(0, 0.05, T)
    common = np.cumsum(steps)
    common = pd.Series(common).rolling(15, min_periods=1, center=True).mean().to_numpy().copy()
    common -= common.mean()
    loadings = rng.normal(1.0, 0.2, n_ch)
    common_effect = common[:, None] * loadings[None, :] * common_scale

    noise = rng.normal(0, 0.25, (T, n_ch))
    I = base + common_effect + noise

    dQ = I_sd * 1e-3 * 72.0                       # µA×h → mAh (72시간 보관 가정)
    delta_ocv = dQ / (5.0*1000) * 100 * 5.0        # mV
    return t, I, delta_ocv, is_defect


def _selftest():
    print("="*78)
    print(" 자체 테스트 A: 공통교란 세기에 따라 raw / 트레이중앙값 / RMT회귀보정 비교")
    print(" (시드 30개 평균 — 한 번의 우연한 결과가 아님을 확인하기 위함)")
    print(" median_sub는 raw와 항상 순위가 동일함 (상수를 빼는 것은 순위를 바꾸지 않음)")
    print("="*78)
    n_seed = 30
    print(f"\n  {'공통교란 세기':>14}{'raw ρ':>14}{'RMT회귀보정 ρ':>18}")
    for scale in (30, 100, 300, 600):
        rows = []
        for seed in range(n_seed):
            t, I, delta_ocv, _ = _make_synthetic_tray(common_scale=scale, seed=seed)
            Xc = I[t <= 600]
            T, N = Xc.shape
            eigvals, eigvecs, _, _, _ = cross_channel_eigen(Xc)
            k = max(int(np.sum(eigvals > mp_edge(N, T))), 1)
            X_clean, _, _ = remove_common_modes(Xc, k)
            rows.append((spearmanr(Xc[-1], delta_ocv).statistic,
                        spearmanr(X_clean[-1], delta_ocv).statistic))
        rows = np.array(rows)
        print(f"  {scale:>14}{rows[:,0].mean():>9.3f}±{rows[:,0].std():.2f}"
              f"{rows[:,1].mean():>13.3f}±{rows[:,1].std():.2f}")
    print("\n  → 공통교란이 약하면(위쪽) raw가 낫거나 비슷하고, 세지면(아래쪽) RMT회귀보정이")
    print("    앞서면서 변동성(±)도 작아진다 — '트레이 잡음이 실제로 셀 신호를 압도할 때만'")
    print("    RMT 보정이 이득이라는 뜻. 실측 데이터에서 이 세기가 어느 쪽인지 확인이 먼저다.")

    print("\n" + "="*78)
    print(" 자체 테스트 B: 공통교란이 강한(600) 한 트레이에서 구간별 상세 비교")
    print("="*78)
    t, I, delta_ocv, is_defect = _make_synthetic_tray(common_scale=600, seed=1)

    for cut in (120, 300, 600):
        m = t <= cut
        Xc = I[m]
        T, N = Xc.shape
        eigvals, eigvecs, _, _, _ = cross_channel_eigen(Xc)
        lam_plus = mp_edge(N, T)
        n_above = int(np.sum(eigvals > lam_plus))
        k = max(n_above, 1)
        X_clean, _, _ = remove_common_modes(Xc, k)

        raw_end, rmt_end = Xc[-1], X_clean[-1]
        median_sub = raw_end - np.median(raw_end)   # 기존 방식(트레이 중앙값 차감) 참고용

        print(f"\n{'-'*78}\n 구간 ≤{cut}s  (T={T}, MP상한={lam_plus:.3f}, 공통모드 {n_above}개 검출 → k={k})")
        for name, sc in [("raw 끝값", raw_end),
                         ("기존 방식: 트레이 중앙값 차감", median_sub),
                         (f"RMT 회귀보정 (공통모드 {k}개 제거)", rmt_end)]:
            rho = spearmanr(sc, delta_ocv).statistic
            print(f"    {name:<32} ρ={rho:+.3f}")
            print_lift(lift(sc, delta_ocv), name)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        _selftest()
    else:
        sig = "I"
        cuts = (180, 300, 600, 900)
        tgt = None
        vs_temp = "--vs-temp" in sys.argv
        for a in sys.argv:
            if a.startswith("--signal="): sig = a.split("=", 1)[1]
            if a.startswith("--cutoff-s="): cuts = tuple(int(x) for x in a.split("=", 1)[1].split(","))
            if a.startswith("--target="): tgt = a.split("=", 1)[1]
        main(args[0], sig, cuts, tgt, vs_temp)
