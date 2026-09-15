# -*- coding: utf-8 -*-
"""실행 결과를 항상 파일로 남긴다.

화면에 나온 것과 똑같은 내용이 analysis/results/ 에 자동 저장된다.
옵션을 주지 않아도 항상 저장되고, 저장 경로는 실행 끝에 출력된다.

  --save=경로   저장 위치를 직접 지정
  --no-save     저장하지 않음
"""
import sys, os, datetime, contextlib

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class _Tee:
    """화면과 파일에 동시에 쓴다. 쓸 때마다 파일을 flush 한다.

    flush 하지 않으면 파이썬이 출력을 버퍼에 모아두었다가 끝날 때 한 번에
    쓴다. 중간에 죽거나 사용자가 끊으면 저장 파일이 비어 버린다.
    오래 걸리는 스크립트일수록 그 구간이 길다.
    """
    def __init__(self, f, stream): self.f, self.s = f, stream
    def write(self, t):
        self.s.write(t)
        self.f.write(t); self.f.flush()
        return len(t)
    def flush(self): self.s.flush(); self.f.flush()
    def isatty(self): return getattr(self.s, "isatty", lambda: False)()


def _path(script, spec):
    stem = os.path.splitext(os.path.basename(str(spec).rstrip("/\\")))[0][:40]
    for ch in '<>:"/\\|?*': stem = stem.replace(ch, "_")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULT_DIR, f"{script}_{stem}_{ts}.txt")


@contextlib.contextmanager
def saving(script, spec, argv=None, save=None, enabled=True):
    """표준출력을 파일로도 복사한다. 실패해도 분석은 그대로 진행한다."""
    if not enabled:
        yield None; return
    path = save or _path(script, spec)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f = open(path, "w", encoding="utf-8")
    except OSError as e:
        print(f"  (결과 저장 실패: {e}. 화면 출력만 합니다.)")
        yield None; return
    old = sys.stdout
    sys.stdout = _Tee(f, old)
    try:
        print(f"# {script}   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# 입력: {spec}")
        if argv: print(f"# 명령: {' '.join(argv)}")
        print()
        yield path
    except BaseException as e:                     # Ctrl+C 도 포함
        print(f"\n# 중단됨: {type(e).__name__}: {e}")
        print("# 여기까지의 출력은 위에 남아 있습니다.")
        raise
    finally:
        sys.stdout = old
        try: f.close()
        except OSError: pass
        print(f"\n  결과 저장: {path}")


def parse(argv):
    """--save= / --no-save 를 읽는다."""
    save = next((a.split("=", 1)[1] for a in argv if a.startswith("--save=")), None)
    return save, "--no-save" not in argv
