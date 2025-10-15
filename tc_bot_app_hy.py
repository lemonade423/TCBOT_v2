import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests
import re
# [ADD] 미리보기/샘플 생성용
import io
from collections import Counter

# ✅ OpenRouter API Key (보안을 위해 secrets.toml 또는 환경변수 사용 권장)
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_KEY")

if not API_KEY:
    st.warning(
        "⚠️ OpenRouter API Key가 설정되지 않았습니다. .streamlit/secrets.toml에 OPENROUTER_API_KEY 항목을 추가하세요."
    )

st.set_page_config(page_title="🧠 TC-Bot: QA 자동화 도우미", layout="wide")
st.title("🤖 TC-Bot: AI 기반 QA 자동화 도우미")

# ✅ 세션 초기화 (탭 선언보다 먼저 수행해야 함)
for key in ["scenario_result", "spec_result", "llm_result", "parsed_df", "last_uploaded_file", "last_model", "last_role", "is_loading"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["is_loading"] is None:
    st.session_state["is_loading"] = False


# ✅ 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox("🤖 사용할 LLM 모델", ["qwen/qwen-max", "mistral"])
    qa_role = st.selectbox("👤 QA 역할", ["기능 QA", "보안 QA", "성능 QA"])
    st.session_state["qa_role"] = qa_role

# ✅ 기존 3개 탭 유지
code_tab , tc_tab, log_tab = st.tabs(
    ["🧪 소스코드 → 테스트케이스 자동 생성","📑 테스트케이스 → 명세서 요약","🐞 에러 로그 → 재현 시나리오"] )

# ✅ LLM 호출 중 경고 표시 (탭 차단하지 않음)
if st.session_state["is_loading"]:
    st.warning("⚠️ 현재 LLM 호출 중입니다. 탭 이동은 가능하지만 다른 요청은 완료 후 시도해 주세요.")
else:
    st.empty()

# ────────────────────────────────────────────────
# 🔧 유틸 함수: 에러 로그 전처리 (기존 유지)
# ────────────────────────────────────────────────
MODEL_TOKEN_LIMITS = {
    "qwen/qwen-max": 30720,
    "mistral": 8192,
}


def safe_char_budget(model: str, token_margin: int = 1024) -> int:
    limit_tokens = MODEL_TOKEN_LIMITS.get(model, 8192)
    usable_tokens = max(1024, limit_tokens - token_margin)
    return usable_tokens * 4


def preprocess_log_text(text: str,
                        context_lines: int = 3,
                        keep_last_lines_if_empty: int = 1500,
                        char_budget: int = 120000) -> tuple[str, dict]:
    lines = text.splitlines()
    total_lines = len(lines)
    non_debug = [(i, line) for i, line in enumerate(lines)
                 if "DEBUG" not in line]
    patt = re.compile(r"(ERROR|Exception|WARN|FATAL)", re.IGNORECASE)
    matched_indices = [i for i, line in non_debug if patt.search(line)]
    selected = set()
    if matched_indices:
        for mi in matched_indices:
            orig_idx = non_debug[mi][0]
            for j in range(max(0, orig_idx - context_lines),
                           min(total_lines, orig_idx + context_lines + 1)):
                selected.add(j)
        focused = [lines[j] for j in sorted(selected)]
        header = [
            "### Log Focus (ERROR/WARN/Exception 중심 발췌)",
            f"- 전체 라인: {total_lines:,}", f"- 컨텍스트 포함 라인: {len(selected):,}", ""
        ]
        trimmed = "\n".join(header + focused)
    else:
        tail = lines[-keep_last_lines_if_empty:]
        header = [
            "### Log Tail (매치 없음 → 마지막 일부 사용)", f"- 전체 라인: {total_lines:,}",
            f"- 사용 라인(마지막): {len(tail):,}", ""
        ]
        trimmed = "\n".join(header + tail)
    if len(trimmed) > char_budget:
        trimmed = trimmed[-char_budget:]
    stats = {
        "total_lines": total_lines,
        "kept_chars": len(trimmed),
        "char_budget": char_budget
    }
    return trimmed, stats

# ────────────────────────────────────────────────
# [ADD] 샘플 파일 생성 & 결과 미리보기(휴리스틱) 유틸
# ────────────────────────────────────────────────
def build_sample_code_zip() -> bytes:
    """간단한 3개 파일로 구성된 샘플 코드 ZIP (테스트케이스 자동 생성 입력용)"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("app.py",
                    "# FILE: app.py\n"
                    "def add(a, b):\n"
                    "    return a + b\n\n"
                    "def div(a, b):\n"
                    "    if b == 0:\n"
                    "        raise ZeroDivisionError('b must not be zero')\n"
                    "    return a / b\n")
        zf.writestr("utils/validator.py",
                    "# FILE: utils/validator.py\n"
                    "def is_email(s: str) -> bool:\n"
                    "    return '@' in s and '.' in s.split('@')[-1]\n")
        zf.writestr("README.md",
                    "# Sample Project\n\n"
                    "- add(a,b), div(a,b), is_email(s) 함수 포함\n"
                    "- 단순 산술/검증 로직으로 테스트케이스 생성 시연용")
    return buf.getvalue()

# [ADD] Tab2용: 샘플 테스트케이스 XLSX (요구사항: Tab2에 필요)
def build_sample_tc_excel() -> bytes:
    df = pd.DataFrame([
        ["TC-001", "덧셈 기능", "a=1, b=2", "3 반환", "High"],
        ["TC-002", "나눗셈 기능(정상)", "a=6, b=3", "2 반환", "Medium"],
        ["TC-003", "나눗셈 기능(예외)", "a=1, b=0", "ZeroDivisionError 발생", "High"],
        ["TC-004", "이메일 검증(정상)", "s='user@example.com'", "True 반환", "Low"],
        ["TC-005", "이메일 검증(이상)", "s='invalid@domain'", "False 또는 규칙 위반 처리", "Low"],
    ], columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="테스트케이스")
    return bio.getvalue()

# [FIX] 결과 미리보기(휴리스틱) - Tab1: 코드 ZIP 분석 확장 (모듈/디렉터리 집계 포함)
def analyze_code_zip(zip_bytes: bytes) -> dict:
    lang_map = {
        ".py": "Python", ".java": "Java", ".js": "JS", ".ts": "TS",
        ".cpp": "CPP", ".c": "C", ".cs": "CS"
    }
    lang_counts = Counter()
    top_functions = []
    total_files = 0
    # [ADD] 모듈(상위 디렉터리) 집계
    module_counts = Counter()
    sample_paths = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = zf.namelist()
            total_files = len(names)
            sample_paths = names[:10]
            for n in names:
                # 모듈명 = 최상위 디렉터리, 없으면 '(root)'
                parts = n.split("/")
                module = parts[0] if len(parts) > 1 else "(root)"
                if not n.endswith("/"):  # 디렉터리 엔트리 제외
                    module_counts[module] += 1

                ext = os.path.splitext(n)[1].lower()
                if ext in lang_map:
                    lang_counts[lang_map[ext]] += 1
                    # 간단 함수/메서드 시그니처 추출(상위 20KB만)
                    try:
                        with zf.open(n) as fh:
                            content = fh.read(20480).decode("utf-8", errors="ignore")
                            for pat in [
                                r"def\s+([a-zA-Z_]\w*)\s*\(",
                                r"function\s+([a-zA-Z_]\w*)\s*\(",
                                r"(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z_<>\[\]]+\s+([a-zA-Z_]\w*)\s*\("
                            ]:
                                top_functions += re.findall(pat, content)
                    except Exception:
                        pass
    except zipfile.BadZipFile:
        pass

    return {
        "total_files": total_files,
        "lang_counts": lang_counts,
        "top_functions": top_functions[:50],   # 상한
        "module_counts": module_counts,        # [ADD]
        "sample_paths": sample_paths           # [ADD]
    }

# [ADD] 예상 테스트케이스 개수 추정(간단 휴리스틱)
def estimate_tc_count(stats: dict) -> int:
    files = max(0, stats.get("total_files", 0))
    langs = sum(stats.get("lang_counts", Counter()).values())
    funcs = len(stats.get("top_functions", []))
    estimate = int(files * 0.3 + langs * 0.7 + funcs * 0.9)
    return max(3, min(estimate, 300))  # 최소 3건, 최대 300건 제한

# [ADD] NEW: 함수명 → TC ID 생성 유틸 (실제와 유사한 도메인형 ID)
def _split_words(name: str) -> list[str]:
    """[ADD] 카멜/스네이크/기타 구분자 → 토큰 리스트"""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)  # camelCase 분리
    s = s.replace("_", " ")
    return [w for w in re.findall(r"[A-Za-z]+", s) if w]

def _abbr(word: str) -> str:
    """[ADD] 도메인 용어 약어화"""
    m = {
        "manager": "Mgr", "management": "Mgmt",
        "controller": "Ctrl", "service": "Svc",
        "repository": "Repo", "configuration": "Config",
        "request": "Req", "response": "Resp",
        "application": "App", "message": "Msg",
        "database": "DB", "client": "Clnt", "server": "Srv"
    }
    return m.get(word.lower(), word.capitalize())

# [FIX] 넘버링 부여: TC-<Base>-### 형식으로 생성되도록 수정
def make_tc_id_from_fn(fn: str, used_ids: set, seq: int | None = None) -> str:
    """
    [FIX] 함수명에서 불용어 제거 → 핵심 키워드 2~3개 → PascalCase/약어화 → 'TC-<Base>-###' 형식
         - seq가 주어지면 해당 넘버를 3자리로 부여
         - seq가 없으면 1부터 증가시키며 중복 없는 번호를 자동 할당
    """
    stop = {
        "get","set","is","has","have","do","make","build","create","update","insert","delete","remove","fetch","load","read","write",
        "put","post","patch","calc","compute","process","handle","run","exec","call","check","validate","convert","parse","format",
        "test","temp","main","init","start","stop","open","close","send","receive","retry","download","upload","save","add","sum","plus","div","divide"
    }
    words = _split_words(fn)
    core = [w for w in words if w.lower() not in stop]
    if not core:
        core = words[:2]  # 불용어만 있는 경우 앞 2개 사용
    core = core[:3]      # 최대 3개 결합
    base = "".join(_abbr(w) for w in core)
    base = re.sub(r"[^A-Za-z0-9]", "", base)[:24] or re.sub(r"[^A-Za-z0-9]", "", fn.title())[:16] or "Auto"

    # 넘버링 로직
    if seq is not None:
        n = seq
    else:
        n = 1
        # 사용중인 같은 base의 최대 번호보다 큰 수 찾기
        pattern = re.compile(rf"^TC-{re.escape(base)}-(\d{{3}})$")
        for uid in used_ids:
            m = pattern.match(uid)
            if m:
                n = max(n, int(m.group(1)) + 1)

    tcid = f"TC-{base}-{n:03d}"
    while tcid in used_ids:
        n += 1
        tcid = f"TC-{base}-{n:03d}"
    used_ids.add(tcid)
    return tcid

# [FIX] NEW: "함수명 분석 기반" 샘플 TC 생성기 (중복 방지 + 2~3건 가변 + 디테일 강화 + 도메인형 TC ID 넘버링)
def build_function_based_sample_tc(top_functions: list[str]) -> pd.DataFrame:
    """
    [FIX] 요구사항 반영:
      - 1) distinct kind 기반 2~3건
      - 2) 입력/예상결과 디테일 템플릿
      - 3) TC ID: TC-<키워드>-### 형식으로 넘버링 부여
      - ※ LLM 생성 TC ID에는 영향 없음 (본 함수는 Auto-Preview 전용)
    """
    rows = []
    used_kinds = set()
    used_ids = set()  # TC ID 중복 방지

    def priority(kind: str) -> str:
        high = {"div", "auth", "write", "delete", "io", "validate"}
        return "High" if kind in high else "Medium"

    def templates_for_kind(kind: str, fn: str):
        fn_disp = fn
        if kind == "add":
            return [
                (f"{fn_disp} 정상 합산", "a=10, b=20 (정상값)", "30 반환"),
                (f"{fn_disp} 합산 경계값", "a=-1, b=1 (음수+양수)", "오버플로우/언더플로우 없이 0 반환")
            ]
        if kind == "div":
            return [
                (f"{fn_disp} 정상 나눗셈", "a=6, b=3 (정상값)", "2 반환(정수/실수 처리 일관)"),
                (f"{fn_disp} 0 나눗셈 예외", "a=1, b=0 (비정상)", "ZeroDivisionError 또는 400/예외 코드")
            ]
        if kind == "read":
            return [
                (f"{fn_disp} 유효 조회", "id=1 (존재)", "정상 데이터 반환(HTTP 200/OK)"),
                (f"{fn_disp} 미존재 조회", "id=999999 (미존재)", "404/빈 결과 반환")
            ]
        if kind == "write":
            return [
                (f"{fn_disp} 유효 쓰기", "payload={'name':'A','value':1}", "201/성공 및 영속 반영"),
                (f"{fn_disp} 필수값 누락", "payload={'value':1} (name 누락)", "400/검증 오류 메시지")
            ]
        if kind == "delete":
            return [
                (f"{fn_disp} 유효 삭제", "id=1 (존재)", "삭제 성공 및 재조회 시 미존재"),
                (f"{fn_disp} 중복/미존재 삭제", "id=999999 (미존재)", "404 또는 멱등 처리")
            ]
        if kind == "auth":
            return [
                (f"{fn_disp} 유효 토큰 접근", "Bearer 유효토큰", "200/권한 허용"),
                (f"{fn_disp} 만료/위조 토큰", "Bearer 만료/위조 토큰", "401/403 접근 거부")
            ]
        if kind == "validate":
            return [
                (f"{fn_disp} 이메일 유효성(정상)", "s='user@example.com'", "True/허용"),
                (f"{fn_disp} 이메일 유효성(이상)", "s='invalid@domain'", "False/422 또는 검증 실패")
            ]
        if kind == "io":
            return [
                (f"{fn_disp} 업로드/다운로드 성공", "파일=1MB, timeout=5s", "성공/정상 응답, 무결성 유지"),
                (f"{fn_disp} 네트워크 타임아웃", "timeout=1s (지연 환경)", "재시도 or 타임아웃 오류 처리")
            ]
        return [
            (f"{fn_disp} 기본 정상 동작", "표준 입력 1세트(정상)", "성공 코드/정상 반환"),
            (f"{fn_disp} 비정상 입력 처리", "필수값 누락 또는 타입 불일치", "명확한 오류 메시지/코드 반환")
        ]

    def classify(fn: str) -> str:
        s = fn.lower()
        if any(k in s for k in ["add", "sum", "plus"]): return "add"
        if any(k in s for k in ["div", "divide"]): return "div"
        if any(k in s for k in ["get", "fetch", "load", "read"]): return "read"
        if any(k in s for k in ["save", "create", "update", "insert", "post", "put"]): return "write"
        if any(k in s for k in ["delete", "remove"]): return "delete"
        if any(k in s for k in ["auth", "login", "signin", "verify", "token"]): return "auth"
        if any(k in s for k in ["email", "validate", "regex", "check"]): return "validate"
        if any(k in s for k in ["upload", "download", "request", "client", "socket"]): return "io"
        return "default"

    # ➊ distinct kind 기준으로 최대 3건 수집 (TC ID에 넘버링 부여)
    candidates = []
    seq_counter = 1  # [FIX] TC ID 넘버링 시작
    for fn in top_functions:
        kind = classify(fn)
        if kind in used_kinds:
            continue
        used_kinds.add(kind)
        title, inp, exp = templates_for_kind(kind, fn)[0]
        tcid = make_tc_id_from_fn(fn, used_ids, seq=seq_counter)  # [FIX] TC-<Base>-### 부여
        seq_counter += 1
        candidates.append([kind, fn, tcid, title, inp, exp, priority(kind)])
        if len(candidates) >= 3:
            break

    # ➋ 결과 구성 (2~3건 보장, 서로 다른 케이스, 넘버링 지속)
    result = []
    if len(candidates) >= 3:
        for c in candidates[:3]:
            kind, fn, tcid, title, inp, exp, pr = c
            result.append([tcid, title, inp, exp, pr])
    elif len(candidates) == 2:
        for c in candidates:
            kind, fn, tcid, title, inp, exp, pr = c
            result.append([tcid, title, inp, exp, pr])
    elif len(candidates) == 1:
        kind, fn, _, _, _, _, pr = candidates[0]
        t_list = templates_for_kind(kind, fn)
        # 두 개 템플릿을 서로 다른 ID로 (넘버링 이어서)
        for (title, inp, exp) in t_list[:2]:
            tcid = make_tc_id_from_fn(fn, used_ids, seq=seq_counter)  # [FIX] 같은 base에 다른 ### 부여
            seq_counter += 1
            result.append([tcid, title, inp, exp, pr])
    else:
        # 함수가 전혀 없는 경우: 기본 2건 (서로 다른 ID, 넘버링 부여)
        tcid1 = make_tc_id_from_fn("Bootstrap_Init", used_ids, seq=1)
        tcid2 = make_tc_id_from_fn("CorePath_Error", used_ids, seq=2)
        result = [
            [tcid1, "엔트리포인트 기본 부팅 검증", "기본 실행 플로우", "에러 없이 초기 화면/상태 도달", "Medium"],
            [tcid2, "핵심 경로 예외 처리 검증", "유효하지 않은 입력(타입 불일치/누락)", "명확한 오류 메시지/코드 반환", "High"],
        ]

    return pd.DataFrame(result, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])

# [ADD] (기존 함수: 언어/모듈까지 반영하던 휴리스틱) — 유지하되, 현재는 사용하지 않음
def build_preview_testcases(stats: dict) -> pd.DataFrame:
    rows = []
    total_files = stats.get("total_files", 0)
    lang_counts: Counter = stats.get("lang_counts", Counter())
    top_functions = stats.get("top_functions", [])
    module_counts: Counter = stats.get("module_counts", Counter())
    if lang_counts:
        lang_str = ", ".join([f"{k} {v}개" for k, v in lang_counts.most_common()])
        rows.append(["TC-PV-LANG", f"언어분포 기반 초기 로딩/파싱 검증 ({lang_str})", "초기 로딩", f"파일 파싱 성공({total_files}개)", "Medium"])
    if top_functions:
        fn = top_functions[0]
        rows.append(["TC-PV-FUNC", f"핵심 함수/엔드포인트 동작 검증({fn})", "경계·무효 포함 2세트", "정상/에러 구분", "High"])
    rows.append(["TC-PV-COV", "모듈 커버리지 초기 점검", f"파일 수={total_files}", f"모듈 수={len(module_counts)}", "Medium"])
    return pd.DataFrame(rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])

# [ADD] 결과 미리보기(휴리스틱) - Tab2/Tab3용 보조 함수(요구상 미사용)
def build_preview_spec(df: pd.DataFrame, summary_type: str) -> str:
    titles = []
    if "기능 설명" in df.columns:
        titles = list(pd.Series(df["기능 설명"]).dropna().astype(str).head(3).unique())
    elif "TC ID" in df.columns:
        titles = [f"{summary_type} 기반: {str(df['TC ID'].iloc[i])}" for i in range(min(3, len(df)))]
    if not titles:
        titles = [f"{summary_type} 초안 항목"]
    lines = []
    for t in titles:
        lines.append(f"- **{t}**\n  - 설명: 입력/예상결과를 기준으로 동작 목적과 예외처리를 요약합니다.\n  - 기대 효과: 기능 명확화, 경계값 확인, 회귀 테스트 기반 확보.")
    return "\n".join(lines)

def build_preview_scenario(raw_log: str) -> str:
    sev_hits = re.findall(r"(ERROR|Exception|WARN|FATAL)", raw_log, flags=re.IGNORECASE)
    sev_stat = Counter([s.upper() for s in sev_hits])
    top = sev_stat.most_common(1)[0][0] if sev_stat else "INFO"
    return (
        "1. 시나리오 제목: 초기 재현 시도 (로그 패턴 기반)\n"
        f"2. 전제 조건: 로그 심각도 분포 {dict(sev_stat)}\n"
        "3. 테스트 입력값: 최소 재현 입력(최근 에러 직전 단계)\n"
        "4. 재현 절차: 에러 유발 직전 흐름 추적 → 동일 환경/버전에서 단계 수행\n"
        f"5. 기대 결과: {top} 레벨 이벤트 재현 및 추가 진단 정보 확보"
    )

# ────────────────────────────────────────────────
# 🧪 TAB 1: 소스코드 → 테스트케이스 자동 생성기
# ────────────────────────────────────────────────
with code_tab:
    st.subheader("🧪 소스코드 기반 테스트케이스 자동 생성기")

    # (유지) 샘플 테스트케이스 엑셀 버튼 없음. 샘플 코드 ZIP만 제공.
    st.download_button(
        "⬇️ 샘플 코드 ZIP 다운로드",
        data=build_sample_code_zip(),
        file_name="sample_code.zip",
        help="간단한 Python 함수/검증 로직 3파일 포함"
    )

    uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드",
                                     type=["zip"],
                                     key="code_zip")

    def need_llm_call(uploaded_file, model, role):
        return uploaded_file and (st.session_state.last_uploaded_file
                                  != uploaded_file.name
                                  or st.session_state.last_model != model
                                  or st.session_state.last_role != role)

    qa_role = st.session_state.get("qa_role", "기능 QA")

    # (유지) 요약 블록
    code_bytes = None
    if uploaded_file:
        code_bytes = uploaded_file.getvalue()
        stats = analyze_code_zip(code_bytes)

        with st.expander("📊 Auto-Preview(요약)", expanded=True):
            if stats["lang_counts"]:
                lang_str = ", ".join([f"{k} {v}개" for k, v in stats["lang_counts"].most_common()])
            else:
                lang_str = "감지된 언어 없음"
            funcs_cnt = len(stats["top_functions"])
            expected_tc = estimate_tc_count(stats)
            st.markdown(
                f"- **파일 수**: {stats['total_files']}\n"
                f"- **언어 분포**: {lang_str}\n"
                f"- **함수/엔드포인트 수(추정)**: {funcs_cnt}\n"
                f"- **예상 테스트케이스 개수(추정)**: {expected_tc}"
            )

        # (유지) 라벨: Auto-Preview(Sample TC) / 생성 로직: 함수명 분석 기반
        with st.expander("🔮 Auto-Preview(Sample TC)", expanded=True):
            sample_df = build_function_based_sample_tc(stats.get("top_functions", []))
            st.dataframe(sample_df, use_container_width=True)

    if uploaded_file and need_llm_call(uploaded_file, model, qa_role):
        st.session_state["is_loading"] = True
        with st.spinner("🔍 LLM 호출 중입니다. 잠시만 기다려 주세요..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(code_bytes if code_bytes is not None else uploaded_file.read())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                full_code = ""
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith((".py", ".java", ".js", ".ts", ".cpp",
                                          ".c", ".cs")):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path,
                                          "r",
                                          encoding="utf-8",
                                          errors="ignore") as f:
                                    code = f.read()
                                    full_code += f"\n\n# FILE: {file}\n{code}"
                            except:
                                continue
            # ⚠️ [중요] LLM 프롬프트/출력 파싱은 변경하지 않음 → LLM 생성 TC ID는 기존과 동일 동작
            prompt = f"""
너는 시니어 QA 엔지니어이며, 현재 '{qa_role}' 역할을 맡고 있다.
아래에 제공된 소스코드를 분석하여 기능 단위의 테스트 시나리오 기반 테스트케이스를 생성하라.

📌 출력 형식은 아래 마크다운 테이블 형태로 작성하되,
우선순위는 반드시 High / Medium / Low 중 하나로 작성할 것:

| TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
|-------|-----------|--------|------------|---------|

소스코드:
{full_code}
"""
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                })
            result = response.json()["choices"][0]["message"]["content"]
            st.session_state.llm_result = result
            rows = []
            for line in result.splitlines():
                if "|" in line and "TC" in line:
                    parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                    if len(parts) == 5:
                        rows.append(parts)
            if rows:
                df = pd.DataFrame(
                    rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])
                st.session_state.parsed_df = df
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.last_model = model
            st.session_state.last_role = qa_role
        st.session_state["is_loading"] = False

    if st.session_state.llm_result:
        st.success("✅ 테스트케이스 생성 완료!")
        st.markdown("## 📋 생성된 테스트케이스")
        st.markdown(st.session_state.llm_result)

    if st.session_state.parsed_df is not None and not need_llm_call(
            uploaded_file, model, qa_role):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            st.session_state.parsed_df.to_excel(tmp.name, index=False)
            tmp.seek(0)
            st.download_button("⬇️ 엑셀 다운로드",
                               data=tmp.read(),
                               file_name="테스트케이스.xlsx")


# ────────────────────────────────────────────────
# 📑 TAB 2: 테스트케이스 → 명세서 요약
# ────────────────────────────────────────────────
with tc_tab:
    st.subheader("📑 테스트케이스 기반 기능/요구사항 명세서 추출기")

    # (요구사항 유지) Tab2에 샘플 테스트케이스 엑셀 다운로드 제공
    st.download_button(
        "⬇️ 샘플 테스트케이스 엑셀 다운로드",
        data=build_sample_tc_excel(),
        file_name="테스트케이스_샘플.xlsx",
        help="필수 컬럼( TC ID, 기능 설명, 입력값, 예상 결과, 우선순위 ) 포함"
    )

    tc_file = st.file_uploader("📂 테스트케이스 파일 업로드 (.xlsx, .csv)",
                               type=["xlsx", "csv"],
                               key="tc_file")
    summary_type = st.selectbox("📌 요약 유형", ["기능 명세서", "요구사항 정의서"],
                                key="summary_type")

    if st.button("🚀 명세서 생성하기", disabled=st.session_state["is_loading"]) and tc_file:
        st.session_state["is_loading"] = True

        # (요구사항) Tab2는 휴리스틱 미리보기 제외 — 기존 로직 유지
        try:
            if tc_file.name.endswith("csv"):
                df = pd.read_csv(tc_file)
            else:
                df = pd.read_excel(tc_file)
        except Exception as e:
            st.session_state["is_loading"] = False
            st.error(f"❌ 파일 읽기 실패: {e}")
            st.stop()

        with st.spinner("🔍 LLM 호출 중입니다. 잠시만 기다려 주세요..."):
            required_cols = ["TC ID", "기능 설명", "입력값", "예상 결과"]
            if not all(col in df.columns for col in required_cols):
                st.session_state["is_loading"] = False
                st.warning("⚠️ 다음 컬럼이 필요합니다: TC ID, 기능 설명, 입력값, 예상 결과")
                st.stop()

            prompt = f"""
너는 테스트케이스를 분석하여 그 기반이 되는 {summary_type}를 작성하는 QA 전문가이다.
다음 테스트케이스들을 분석하여 기능명 또는 요구사항 제목과 함께, 설명과 목적을 자연어로 요약하라.

형식:
- 기능명 또는 요구사항 제목
- 설명
- 기대 효과

테스트케이스 목록:
{df.to_csv(index=False)}
"""
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                })
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                st.session_state.spec_result = result
            else:
                st.error("❌ LLM 호출 실패")
                st.text(response.text)

        st.session_state["is_loading"] = False

    if st.session_state.spec_result:
        st.success("✅ 명세서 생성 완료!")
        st.markdown("## 📋 자동 생성된 명세서")
        st.markdown(st.session_state.spec_result)
        st.download_button("⬇️ 명세서 텍스트 다운로드",
                           data=st.session_state.spec_result,
                           file_name="기능_요구사항_명세서.txt")


# ────────────────────────────────────────────────
# 🐞 TAB 3: 에러 로그 → 재현 시나리오 생성기
# ────────────────────────────────────────────────
with log_tab:
    st.subheader("🐞 에러 로그 기반 재현 시나리오 생성기")

    # ✅ 샘플 에러 로그 다운로드 버튼 (기존 유지)
    sample_log = """[InstallShield Silent]
    Version=v7.00
    File=Log File
    [ResponseResult]
    ResultCode=0
    [Application]
    Name=Realtek Audio Driver
    Version=4.92
    Company=Realtek Semiconductor Corp.
    Lang=0412
    """

    st.download_button(
        "⬇️ 샘플 에러 로그 다운로드",
        data=sample_log,
        file_name="sample_error_log.log",
        disabled=st.session_state["is_loading"]
    )

    log_file = st.file_uploader("📂 에러 로그 파일 업로드 (.log, .txt)",
                                type=["log", "txt"],
                                key="log_file")
    if not API_KEY:
        st.warning("🔐 OpenRouter API Key가 설정되지 않았습니다.")

    # (요구사항) Tab3는 휴리스틱 미리보기 제외 — 기존 로직 유지
    raw_log_cache = None
    if log_file:
        raw_log_cache = log_file.read().decode("utf-8", errors="ignore")

    if st.button("🚀 시나리오 생성하기", disabled=st.session_state["is_loading"]) and raw_log_cache:
        st.session_state["is_loading"] = True
        with st.spinner("LLM을 호출 중입니다..."):
            qa_role = st.session_state.get("qa_role", "기능 QA")
            chosen_model = model
            budget = safe_char_budget(chosen_model, token_margin=1024)
            focused_log, stats = preprocess_log_text(
                raw_log_cache,
                context_lines=5,
                keep_last_lines_if_empty=2000,
                char_budget=budget)
            st.info(
                f"전처리 결과: 문자 {stats['kept_chars']:,}/{stats['char_budget']:,} 사용 (전체 라인 {stats['total_lines']:,})."
            )
            st.markdown("**전처리 스니펫 (상위 120줄):**")
            st.code("\n".join(focused_log.splitlines()[:120]), language="text")

            prompt = f"""너는 시니어 QA 엔지니어이며, 현재 '{qa_role}' 역할을 맡고 있다.
아래 요약·발췌한 로그를 분석하여 해당 오류를 재현할 수 있는 테스트 시나리오를 작성하라.

시나리오 형식:
1. 시나리오 제목:
2. 전제 조건:
3. 테스트 입력값:
4. 재현 절차:
5. 기대 결과:

전처리된 에러 로그:
{focused_log}
"""
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": chosen_model,
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }],
                        "temperature": 0.2
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    st.session_state.scenario_result = content
                else:
                    st.error("❌ LLM 호출 실패")
                    st.caption("서버 응답:")
                    st.text(response.text)
            except requests.exceptions.RequestException as e:
                st.error("❌ 네트워크 오류 발생")
                st.exception(e)
        st.session_state["is_loading"] = False

    if st.session_state.scenario_result:
        st.success("✅ 재현 시나리오 생성 완료!")
        st.markdown("## 📋 자동 생성된 테스트 시나리오")
        st.markdown(st.session_state.scenario_result)
        st.download_button("⬇️ 시나리오 텍스트 다운로드",
                           data=st.session_state.scenario_result,
                           file_name="재현_시나리오.txt")
