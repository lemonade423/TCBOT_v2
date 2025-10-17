import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests
import re
# [ADD] 유틸/미리보기/엑셀용
import io
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path

# ✅ OpenRouter API Key (보안을 위해 secrets.toml 또는 환경변수 사용 권장)
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_KEY")

if not API_KEY:
    st.warning(
        "⚠️ OpenRouter API Key가 설정되지 않았습니다. .streamlit/secrets.toml에 OPENROUTER_API_KEY 항목을 추가하세요."
    )

st.set_page_config(page_title="🧠 TC-Bot: QA 자동화 도우미", layout="wide")
st.title("🤖 TC-Bot: AI 기반 QA 자동화 도우미")

# ✅ 세션 초기화
for key in ["scenario_result", "spec_result", "llm_result", "parsed_df", "last_uploaded_file", "last_model", "last_role", "is_loading"]:
    if key not in st.session_state:
        st.session_state[key] = None

# [ADD] 기능별 그룹 보관 + 정규화 원문 보관 + 기능힌트 보관
if "parsed_groups" not in st.session_state:
    st.session_state["parsed_groups"] = None
if "normalized_markdown" not in st.session_state:
    st.session_state["normalized_markdown"] = None
if "feature_hints" not in st.session_state:
    st.session_state["feature_hints"] = None

if st.session_state["is_loading"] is None:
    st.session_state["is_loading"] = False


# ✅ 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox("🤖 사용할 LLM 모델", ["qwen/qwen-max", "mistral"])
    qa_role = st.selectbox("👤 QA 역할", ["기능 QA", "보안 QA", "성능 QA"])
    st.session_state["qa_role"] = qa_role

# ✅ 탭 구성
code_tab, tc_tab, log_tab = st.tabs(
    ["🧪 소스코드 → 테스트케이스 자동 생성", "📑 테스트케이스 → 명세서 요약", "🐞 에러 로그 → 재현 시나리오"]
)

# ✅ LLM 호출 중 경고 표시
if st.session_state["is_loading"]:
    st.warning("⚠️ 현재 LLM 호출 중입니다. 잠시만 기다려 주세요.")
else:
    st.empty()

# ────────────────────────────────────────────────
# [ADD] 샘플코드 및 테스트케이스 미리보기
# ────────────────────────────────────────────────
def build_sample_code_zip() -> bytes:
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

# ────────────────────────────────────────────────
# [FIX] NEW: "함수명 분석 기반" 샘플 TC 생성기
# ────────────────────────────────────────────────
def make_tc_id_from_fn(fn: str, used_ids: set, seq: int | None = None) -> str:
    stop = {"get","set","is","has","have","do","make","build","create","update","insert","delete","remove","fetch","load","read","write",
            "put","post","patch","calc","compute","process","handle","run","exec","call","check","validate","convert","parse","format",
            "test","temp","main","init","start","stop","open","close","send","receive","retry","download","upload","save","add","sum","plus","div","divide"}
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn).replace("_"," ")
    words = [w for w in re.findall(r"[A-Za-z]+", s) if w.lower() not in stop] or re.findall(r"[A-Za-z]+", s)[:2]
    base = "".join(w.capitalize() for w in words[:3])
    base = re.sub(r"[^A-Za-z0-9]", "", base) or "Auto"
    n = 1 if seq is None else seq
    tcid = f"TC-{base}-{n:03d}"
    while tcid in used_ids:
        n += 1
        tcid = f"TC-{base}-{n:03d}"
    used_ids.add(tcid)
    return tcid

def build_function_based_sample_tc(top_functions: list[str]) -> pd.DataFrame:
    rows = []
    used_kinds = set()
    used_ids = set()

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

    candidates = []
    seq_counter = 1
    for fn in top_functions:
        kind = classify(fn)
        if kind in used_kinds:
            continue
        used_kinds.add(kind)
        title, inp, exp = templates_for_kind(kind, fn)[0]
        tcid = make_tc_id_from_fn(fn, used_ids, seq=seq_counter)
        seq_counter += 1
        candidates.append([kind, fn, tcid, title, inp, exp, priority(kind)])
        if len(candidates) >= 3:
            break

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
        for (title, inp, exp) in t_list[:2]:
            tcid = make_tc_id_from_fn(fn, used_ids, seq=seq_counter)
            seq_counter += 1
            result.append([tcid, title, inp, exp, pr])
    else:
        tcid1 = make_tc_id_from_fn("Bootstrap_Init", used_ids, seq=1)
        tcid2 = make_tc_id_from_fn("CorePath_Error", used_ids, seq=2)
        result = [
            [tcid1, "엔트리포인트 기본 부팅 검증", "기본 실행 플로우", "에러 없이 초기 화면/상태 도달", "Medium"],
            [tcid2, "핵심 경로 예외 처리 검증", "유효하지 않은 입력(타입 불일치/누락)", "명확한 오류 메시지/코드 반환", "High"],
        ]
    return pd.DataFrame(result, columns=["TC ID","기능 설명","입력값","예상 결과","우선순위"])

# ────────────────────────────────────────────────
# [FIX] TAB1: 소스코드 → 테스트케이스 자동 생성
# ────────────────────────────────────────────────
with code_tab:
    st.subheader("🧪 소스코드 기반 테스트케이스 자동 생성기")
    st.download_button("⬇️ 샘플 코드 ZIP 다운로드", data=build_sample_code_zip(), file_name="sample_code.zip")

    uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드", type=["zip"], key="code_zip")

    # [FIX] — LLM 호출 예시 프롬프트 수정
    prompt = f"""
너는 시니어 QA 엔지니어이며, 현재 '{qa_role}' 역할을 맡고 있다.
아래 소스코드를 분석하여 **기능별 섹션**으로 테스트케이스를 작성하라.

반드시 아래 형식을 지켜라:
- 각 기능은 "## 기능명" 헤딩으로 시작한다.
- 각 기능 섹션마다 **하나의 마크다운 테이블**만 포함한다.
- 테이블 컬럼: | TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
- **TC ID는 반드시 `tc-<feature-key>-NNN` 형식**을 사용하라. (예: `tc-alarm-001`)
  - 각 기능 섹션마다 NNN은 001부터 다시 시작한다.
- 기능 섹션 외의 설명은 자동 생성할 필요 없음 (테이블 아래 설명은 제외).
"""

    # [FIX] 결과 표시 — 검정색 문구 + 엑셀 다운로드 복원
    if st.session_state.llm_result:
        st.success("✅ 테스트케이스 생성 완료!")
        st.markdown("## 📋 생성된 테스트케이스")
        st.markdown(
            """
            <div style='color:black; font-size:0.9rem;'>
            아래는 제공된 소스코드를 분석하여 작성한 기능 단위 테스트 시나리오 기반 테스트케이스입니다.<br>
            각 기능 및 함수의 동작을 검증하기 위해 다양한 입력값과 조건을 고려하였으며,<br>
            우선순위를 High, Medium, 또는 Low로 지정했습니다.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.normalized_markdown or st.session_state.llm_result)

        # 엑셀 다운로드 버튼 복원
        if st.session_state.parsed_groups:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
                    for key, df in st.session_state.parsed_groups.items():
                        df.to_excel(writer, index=False, sheet_name=key[:31])
                tmp.seek(0)
                st.download_button(
                    "⬇️ 엑셀 다운로드",
                    data=tmp.read(),
                    file_name="테스트케이스.xlsx",
                    help="기능별 시트로 구분된 테스트케이스 엑셀 파일"
                )
