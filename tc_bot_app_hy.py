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

# [FIX] 결과 미리보기(휴리스틱) 표를 업로드 코드 기반으로 "동적 생성"
def build_preview_testcases(stats: dict) -> pd.DataFrame:
    rows = []

    # 동적 값 계산
    total_files = stats.get("total_files", 0)
    lang_counts: Counter = stats.get("lang_counts", Counter())
    top_functions = stats.get("top_functions", [])
    module_counts: Counter = stats.get("module_counts", Counter())

    # 1) 언어 기반 로딩 테스트 (동적 설명/결과/우선순위/ID)
    if lang_counts:
        lang_str = ", ".join([f"{k} {v}개" for k, v in lang_counts.most_common()])
        mixed = len(lang_counts) > 1
        prio1 = "High" if mixed or total_files >= 20 else "Medium"
        tcid1 = f"TC-PV-LANG-{len(lang_counts)}L-{total_files}F"
        desc1 = f"언어분포 기반 초기 로딩/파싱 검증 ({lang_str})"
        expect1 = f"주요 언어별 파서 초기화 및 파일 파싱 성공({total_files}개)"
    else:
        prio1 = "Medium"
        tcid1 = f"TC-PV-LANG-0L-{total_files}F"
        desc1 = "단일언어/언어 미검출 프로젝트 로딩"
        expect1 = f"기본 파서로 전체 파일({total_files}개) 파싱 성공"

    rows.append([tcid1, desc1, "초기 로딩", expect1, prio1])

    # 2) 핵심 함수/엔드포인트 검증 (탐지된 함수명 활용)
    if top_functions:
        fn = top_functions[0]
        prio2 = "High" if len(top_functions) >= 5 else "Medium"
        tcid2 = f"TC-PV-FUNC-{fn[:12]}"
        desc2 = f"핵심 함수/엔드포인트 동작 검증({fn})"
        expect2 = "유효 입력 → 정상 반환 / 무효 입력 → 명확한 예외/에러 코드"
        input2 = "경계·무효 포함 2세트"
    else:
        prio2 = "Medium"
        tcid2 = "TC-PV-FUNC-NONE"
        desc2 = "엔드포인트/함수 미검출 시 기본 동작"
        expect2 = "기본 실행 경로에서 에러 없이 부팅 및 핵심 화면 진입"
        input2 = "기본 실행"

    rows.append([tcid2, desc2, input2, expect2, prio2])

    # 3) 모듈(디렉터리) 커버리지 초기 점검 (모듈명/개수 반영)
    if module_counts:
        top_mods = [m for m, _ in module_counts.most_common(3)]
        mod_str = ", ".join(top_mods)
        prio3 = "High" if len(module_counts) >= 5 else "Medium"
        tcid3 = f"TC-PV-COV-{len(module_counts)}M"
        desc3 = f"모듈 커버리지 초기 점검 (주요: {mod_str})"
        expect3 = f"각 모듈별 최소 1개 케이스 존재(모듈 수={len(module_counts)})"
    else:
        prio3 = "Medium"
        tcid3 = "TC-PV-COV-0M"
        desc3 = "모듈 구조 미검출 시 기본 커버리지 점검"
        expect3 = "파일 단위로 최소 1개 케이스 매핑"

    rows.append([tcid3, desc3, f"파일 수={total_files}", expect3, prio3])

    return pd.DataFrame(rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])

# [ADD] 결과 미리보기(휴리스틱) - Tab2: (요구에 따라 제거) → 함수/호출 남겨두지 않음
def build_preview_spec(df: pd.DataFrame, summary_type: str) -> str:
    # (남겨두지만 사용하지 않음) — 요구사항에 따라 Tab2 미리보기 호출 제거
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

# [ADD] 결과 미리보기(휴리스틱) - Tab3: (요구에 따라 제거) → 함수/호출 남겨두지 않음
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

    # [FIX] 강화된 결과 미리보기: "요약 + 3건 프리뷰" 및 텍스트 라벨 변경
    code_bytes = None
    if uploaded_file:
        code_bytes = uploaded_file.getvalue()
        stats = analyze_code_zip(code_bytes)

        # ── [FIX] 라벨 변경: "결과 요약(휴리스틱)" → "Auto-Preview(요약)"
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

        # ── [FIX] 라벨 변경: "결과 미리보기(휴리스틱: 테스트케이스 3건)" → "Auto-Preview(TC 예상)"
        with st.expander("🔮 Auto-Preview(TC 예상)", expanded=True):
            pv_df = build_preview_testcases(stats)
            st.dataframe(pv_df, use_container_width=True)

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

        # [FIX] (요구사항1) Tab2의 휴리스틱 미리보기 제거 — LLM 호출 전 프리뷰 표시 없음
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

    # [FIX] (요구사항1) Tab3의 휴리스틱 미리보기 제거 — 업로드 즉시 프리뷰 표시 없음
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
