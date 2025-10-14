import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests
import re
# [ADD] 미리보기/샘플 ZIP 생성을 위한 in-memory 버퍼
import io

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

# [FIX] 미리보기 전용 탭 제거(요구사항2) → 기존 3개 탭만 유지
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
# [ADD] 샘플 코드 ZIP 생성(기존 흐름 영향 없음)
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

# ────────────────────────────────────────────────
# 🧪 TAB 1: 소스코드 → 테스트케이스 자동 생성기
# ────────────────────────────────────────────────
with code_tab:
    st.subheader("🧪 소스코드 기반 테스트케이스 자동 생성기")

    # [FIX] 요구1: 샘플 테스트케이스 엑셀 버튼 제거, 샘플 코드 ZIP만 유지
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

    # [ADD] 요구2: LLM 실행 전 미리보기(해당 탭에서 표시)
    code_bytes = None
    if uploaded_file:
        code_bytes = uploaded_file.getvalue()  # 이후 LLM 처리에서도 재사용
        try:
            with zipfile.ZipFile(io.BytesIO(code_bytes), "r") as zf:
                file_list = zf.namelist()
                src_list = [f for f in file_list if f.endswith((".py",".java",".js",".ts",".cpp",".c",".cs"))]
                with st.expander("👀 업로드 ZIP 미리보기", expanded=True):
                    st.write(f"- 파일 수: **{len(file_list)}**  ·  소스 코드 파일 수: **{len(src_list)}**")
                    if src_list:
                        st.write("샘플(상위 5개 경로):")
                        st.code("\n".join(src_list[:5]), language="bash")
                        sel = src_list[0]
                        with zf.open(sel) as fh:
                            snippet = fh.read().decode("utf-8", errors="ignore")
                            st.markdown(f"**스니펫:** `{sel}` (상위 80줄)")
                            st.code("\n".join(snippet.splitlines()[:80]) or "(빈 파일)", language="python")
                    else:
                        st.warning("소스 코드 확장자(.py/.java/.js/.ts/.cpp/.c/.cs)가 감지되지 않았습니다.")
        except zipfile.BadZipFile:
            st.error("ZIP 형식이 올바르지 않습니다.")

    if uploaded_file and need_llm_call(uploaded_file, model, qa_role):
        st.session_state["is_loading"] = True
        with st.spinner("🔍 LLM 호출 중입니다. 잠시만 기다려 주세요..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, uploaded_file.name)
                # [FIX] 미리보기에서 읽은 바이트 재사용(업로드 객체 재읽기 방지)
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

    # [FIX] 요구1에 따라 Tab1만 버튼 제거, Tab2는 기존대로 동작(추가 버튼 없음)

    tc_file = st.file_uploader("📂 테스트케이스 파일 업로드 (.xlsx, .csv)",
                               type=["xlsx", "csv"],
                               key="tc_file")
    summary_type = st.selectbox("📌 요약 유형", ["기능 명세서", "요구사항 정의서"],
                                key="summary_type")

    if st.button("🚀 명세서 생성하기", disabled=st.session_state["is_loading"]) and tc_file:
        st.session_state["is_loading"] = True

        # [ADD] 요구2: LLM 실행 전 미리보기(해당 탭에서 표시)
        try:
            if tc_file.name.endswith("csv"):
                df = pd.read_csv(tc_file)
            else:
                df = pd.read_excel(tc_file)
        except Exception as e:
            st.session_state["is_loading"] = False
            st.error(f"❌ 파일 읽기 실패: {e}")
            st.stop()

        with st.expander("👀 테스트케이스 미리보기", expanded=True):
            st.write("행/열:", df.shape)
            st.dataframe(df.head(20))
            required_cols = ["TC ID", "기능 설명", "입력값", "예상 결과"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.warning("필수 컬럼 누락: " + ", ".join(missing))
            else:
                st.success("필수 컬럼 확인 완료")
            # 간단 분석
            if "TC ID" in df.columns:
                dup_cnt = df["TC ID"].duplicated().sum()
                if dup_cnt:
                    st.warning(f"중복된 TC ID {dup_cnt}건 감지")
            if "우선순위" in df.columns:
                dist = df["우선순위"].value_counts(dropna=False).to_dict()
                st.info("우선순위 분포: " + ", ".join([f"{k}:{v}" for k,v in dist.items()]))

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

    # [ADD] 요구2: LLM 실행 전 로그 미리보기(해당 탭에서 표시)
    raw_log_cache = None
    if log_file:
        raw_log_cache = log_file.read().decode("utf-8", errors="ignore")
        with st.expander("👀 로그 미리보기", expanded=True):
            st.write(f"- 총 문자 수: **{len(raw_log_cache):,}**")
            patt = re.compile(r"(ERROR|Exception|WARN|FATAL)", re.IGNORECASE)
            hits = len(patt.findall(raw_log_cache))
            st.info(f"심각도 키워드 감지 개수: {hits}")
            # 원본 상위 80줄
            st.markdown("**원본 스니펫 (상위 80줄):**")
            st.code("\n".join(raw_log_cache.splitlines()[:80]) or "(빈 파일)", language="text")

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
