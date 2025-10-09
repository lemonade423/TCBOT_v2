import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests
import io
from datetime import datetime

# ✅ OpenRouter API KEY (보안 주의!)
API_KEY = "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"

st.set_page_config(page_title="TC-Bot v3", layout="wide")
st.title("🧪 TC-Bot v3: 테스트케이스 자동 생성기")

# ─────────────────────────────────────────────
# 🧩 샘플코드 ZIP 생성 유틸
# ─────────────────────────────────────────────
def build_sample_project_zip() -> bytes:
    """
    업로드 없이도 파서/LLM 파이프라인을 곧장 시험할 수 있는
    다언어( .py / .java / .js ) 샘플 소스를 ZIP(in-memory)으로 생성.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # Python 샘플
        z.writestr(
            "sample_project_py/app.py",
            '''"""
샘플 파이썬 서비스
- /health 엔드포인트: 상태 확인
- /sum?a=1&b=2 합계 계산
"""
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/sum")
def sum_api():
    try:
        a = float(request.args.get("a", 0))
        b = float(request.args.get("b", 0))
        return jsonify({"result": a + b})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
'''
        )
        z.writestr(
            "sample_project_py/requirements.txt",
            "flask==3.0.3\n"
        )

        # Java 샘플
        z.writestr(
            "sample_project_java/src/main/java/com/example/CalcService.java",
            '''package com.example;

public class CalcService {
    public int add(int a, int b) { return a + b; }
    public int sub(int a, int b) { return a - b; }
    public boolean isEven(int n) { return n % 2 == 0; }
}
'''
        )
        z.writestr(
            "sample_project_java/README.md",
            "# Java 샘플\n- 간단한 사칙연산/짝수판별 메소드 포함"
        )

        # JS 샘플
        z.writestr(
            "sample_project_js/index.js",
            '''// 간단한 입력 검증 + 합계
export function sum(a, b) {
  if (typeof a !== "number" || typeof b !== "number") {
    throw new Error("Invalid input");
  }
  return a + b;
}
'''
        )
        z.writestr(
            "sample_project_js/package.json",
            '''{
  "name": "sample-project-js",
  "version": "1.0.0",
  "type": "module",
  "main": "index.js"
}
'''
        )

        # 안내 문서
        z.writestr(
            "README.md",
            f"""# TC-Bot 샘플 코드 번들
업로드 없이도 테스트케이스 생성을 바로 시험할 수 있도록 만든 예제 소스입니다.
- Python(Flask) / Java / JavaScript 예제 포함
- 파서 검증용으로 다양한 확장자/디렉토리 구조 제공

생성 시각: {datetime.now().isoformat(timespec='seconds')}
"""
        )
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# 📦 샘플코드 다운로드 UI
# ─────────────────────────────────────────────
with st.container():
    st.subheader("📦 샘플코드 다운로드 (업로드 없이 바로 테스트)")
    st.caption("파이썬/자바/자바스크립트 혼합 예제 포함 · 파서/테이블 변환 테스트에 적합")
    sample_zip_bytes = build_sample_project_zip()
    st.download_button(
        "⬇️ 샘플코드 .zip 다운로드",
        data=sample_zip_bytes,
        file_name="tc-bot-sample-code.zip",
        mime="application/zip",
        help="예제 소스(zip)를 내려받아 바로 업로드 테스트에 사용하세요."
    )

# ─────────────────────────────────────────────
# ✅ 사이드바 입력
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox("🤖 사용할 LLM 모델", ["qwen/qwen-max", "mistral"])
    role = st.selectbox("👤 QA 역할", ["기능 QA", "보안 QA", "성능 QA"])

# ✅ 세션 초기화
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "last_role" not in st.session_state:
    st.session_state.last_role = None
if "llm_result" not in st.session_state:
    st.session_state.llm_result = None
if "parsed_df" not in st.session_state:
    st.session_state.parsed_df = None

uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드", type=["zip"])

def need_llm_call(uploaded_file, model, role):
    # 이전 세션 상태와 비교
    return (
        uploaded_file is not None
        and (
            st.session_state.last_uploaded_file != uploaded_file.name
            or st.session_state.last_model != model
            or st.session_state.last_role != role
        )
    )

# ✅ LLM 호출 조건 확인
if uploaded_file and need_llm_call(uploaded_file, model, role):
    if not API_KEY:
        st.error("🔑 OpenRouter API Key가 설정되지 않았습니다. secrets 혹은 환경변수를 확인하세요.")
    else:
        with st.spinner("🔍 LLM 호출 중입니다. 잠시만 기다려 주세요..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                full_code = ""
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith((".py", ".java", ".js", ".ts", ".cpp", ".c", ".cs")):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    code = f.read()
                                    rel = os.path.relpath(file_path, tmpdir)
                                    full_code += f"\n\n# FILE: {rel}\n{code}"
                            except Exception:
                                continue

            # ✅ Prompt 구성
            prompt = f"""
너는 시니어 QA 엔지니어이며, 현재 '{role}' 역할을 맡고 있다.
아래에 제공된 소스코드를 분석하여 기능 단위의 테스트 시나리오 기반 테스트케이스를 생성하라.

📌 출력 형식은 아래 마크다운 테이블 형태로 작성하되,
우선순위는 반드시 High / Medium / Low 중 하나로 작성할 것:

| TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
|-------|-----------|--------|-----------|----------|

소스코드:
{full_code}
"""

            # ✅ LLM 호출
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=60,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                st.error(f"LLM 호출 실패: {e}")
                response = None

            if response is not None:
                try:
                    result = response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    st.error(f"응답 파싱 실패: {e}")
                    result = ""

                st.session_state.llm_result = result

                # ✅ 결과 파싱
                rows = []
                for line in result.splitlines():
                    if "|" in line and "TC" in line:
                        parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                        if len(parts) == 5:
                            rows.append(parts)

                if rows:
                    df = pd.DataFrame(
                        rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"]
                    )
                    st.session_state.parsed_df = df

                # ✅ 세션 상태 업데이트
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.last_model = model
                st.session_state.last_role = role

# ✅ 결과 렌더링
if st.session_state.llm_result:
    st.success("✅ 테스트케이스 생성 완료!")
    st.markdown("## 📋 생성된 테스트케이스")
    st.markdown(st.session_state.llm_result)

# ✅ 엑셀 다운로드
if st.session_state.parsed_df is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        st.session_state.parsed_df.to_excel(tmp.name, index=False)
        tmp.seek(0)
        st.download_button(
            "⬇️ 엑셀 다운로드",
            data=tmp.read(),
            file_name="테스트케이스.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
