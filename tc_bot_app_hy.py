import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests

# --- [추가 import: 샘플 ZIP/프리뷰에만 사용, 기존 흐름에 영향 없음] ---
import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
# -----------------------------------------------------------------------

# ✅ OpenRouter API KEY (보안 주의!)
API_KEY = "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"

st.set_page_config(page_title="TC-Bot v3", layout="wide")
st.title("🧪 TC-Bot v3: 테스트케이스 자동 생성기")

# =========================
# 추가 기능 1) 📦 샘플코드 ZIP 다운로드
# =========================
def build_sample_project_zip() -> bytes:
    """
    업로드 없이도 파서/미리보기 흐름을 시험 가능한
    Python/Java/JavaScript 혼합 샘플 프로젝트 ZIP을 in-memory로 생성
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
        b = float(request.args.get("b", "0"))
        return jsonify({"result": a + b})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
'''
        )
        z.writestr("sample_project_py/requirements.txt", "flask==3.0.3\n")

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

with st.container():
    st.subheader("📦 샘플코드 다운로드 (업로드 없이 바로 테스트)")
    st.caption("파이썬/자바/자바스크립트 혼합 예제 포함 · 파서/미리보기 흐름 검증에 적합")
    sample_zip_bytes = build_sample_project_zip()
    st.download_button(
        "⬇️ 샘플코드 .zip 다운로드",
        data=sample_zip_bytes,
        file_name="tc-bot-sample-code.zip",
        mime="application/zip",
        help="예제 소스(zip)를 내려받아 바로 업로드 테스트에 사용하세요.",
        key="dl_sample_zip",
    )

# ✅ 사이드바 입력
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

# --- [추가: Auto-Flow Preview용 상태만 별도 key로 보관] ---
st.session_state.setdefault("preview_stats", None)
st.session_state.setdefault("preview_df", None)
# ----------------------------------------------------------------

uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드", type=["zip"])

def need_llm_call(uploaded_file, model, role):
    # 이전 세션 상태와 비교 (기존 로직 유지)
    return (uploaded_file is not None
            and (st.session_state.last_uploaded_file != uploaded_file.name
                 or st.session_state.last_model != model
                 or st.session_state.last_role != role))

# =========================
# 추가 기능 2) 🔎 Auto-Flow Preview
#  - 업로드 ZIP을 LLM 호출 전에 빠르게 스캔하여
#    (파일 수/언어/함수·엔드포인트) 요약 + 휴리스틱 미리보기 3건
# =========================
LANG_EXT = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".cpp": "C++",
    ".c": "C",
    ".cs": "C#",
}

def extract_functions(file_path: Path, text: str):
    funcs = []
    try:
        if file_path.suffix == ".py":
            funcs += re.findall(r"def\s+([a-zA-Z_]\w*)\s*\(", text)
            # Flask/FastAPI 엔드포인트 감지
            funcs += re.findall(r"@app\.(?:get|post|put|delete|patch)\(['\"]/([^\)'\"]+)", text)
        elif file_path.suffix == ".java":
            funcs += re.findall(r"(?:public|private|protected)\s+[<>\w\[\]]+\s+([a-zA-Z_]\w*)\s*\(", text)
        elif file_path.suffix in [".js", ".ts"]:
            funcs += re.findall(r"function\s+([a-zA-Z_]\w*)\s*\(", text)
            funcs += re.findall(r"export\s+function\s+([a-zA-Z_]\w*)\s*\(", text)
    except Exception:
        pass
    # 중복 제거, 최대 10개
    seen, uniq = set(), []
    for f in funcs:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return uniq[:10]

def analyze_source_tree(root_dir: str, role: str):
    exts, file_list, functions = [], [], []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            p = Path(r) / fn
            ext = p.suffix.lower()
            if ext in LANG_EXT:
                file_list.append(str(p))
                exts.append(ext)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                    functions.extend([f"{Path(p).name}:{n}" for n in extract_functions(p, txt)])
                except Exception:
                    continue
    lang_counts = Counter(LANG_EXT[e] for e in exts)
    total_files = len(file_list)
    weight = {"기능 QA": 1.2, "보안 QA": 1.1, "성능 QA": 1.0}.get(role, 1.0)
    estimated_cases = max(5, int(len(functions) * 1.5 * weight))
    return {
        "total_files": total_files,
        "lang_counts": lang_counts,
        "top_functions": functions[:10],
        "estimated_cases": estimated_cases
    }

def build_preview_testcases(stats):
    # 휴리스틱 기반 미리보기 3건
    rows = []
    lang_str = ", ".join([f"{k} {v}개" for k, v in stats["lang_counts"].most_common()])
    rows.append(["TC-PV-001", "언어 혼합 프로젝트 로딩", f"언어분포: {lang_str}", "모든 파일 파싱 성공", "High"])
    if stats["top_functions"]:
        fn = stats["top_functions"][0]
        rows.append(["TC-PV-002", f"핵심 함수/엔드포인트 동작 검증({fn})", "유효/무효 입력 2세트", "정상/에러 응답 구분", "High"])
    else:
        rows.append(["TC-PV-002", "엔드포인트/함수 미검출 시 기본 동작", "기본 실행", "에러 없이 앱 부팅", "Medium"])
    rows.append(["TC-PV-003", "대상 코드 범위 커버리지 초기 점검", f"파일 수={stats['total_files']}", "주요 모듈별 1개 이상 케이스 존재", "Medium"])
    return pd.DataFrame(rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])

# 업로드되면, LLM 호출 조건과 상관없이 "미리보기"만 먼저 수행 (기존 로직에 영향 없음)
if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir_preview:
        try:
            # 업로드 ZIP 임시 저장/추출
            zip_path = os.path.join(tmpdir_preview, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir_preview)

            # 분석 & 미리보기 표 생성
            stats = analyze_source_tree(tmpdir_preview, role)
            st.session_state.preview_stats = stats
            st.session_state.preview_df = build_preview_testcases(stats)

            # UI 표시
            with st.expander("🔎 Auto-Flow Preview (LLM 호출 전 빠른 요약)", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("파일 수", f"{stats['total_files']}개")
                lang_top = stats["lang_counts"].most_common(1)[0][0] if stats["lang_counts"] else "-"
                c2.metric("주요 언어", lang_top)
                c3.metric("예상 TC 수", stats["estimated_cases"])
                c4.metric("함수/엔드포인트 감지", f"{len(stats['top_functions'])}개")
                st.caption("※ 아래 미리보기는 휴리스틱 기반입니다. 최종 결과는 LLM 생성 후 갱신됩니다.")
                st.dataframe(st.session_state.preview_df, use_container_width=True)

        except Exception as e:
            # 미리보기 실패해도 LLM 본 흐름은 그대로 진행 가능하도록 경고만 표기
            st.warning(f"Auto-Flow Preview 중 경고: {e}")

# =========================
# (아래부터는 소스1의 기존 로직 그대로)
# =========================

# ✅ LLM 호출 조건 확인
if uploaded_file and need_llm_call(uploaded_file, model, role):
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
                    if file.endswith(
                        (".py", ".java", ".js", ".ts", ".cpp", ".c", ".cs")):
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

        # ✅ Prompt 구성
        prompt = f"""
        너는 시니어 QA 엔지니어이며, 현재 '{role}' 역할을 맡고 있다.
        아래에 제공된 소스코드를 분석하여 기능 단위의 테스트 시나리오 기반 테스트케이스를 생성하라.

        📌 출력 형식은 아래 마크다운 테이블 형태로 작성하되,
        우선순위는 반드시 High / Medium / Low 중 하나로 작성할 것:

        | TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
        |-------|-----------|--------|------------|---------|

        소스코드:
        {full_code}
        """

        # ✅ LLM 호출
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

        # ✅ 결과 파싱
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
        st.download_button("⬇️ 엑셀 다운로드",
                           data=tmp.read(),
                           file_name="테스트케이스.xlsx")
