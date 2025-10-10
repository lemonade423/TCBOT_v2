import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests
import io
import re
import time
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# 🔐 API Key 정규화 / 지문
# ─────────────────────────────────────────────
def normalize_api_key(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"[\u200B\u200C\u200D\u2060\ufeff]", "", raw)  # 제로폭 제거
    raw = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", raw)  # 유니코드 대시 정규화
    raw = re.sub(r"\s+", "", raw)  # 공백/개행 제거
    return raw.strip()

def fingerprint(s: str) -> str:
    if not s:
        return "(empty)"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]
    head = s[:4] if len(s) >= 4 else s
    tail = s[-4:] if len(s) >= 4 else s
    return f"{head}…{tail} | sha256:{h}"

# ✅ OpenRouter API KEY (하드코딩 + 정규화)
_raw_key = "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"
API_KEY = normalize_api_key(_raw_key)

st.set_page_config(page_title="TC-Bot v3", layout="wide")
st.title("🧪 TC-Bot v3: 테스트케이스 자동 생성기")

# ─────────────────────────────────────────────
# 🧩 샘플코드 ZIP 생성 유틸
# ─────────────────────────────────────────────
def build_sample_project_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
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
        key="dl_sample_zip",
        help="예제 소스(zip)를 내려받아 바로 업로드 테스트에 사용하세요."
    )

# ─────────────────────────────────────────────
# 🔗 헤더 빌더 (서버/브라우저 키 지원)
# ─────────────────────────────────────────────
def headers_server_only():
    return {"Authorization": f"Bearer {API_KEY}"}

def headers_browser_mode(referer: str, title: str = "TC-Bot v3"):
    return {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": referer,
        "X-Title": title
    }

# ─────────────────────────────────────────────
# 🔎 프리플라이트 + 키 지문(사이드바)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 키/연결 프리플라이트")
    st.caption("키 지문 (앞/뒤 4자리 + sha256-10)")
    st.code(fingerprint(API_KEY))
    st.caption(
        "Prefix OK: "
        + ("✅" if API_KEY.startswith("sk-or-v1-") else "❌")
        + "  |  Contains space: "
        + ("❌" if " " in API_KEY else "✅")
    )
    referer_input = st.text_input(
        "HTTP-Referer (도메인)",
        value="http://localhost:8501",
        key="http_referer_input"
    )
    if st.checkbox("프리플라이트 실행(/v1/models)", value=False, key="prefetch_models"):
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers_server_only(),
                timeout=15
            )
            st.write("프리플라이트 상태:", r.status_code)
            if r.status_code == 200:
                st.success("✅ 키 유효 · 네트워크 정상")
            else:
                st.error("❌ 프리플라이트 실패")
                st.code(r.text)
        except Exception as e:
            st.error(f"연결 오류: {e}")

# ─────────────────────────────────────────────
# ✅ 사이드바 입력 (고유 key 부여)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox(
        "🤖 사용할 LLM 모델",
        ["qwen/qwen-max", "mistral"],
        key="model_select"
    )
    role = st.selectbox(
        "👤 QA 역할",
        ["기능 QA", "보안 QA", "성능 QA"],
        key="role_select"
    )

# ✅ 세션 초기화
session_defaults = {
    "last_uploaded_file": None,
    "last_model": None,
    "last_role": None,
    "llm_result": None,
    "parsed_df": None,
    "preview_df": None,
    "preview_stats": None,
}
for k, v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 📂 업로더 — 여기 ‘한 번만’ 호출 & 고유 key 지정
uploaded_file = st.file_uploader(
    "📂 소스코드 zip 파일 업로드",
    type=["zip"],
    key="zip_uploader"  # ← 중복 방지
)

def need_llm_call(uploaded_file, model, role):
    return (
        uploaded_file is not None
        and (
            st.session_state.last_uploaded_file != uploaded_file.name
            or st.session_state.last_model != model
            or st.session_state.last_role != role
        )
    )

# ─────────────────────────────────────────────
# 🔎 코드 분석(미리보기) 유틸
# ─────────────────────────────────────────────
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
            funcs += re.findall(r"@app\.(?:get|post|put|delete|patch)\(['\"]/([^\)'\"]+)", text)
        elif file_path.suffix == ".java":
            funcs += re.findall(r"(?:public|private|protected)\s+[<>\w\[\]]+\s+([a-zA-Z_]\w*)\s*\(", text)
        elif file_path.suffix in [".js", ".ts"]:
            funcs += re.findall(r"function\s+([a-zA-Z_]\w*)\s*\(", text)
            funcs += re.findall(r"export\s+function\s+([a-zA-Z_]\w*)\s*\(", text)
    except Exception:
        pass
    seen = set()
    uniq = []
    for f in funcs:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq[:10]

def analyze_source_tree(root_dir: str, role: str):
    exts = []
    file_list = []
    functions = []
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
    rows = []
    lang_str = ", ".join([f"{k} {v}개" for k, v in stats["lang_counts"].most_common()])
    rows.append(["TC-PV-001", "언어 혼합 프로젝트 로딩", f"언어분포: {lang_str}", "모든 파일 파싱 성공", "High"])
    if stats["top_functions"]:
        fn = stats["top_functions"][0]
        rows.append(["TC-PV-002", f"핵심 함수/엔드포인트 동작 검증({fn})", "유효/무효 입력 2세트", "정상/에러 응답 구분", "High"])
    else:
        rows.append(["TC-PV-002", "엔드포인트/함수 미검출 시 기본 동작", "기본 실행", "에러 없이 앱 부팅", "Medium"])
    rows.append(["TC-PV-003", "대상 코드 범위 커버리지 초기 점검", f"파일 수={stats['total_files']}", "주요 모듈별 1개 이상 케이스 존재", "Medium"])
    df = pd.DataFrame(rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"])
    return df

# ─────────────────────────────────────────────
# 🔗 OpenRouter 호출 (401이면 Browser 헤더로 자동 재시도)
# ─────────────────────────────────────────────
def call_openrouter(model: str, prompt: str, referer_for_retry: str, timeout=60):
    if not API_KEY or not API_KEY.startswith("sk-or-v1-"):
        raise RuntimeError("API_KEY 형식 오류 (예: sk-or-v1-...)")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}

    # 1차: 서버키 스타일 (Authorization만)
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers_server_only(),
        json=payload,
        timeout=timeout,
        allow_redirects=True,
    )
    if resp.status_code != 401:
        return resp

    # 2차: 401이면 Browser 키로 간주하고 재시도 (Referer/X-Title 포함)
    resp2 = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers_browser_mode(referer_for_retry, title="TC-Bot v3"),
        json=payload,
        timeout=timeout,
        allow_redirects=True,
    )
    return resp2

# ─────────────────────────────────────────────
# ✅ LLM 호출 파이프라인 + Auto-Flow Preview
# ─────────────────────────────────────────────
if uploaded_file and need_llm_call(uploaded_file, model, role):
    if not API_KEY:
        st.error("🔑 OpenRouter API Key가 비어 있습니다.")
    else:
        st.markdown("### 🔎 Auto-Flow Preview")
        c1, c2, c3, c4 = st.columns(4)
        status_box = st.empty()
        stage_bar = st.progress(0, text="준비 중…")
        preview_placeholder = st.empty()

        stage_bar.progress(10, text="코드 파싱 준비 중…")
        status_box.info("⏳ 업로드 파일을 임시 폴더에 추출합니다.")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            stage_bar.progress(20, text="압축 해제 중…")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            time.sleep(0.2)

            stage_bar.progress(40, text="언어/파일/함수 특징 추출…")
            status_box.info("🔍 언어 비율, 파일 개수, 함수/엔드포인트를 분석합니다.")
            stats = analyze_source_tree(tmpdir, role)
            st.session_state.preview_stats = stats

            c1.metric("파일 수", f"{stats['total_files']}개")
            lang_top = stats["lang_counts"].most_common(1)[0][0] if stats["lang_counts"] else "-"
            c2.metric("주요 언어", lang_top)
            c3.metric("예상 TC 수", stats["estimated_cases"])
            c4.metric("감지된 함수/엔드포인트", f"{len(stats['top_functions'])}개")

            stage_bar.progress(60, text="미리보기 테스트케이스 생성…")
            st.session_state.preview_df = build_preview_testcases(stats)
            with preview_placeholder.container():
                st.caption("※ 아래 미리보기는 휴리스틱 기반으로 생성됩니다. 최종 결과는 LLM 생성 후 갱신됩니다.")
                st.dataframe(st.session_state.preview_df, use_container_width=True)

            stage_bar.progress(75, text="프롬프트 구성…")
            status_box.info("🧠 LLM 프롬프트를 구성합니다.")
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

        stage_bar.progress(85, text="LLM 생성 중…")
        status_box.warning("🤖 LLM이 테스트케이스를 생성 중입니다. 잠시만 기다려 주세요…")
        try:
            response = call_openrouter(model, prompt, referer_for_retry=referer_input, timeout=60)
            if response.status_code != 200:
                st.error(f"LLM 호출 실패: HTTP {response.status_code}")
                try:
                    st.code(response.text, language="json")
                except Exception:
                    pass
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

            rows = []
            for line in result.splitlines():
                if "|" in line and "TC" in line:
                    parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                    if len(parts) == 5:
                        rows.append(parts)

            if rows:
                st.session_state.parsed_df = pd.DataFrame(
                    rows, columns=["TC ID", "기능 설명", "입력값", "예상 결과", "우선순위"]
                )

            stage_bar.progress(100, text="완료")
            status_box.success("✅ 테스트케이스 생성 완료!")
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
            key="dl_excel"
        )

# ✅ (언제든) 미리보기 보관 영역 표시
if st.session_state.preview_df is not None and st.session_state.parsed_df is None:
    st.markdown("### 👀 미리보기(휴리스틱)")
    st.dataframe(st.session_state.preview_df, use_container_width=True)
    if st.session_state.preview_stats:
        with st.expander("📊 분석 요약(미리보기)"):
            s = st.session_state.preview_stats
            st.write("- 파일 수:", s["total_files"])
            st.write("- 언어 분포:", dict(s["lang_counts"]))
            st.write("- 감지된 함수/엔드포인트:", s["top_functions"])
            st.write("- 예상 테스트케이스 수:", s["estimated_cases"])

# ✅ 세션 상태 업데이트 (마지막에)
if uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file.name
    st.session_state.last_model = model
    st.session_state.last_role = role
