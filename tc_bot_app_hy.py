import streamlit as st
import os, zipfile, tempfile, pandas as pd, requests, io, re, time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ✅ OpenRouter API KEY (하드코딩 + 안전가드)
_raw_key = "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"
API_KEY = (_raw_key or "").strip()  # ← 숨은 공백/개행 제거가 핵심

st.set_page_config(page_title="TC-Bot v3", layout="wide")
st.title("🧪 TC-Bot v3: 테스트케이스 자동 생성기")

# ─────────────────────────────────────────────
# (중략) — build_sample_project_zip(), 미리보기 유틸 등 기존 소스2 본문은 그대로 두세요
# ─────────────────────────────────────────────

# 🔗 OpenRouter 헤더 — 소스1과 동일한 '최소 헤더'만 사용
def openrouter_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        # requests의 json= 사용 시 Content-Type 자동 설정됨
    }

# 🔍 프리플라이트: 키/연결 진단(선택) — 사이드에 토글로 붙이면 편함
with st.sidebar:
    if st.checkbox("🔎 OpenRouter 프리플라이트 실행", value=False):
        try:
            r = requests.get("https://openrouter.ai/api/v1/models",
                             headers=openrouter_headers(), timeout=15)
            st.write("프리플라이트 /v1/models 상태:", r.status_code)
            if r.status_code == 200:
                st.success("✅ 키 유효 · 통신 정상")
            else:
                st.error("❌ 프리플라이트 실패")
                st.code(r.text)
        except Exception as e:
            st.error(f"연결 오류: {e}")

# ✅ 사이드바 입력 — 소스1과 동일 alias 사용
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox("🤖 사용할 LLM 모델", ["qwen/qwen-max", "mistral"])
    role = st.selectbox("👤 QA 역할", ["기능 QA", "보안 QA", "성능 QA"])

# ✅ 세션 초기화 … (소스2 그대로 유지)
# uploaded_file/need_llm_call 등 기존 로직 그대로 유지

# … (중략: 미리보기, 프롬프트 구성 등 기존 소스2 그대로)

# 5) LLM 호출 — 소스1과 동일한 방식으로 최소 헤더/바디 전송
def call_openrouter(model: str, prompt: str, timeout=60):
    if not API_KEY or not API_KEY.startswith("sk-or-v1-"):
        raise RuntimeError("API_KEY가 비어있거나 형식이 잘못되었습니다. (예: sk-or-v1-...)")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    # 중요: allow_redirects=False 로 리다이렉트 시 쿠키/헤더 변형 방지(보수적)
    return requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=openrouter_headers(),
        json=payload,
        timeout=timeout,
        allow_redirects=False,
    )

# … (중략) 프롬프트 준비 완료 후 ↓ 아래처럼 호출 교체
# stage_bar.progress(85, text="LLM 생성 중…")
# status_box.warning("🤖 LLM이 테스트케이스를 생성 중입니다. 잠시만 기다려 주세요…")
# try/except 구조도 그대로 두되, 함수 사용

# 예: (기존 response = requests.post(...)) 부분을 아래로 교체
try:
    response = call_openrouter(model, prompt, timeout=60)
    if response.status_code != 200:
        st.error(f"LLM 호출 실패: HTTP {response.status_code}")
        st.code(response.text)  # 원문 바디 그대로
        response.raise_for_status()
except requests.RequestException as e:
    st.error(f"LLM 호출 실패: {e}")
    response = None

# 이후 파싱/표시 로직은 소스2 그대로 유지
