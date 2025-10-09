import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import requests
import io
from datetime import datetime

# ✅ OpenRouter API KEY (하드코딩 사용)
API_KEY = "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"

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
        b = float(request.args.get("b", 0))
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
        help="예제 소스(zip)를 내려받아 바로 업로드 테스트에 사용하세요."
    )

# ─────────────────────────────────────────────
# ✅ 사이드바 입력
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    # ⚠️ OpenRouter에서
