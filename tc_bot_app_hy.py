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
    "OPENROUTER_API_KEY"
)
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

# [ADD] 기능별 그룹 보관 + 정규화 원문 보관 + 기능힌트 보관
if "parsed_groups" not in st.session_state:
    st.session_state["parsed_groups"] = None
if "normalized_markdown" not in st.session_state:
    st.session_state["normalized_markdown"] = None
# [ADD] 업로드 코드에서 추출한 기능 힌트 저장 (후처리 분리에 활용)
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
code_tab , tc_tab, log_tab = st.tabs(
    ["🧪 소스코드 → 테스트케이스 자동 생성","📑 테스트케이스 → 명세서 요약","🐞 에러 로그 → 재현 시나리오"]
)

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

def preprocess_log_text(text: str, context_lines: int = 3, keep_last_lines_if_empty: int = 1500, char_budget: int = 120000) -> tuple[str, dict]:
    lines = text.splitlines()
    total_lines = len(lines)
    non_debug = [(i, line) for i, line in enumerate(lines) if "DEBUG" not in line]
    patt = re.compile(r"(ERROR|Exception|WARN|FATAL)", re.IGNORECASE)
    matched_indices = [i for i, line in non_debug if patt.search(line)]

    selected = set()
    if matched_indices:
        for mi in matched_indices:
            orig_idx = non_debug[mi][0]
            for j in range(max(0, orig_idx - context_lines), min(total_lines, orig_idx + context_lines + 1)):
                selected.add(j)
        focused = [lines[j] for j in sorted(selected)]
        header = [
            "### Log Focus (ERROR/WARN/Exception 중심 발췌)",
            f"- 전체 라인: {total_lines:,}",
            f"- 컨텍스트 포함 라인: {len(selected):,}",
            ""
        ]
        trimmed = "\n".join(header + focused)
    else:
        tail = lines[-keep_last_lines_if_empty:]
        header = [
            "### Log Tail (매치 없음 → 마지막 일부 사용)",
            f"- 전체 라인: {total_lines:,}",
            f"- 사용 라인(마지막): {len(tail):,}",
            ""
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
# [ADD] 샘플 파일/샘플 TC 엑셀 빌더 (기존 유지)
# ────────────────────────────────────────────────
def build_sample_code_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("app.py", "# FILE: app.py\n"
                    "def add(a, b):\n"
                    "    return a + b\n\n"
                    "def div(a, b):\n"
                    "    if b == 0:\n"
                    "        raise ZeroDivisionError('b must not be zero')\n"
                    "    return a / b\n")
        zf.writestr("utils/validator.py", "# FILE: utils/validator.py\n"
                    "def is_email(s: str) -> bool:\n"
                    "    return '@' in s and '.' in s.split('@')[-1]\n")
        zf.writestr("README.md", "# Sample Project\n\n"
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
# [ADD] 코드 ZIP 분석/프리뷰 유틸 (기존 로직 확장: 클래스/파일/디렉터리 → 기능힌트 추출)
# ────────────────────────────────────────────────
def _norm_key(s: str) -> str:
    s = re.sub(r"[^\w\-]+", " ", s)
    s = re.sub(r"[_\s]+", "-", s).strip("-").lower()
    return s or "general"

def _display_from_key(key: str) -> str:
    parts = [p for p in key.split("-") if p]
    return "".join(w.capitalize() for w in parts) or "General"

def analyze_code_zip(zip_bytes: bytes) -> dict:
    lang_map = {
        ".py": "Python",
        ".java": "Java",
        ".js": "JS",
        ".ts": "TS",
        ".cpp": "CPP",
        ".c": "C",
        ".cs": "CS"
    }
    lang_counts = Counter()
    top_functions = []
    classes = []
    total_files = 0
    module_counts = Counter()
    sample_paths = []
    feature_keys = set()  # [ADD] 기능 후보 키

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = zf.namelist()
            total_files = len(names)
            sample_paths = names[:10]

            for n in names:
                if n.endswith("/"):
                    first = n.strip("/").split("/")[0]
                    if first:
                        feature_keys.add(_norm_key(first))
                    continue
                parts = n.split("/")
                module = parts[0] if len(parts) > 1 else "(root)"
                module_counts[module] += 1

                ext = os.path.splitext(n)[1].lower()
                stem = os.path.splitext(os.path.basename(n))[0]
                if stem:
                    feature_keys.add(_norm_key(stem))

                if ext in lang_map:
                    lang_counts[lang_map[ext]] += 1

                try:
                    with zf.open(n) as fh:
                        content = fh.read(100_000).decode("utf-8", errors="ignore")
                        for pat in [
                            r"def\s+([a-zA-Z_]\w*)\s*\(",
                            r"function\s+([a-zA-Z_]\w*)\s*\(",
                            r"(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z_<>\[\]]+\s+([a-zA-Z_]\w*)\s*\("
                        ]:
                            top_functions += re.findall(pat, content)
                        for cpat in [
                            r"class\s+([A-Z][A-Za-z0-9_]*)",
                            r"(?:public|final|abstract)\s+class\s+([A-Z][A-Za-z0-9_]*)"
                        ]:
                            classes += re.findall(cpat, content)
                except Exception:
                    pass

        for name in classes[:80]:
            feature_keys.add(_norm_key(name))
        for fn in top_functions[:120]:
            feature_keys.add(_norm_key(fn))

    except zipfile.BadZipFile:
        pass

    generic = {"app","main","index","core","utils","common","service","controller","model","routes","handler","api","src","test","tests"}
    feature_keys = {k for k in feature_keys if k and k not in generic}

    return {
        "total_files": total_files,
        "lang_counts": lang_counts,
        "top_functions": top_functions[:200],
        "module_counts": module_counts,
        "sample_paths": sample_paths,
        "classes": classes[:200],
        "feature_keys": sorted(feature_keys)[:40],
    }

def estimate_tc_count(stats: dict) -> int:
    files = max(0, stats.get("total_files", 0))
    langs = sum(stats.get("lang_counts", Counter()).values())
    funcs = len(stats.get("top_functions", []))
    estimate = int(files * 0.3 + langs * 0.7 + funcs * 0.9)
    return max(3, min(estimate, 300))

# ────────────────────────────────────────────────
# [ADD] LLM 결과 파싱/정규화 유틸 확장 (기능 힌트 기반 강제 분리)
# ────────────────────────────────────────────────
def _strip_code_fences(md: str) -> str:
    return re.sub(r"```.*?```", "", md, flags=re.DOTALL)

def _parse_md_tables_with_heading(md_text: str) -> list[tuple[str, pd.DataFrame]]:
    text = _strip_code_fences(md_text)
    lines = text.splitlines()
    tables = []
    i = 0
    last_heading = None
    heading_line = -999

    while i < len(lines):
        line = lines[i].rstrip()

        m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if m:
            last_heading = m.group(1).strip()
            heading_line = i
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.search(r"\|\s*:?-{2,}\s*\|", lines[i + 1]):
            feature_name = last_heading if 0 <= (i - heading_line - 1) <= 3 else ""
            j = i + 2
            rows = [line, lines[i + 1]]
            while j < len(lines):
                cur = lines[j]
                if cur.strip() == "" or ("|" not in cur):
                    break
                rows.append(cur)
                j += 1
            df = _md_table_to_df("\n".join(rows))
            if df is not None and len(df.columns) >= 3:
                tables.append((feature_name, df))
            i = j
            continue

        i += 1

    return tables

def _md_table_to_df(table_str: str) -> pd.DataFrame | None:
    raw = [r for r in table_str.splitlines() if r.strip()]
    if len(raw) < 2:
        return None
    headers = [h.strip() for h in raw[0].strip("|").split("|")]
    data_lines = [r for r in raw[2:]]
    rows = []
    for line in data_lines:
        if "|" not in line:
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) != len(headers):
            continue
        rows.append(parts)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=headers)

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    header_map = {
        "TC ID":"TC ID","TCID":"TC ID","ID":"TC ID","케이스ID":"TC ID",
        "기능 설명":"기능 설명","기능설명":"기능 설명","Feature":"기능 설명","Description":"기능 설명","기능":"기능 설명",
        "입력값":"입력값","Input":"입력값","입력":"입력값","Parameters":"입력값",
        "예상 결과":"예상 결과","Expected":"예상 결과","Output":"예상 결과","기대 결과":"예상 결과","결과":"예상 결과",
        "우선순위":"우선순위","Priority":"우선순위","우선 순위":"우선순위"
    }
    new_cols = {}
    for c in df.columns:
        key = header_map.get(str(c).strip(), None)
        if key:
            new_cols[c] = key
    df2 = df.rename(columns=new_cols)
    for c in ["TC ID","기능 설명","입력값","예상 결과","우선순위"]:
        if c not in df2.columns:
            df2[c] = ""
    return df2[["TC ID","기능 설명","입력값","예상 결과","우선순위"]]

def _normalize_feature_key(name: str, sample_row: dict | None = None) -> tuple[str,str]:
    key = (name or "").strip()
    if not key and sample_row:
        tcid = str(sample_row.get("TC ID",""))
        feat = str(sample_row.get("기능 설명",""))
        m = re.match(r"(?i)TC[-_]?([A-Za-z0-9]+)", tcid)
        if m and m.group(1) and not m.group(1).isdigit():
            key = m.group(1)
        if not key:
            tks = re.findall(r"[A-Za-z][A-Za-z0-9]+", feat)
            if tks:
                key = "".join(tks[:2])
    key = key or "General"
    sheet = re.sub(r"[^A-Za-z0-9가-힣_ -]", "", key).strip()
    key_id = re.sub(r"[^A-Za-z0-9 ]", "", sheet).strip().lower().replace(" ", "-") or "general"
    return sheet, key_id

def _extract_prefix_from_tcid(tcid: str) -> str | None:
    m = re.match(r"(?i)^TC[-_]?([A-Za-z][A-Za-z0-9]+)-\d{2,4}$", str(tcid).strip())
    if m:
        return m.group(1).lower()
    return None

# [ADD] 기능 힌트(aliases) 생성
def build_feature_hints(stats: dict) -> dict:
    keys = stats.get("feature_keys", []) or []
    aliases = defaultdict(set)
    for k in keys:
        aliases[k].add(k)
        aliases[k].add(k.replace("-", ""))
    for name in (stats.get("classes") or []) + (stats.get("top_functions") or []):
        norm = _norm_key(name)
        if not norm:
            continue
        target = None
        for k in keys:
            if norm.startswith(k) or k.startswith(norm) or norm.replace("-","") in k.replace("-",""):
                target = k; break
        if target:
            aliases[target].update({norm, norm.replace("-", ""), name.lower()})
    return {k: sorted(v) for k, v in aliases.items()}

# [ADD] 힌트 기반 행→기능 키 추정
def _infer_key_from_row_with_hints(row: pd.Series, hints: dict) -> str:
    text = " ".join([str(row.get(c,"")) for c in ["TC ID","기능 설명","입력값","예상 결과"]]).lower()
    pref = _extract_prefix_from_tcid(str(row.get("TC ID","")))
    if pref:
        return pref
    best_key, best_hits = None, 0
    for key, toks in hints.items():
        hits = 0
        for t in toks:
            t2 = t.lower()
            if t2 and t2 in text:
                hits += 1
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key or "general"

def split_single_df_feature_aware(df: pd.DataFrame, hints: dict) -> dict[str, pd.DataFrame]:
    df2 = _normalize_headers(df).fillna("")
    df2["_k_"] = df2.apply(lambda r: _infer_key_from_row_with_hints(r, hints), axis=1)
    groups = {}
    for k, sub in df2.groupby("_k_"):
        sub = sub.drop(columns=["_k_"]).reset_index(drop=True)
        sheet, key_id = _normalize_feature_key(k, sub.iloc[0].to_dict() if len(sub) else None)
        sub["TC ID"] = [f"tc-{key_id}-{i:03d}" for i in range(1, len(sub)+1)]
        groups[sheet[:31] or "General"] = sub
    return groups

def group_tables_and_renumber(md_text: str) -> dict[str, pd.DataFrame]:
    tbls = _parse_md_tables_with_heading(md_text)
    if not tbls:
        return {}
    groups: dict[str, pd.DataFrame] = {}
    unnamed_count = 0
    for (heading, df) in tbls:
        df_norm = _normalize_headers(df).fillna("")
        sample_row = df_norm.iloc[0].to_dict() if len(df_norm) else {}
        sheet_name, key_id = _normalize_feature_key(heading, sample_row)
        if not heading:
            unnamed_count += 1
            sheet_name = f"{sheet_name}-{unnamed_count}"
        df_norm["TC ID"] = [f"tc-{key_id}-{i:03d}" for i in range(1, len(df_norm)+1)]
        final_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
        cnt = 2
        while final_name in groups:
            candidate = (sheet_name[:27] + f"-{cnt}") if len(sheet_name) > 27 else f"{sheet_name}-{cnt}"
            final_name = candidate[:31]
            cnt += 1
        groups[final_name] = df_norm
    return groups

def concat_groups_for_view(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame(columns=["기능","TC ID","기능 설명","입력값","예상 결과","우선순위"])
    view_rows = []
    for sheet, df in groups.items():
        df2 = df.copy()
        df2["기능"] = sheet
        view_rows.append(df2)
    return pd.concat(view_rows, ignore_index=True)[["기능","TC ID","기능 설명","입력값","예상 결과","우선순위"]]

def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = ["TC ID","기능 설명","입력값","예상 결과","우선순위"]
    use_cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(use_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(use_cols)) + " |"
    rows = []
    for _, r in df[use_cols].iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in use_cols) + " |")
    return "\n".join([header, sep] + rows)

# [FIX] 우선순위 정규화/추론
def _normalize_priority_token(v: str) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    mapping = {
        "1": "High", "h": "High", "high": "High", "높음": "High", "상": "High", "필수": "High", "critical": "High", "crit": "High",
        "2": "Medium", "m": "Medium", "med": "Medium", "medium": "Medium", "중간": "Medium", "보통": "Medium", "중": "Medium",
        "3": "Low", "l": "Low", "low": "Low", "낮음": "Low", "하": "Low", "optional": "Low", "옵션": "Low"
    }
    return mapping.get(s, "High" if "high" in s else ("Medium" if "med" in s else ("Low" if "low" in s else "")))

def _infer_priority_from_text(text: str) -> str:
    s = (text or "").lower()
    if any(k in s for k in ["zerodivision", "division by zero", "0으로", "error", "exception", "fatal", "권한", "unauthorized", "forbidden", "not found", "401", "403", "404", "timeout", "타임아웃", "invalid", "오류"]):
        return "High"
    if any(k in s for k in ["경계", "boundary", "edge", "최대", "최소", "음수", "소수", "float", "overflow", "underflow"]):
        return "Medium"
    return "Medium"

def _ensure_priorities(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "우선순위" not in df2.columns:
        df2["우선순위"] = ""
    norm_vals = []
    for _, row in df2.iterrows():
        raw = str(row.get("우선순위", "")).strip()
        merged_text = " ".join([str(row.get(c, "")) for c in ["기능 설명", "입력값", "예상 결과"]])
        norm = _normalize_priority_token(raw)
        if not norm:
            norm = _infer_priority_from_text(merged_text)
        norm_vals.append(norm)
    df2["우선순위"] = norm_vals
    return df2

def _priority_counts(df: pd.DataFrame) -> dict:
    df2 = _ensure_priorities(df)
    vals = df2["우선순위"].astype(str).str.strip().str.title().tolist()
    c = Counter(vals)
    return {"High": c.get("High", 0), "Medium": c.get("Medium", 0), "Low": c.get("Low", 0)}

# [ADD] 보조 추출기
def _extract_endpoints(text: str) -> list[str]:
    eps = set(re.findall(r"/[A-Za-z0-9_\-./]+", text))
    cleaned = sorted({e.strip().rstrip(".,)") for e in eps if len(e) <= 64})
    return cleaned[:5]

def _classify_scenario_bucket(s: str) -> str:
    s = s.lower()
    if any(k in s for k in ["오류", "error", "예외", "invalid", "0으로", "zero", "null", "timeout", "권한", "401", "403", "404"]):
        return "예외"
    if any(k in s for k in ["경계", "boundary", "최대", "최소", "음수", "소수", "edge", "limit"]):
        return "경계"
    return "정상"

# [ADD] 핵심: 이름+내용 기반 기능 설명 생성기 (처음 보는 함수도 소스/TC 텍스트로 추론)
def _feature_desc_from_name_and_content(name: str, merged_text: str) -> str:
    n = (name or "").lower()
    t = (merged_text or "").lower()

    # 공통 키워드
    has_json = any(k in t for k in ["json", "application/json", "{", "}", "직렬화", "serialize", "deserialize"])
    has_health = "/health" in t or "health" in n
    has_sum = any(k in (n + " " + t) for k in ["sum", "add", "덧셈", "합계", "합산"])
    has_sub = any(k in (n + " " + t) for k in ["sub", "subtract", "차감", "감산"])
    has_email = any(k in (n + " " + t) for k in ["email", "이메일"])
    has_file = any(k in (n + " " + t) for k in ["file", "파일"])
    has_write = any(k in (n + " " + t) for k in ["write", "쓰기", "저장"])
    has_read = any(k in (n + " " + t) for k in ["read", "읽기", "load"])
    has_encoding = any(k in t for k in ["euc-kr", "utf-8", "charset"])
    has_https = any(k in (n + " " + t) for k in ["httpsurlconnection", "https", "ssl", "tls"])
    has_stream = any(k in (n + " " + t) for k in ["bytearrayoutputstream", "inputstream", "stream"])
    has_alarm = any(k in (n + " " + t) for k in ["alarm", "알림"])
    has_exception = any(k in (n + " " + t) for k in ["exception", "에러", "오류", "sqlexception", "ioexception"])
    eps = _extract_endpoints(merged_text)

    # 1) 가장 특정한 것부터
    if has_health:
        return "헬스체크 엔드포인트의 가용성과 응답 정합성을 확인합니다."
    if has_alarm:
        return "알림(Alarm) 요청/호출을 수행하며 대상/시각/시퀀스 파라미터를 처리합니다."
    if has_https:
        return "지정된 URL과 HTTPS 연결을 열고 요청/응답을 처리합니다."
    if has_stream:
        return "입력 스트림에서 바이트를 읽어 메모리 버퍼에 기록/변환합니다."
    if "jsonconvert" in n or (has_json and ("convert" in n or "serialize" in t or "직렬" in t)):
        return "객체/데이터를 JSON으로 직렬화하여 응답하거나 역직렬화합니다."
    if has_file and has_write and has_encoding:
        return "문자열과 파일명을 받아 지정 인코딩으로 파일을 생성/작성합니다."
    if has_file and has_write:
        return "파일이 없으면 생성하고, 내용을 기록하여 저장합니다."
    if has_file and has_read:
        return "존재하는 파일을 열어 내용을 읽어 반환합니다."
    if has_email:
        return "문자열이 유효한 이메일 형식인지 검증합니다."
    if has_sum and has_json and eps:
        return "REST API로 두 수의 합을 계산해 JSON 형태로 반환합니다."
    if has_sum:
        return "두 수의 합을 계산해 결과를 반환합니다."
    if has_sub:
        return "두 수의 차를 계산해 결과를 반환합니다."
    if "iseven" in n or "짝수" in t:
        return "입력이 짝수인지 여부를 판별합니다."
    if has_exception:
        return "예외 발생 시 자원해제·로깅·오류 응답 등 예외 처리를 수행합니다."
    if eps:
        return f"{', '.join(eps)} 엔드포인트의 요청/응답 동작을 검증합니다."
    # 2) 기본값 (일반화)
    return f"‘{name}’ 기능의 핵심 동작을 검증합니다."

# [FIX] 실제로 화면에 넣을 동적 설명 마크다운 생성
#      (출력: 기능설명, 우선순위 분포, 요약 / 헤더: Feature (총 N건))
def build_dynamic_explanations(groups: dict[str, pd.DataFrame]) -> str:
    if not groups:
        return "_설명을 생성할 데이터가 없습니다._"

    parts = []
    for feature_name, df in groups.items():
        df_norm = _ensure_priorities(df)
        total = len(df_norm)

        # 그룹 전체 텍스트 수집
        merged_text = " ".join(
            df_norm[["기능 설명","입력값","예상 결과"]].astype(str).fillna("").values.ravel().tolist()
        )

        # [FIX] 이름+내용 기반 설명 (첫 행 복사 금지)
        func_desc = _feature_desc_from_name_and_content(feature_name, merged_text)

        # 우선순위 분포
        pr = _priority_counts(df_norm)

        # 버킷 기반 요약
        buckets = Counter()
        for _, row in df_norm.iterrows():
            s = " ".join([str(row.get(c,"")) for c in ["기능 설명","입력값","예상 결과"]])
            buckets[_classify_scenario_bucket(s)] += 1
        endpoints = _extract_endpoints(merged_text)

        parts.append(f"#### {feature_name} (총 {total}건)")
        parts.append(f"- **기능 설명**: {func_desc}")
        parts.append(f"- **우선순위 분포**: High {pr['High']} · Medium {pr['Medium']} · Low {pr['Low']}")

        summary_bits = []
        if buckets.get("예외", 0) > 0:
            summary_bits.append("예외 처리로 안정성 검증을 강화")
        if buckets.get("경계", 0) > 0:
            summary_bits.append("경계 입력을 포함해 견고성 확인")
        if endpoints:
            summary_bits.append("관련 엔드포인트 동작 일관성 확인")
        if not summary_bits:
            summary_bits.append("정상·경계 상황을 균형 있게 검증")
        parts.append(f"- **요약**: {', '.join(summary_bits)}.")
        parts.append("")

    return "\n".join(parts).strip()

# ────────────────────────────────────────────────
# 🧪 TAB 1: 소스코드 → 테스트케이스 자동 생성기
# ────────────────────────────────────────────────
with code_tab:
    st.subheader("🧪 소스코드 기반 테스트케이스 자동 생성기")

    st.download_button(
        "⬇️ 샘플 코드 ZIP 다운로드",
        data=build_sample_code_zip(),
        file_name="sample_code.zip",
        help="간단한 Python 함수/검증 로직 3파일 포함"
    )

    uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드", type=["zip"], key="code_zip")

    def need_llm_call(uploaded_file, model, role):
        return uploaded_file and (st.session_state.last_uploaded_file != uploaded_file.name or st.session_state.last_model != model or st.session_state.last_role != role)

    qa_role = st.session_state.get("qa_role", "기능 QA")

    code_bytes = None
    stats = {"total_files":0,"lang_counts":Counter(),"top_functions":[]}
    if uploaded_file:
        code_bytes = uploaded_file.getvalue()
        stats = analyze_code_zip(code_bytes)
        st.session_state.feature_hints = build_feature_hints(stats)

        with st.expander("📊 Auto-Preview(요약)", expanded=True):
            lang_str = ", ".join([f"{k} {v}개" for k, v in stats["lang_counts"].most_common()]) if stats["lang_counts"] else "감지된 언어 없음"
            funcs_cnt = len(stats["top_functions"])
            expected_tc = estimate_tc_count(stats)
            st.markdown(
                f"- **파일 수**: {stats['total_files']}\n"
                f"- **언어 분포**: {lang_str}\n"
                f"- **함수/엔드포인트 수(추정)**: {funcs_cnt}\n"
                f"- **예상 테스트케이스 개수(추정)**: {expected_tc}"
            )

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
                        if file.endswith((".py", ".java", ".js", ".ts", ".cpp", ".c", ".cs")):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    code = f.read()
                                full_code += f"\n\n# FILE: {file}\n{code}"
                            except:
                                continue

                feature_hints = st.session_state.get("feature_hints") or {}
                hint_blocks = []
                for key, toks in feature_hints.items():
                    disp = _display_from_key(key)
                    toks_view = ", ".join(sorted(set(toks))[:6])
                    hint_blocks.append(f"- {disp} (key={key}) : {toks_view}")
                hints_md = "\n".join(hint_blocks) if hint_blocks else "- General (key=general)"

                prompt = f"""
너는 시니어 QA 엔지니어이며, 현재 '{qa_role}' 역할을 맡고 있다. 아래 소스코드를 분석하여 **기능별 섹션**으로 테스트케이스를 작성하라. 반드시 아래 형식을 지켜라:
- 각 기능은 "## 기능명" 헤딩으로 시작한다. (예: ## AlarmManager)
- 각 기능 섹션마다 **하나의 마크다운 테이블**만 포함한다.
- 테이블 컬럼: | TC ID | 기능 설명 | 입력값 | 예상 결과 | 우선순위 |
- **TC ID는 반드시 tc-<feature-key>-NNN 형식**을 사용하라. (예: tc-alarm-001)
- <feature-key>는 아래 힌트 목록의 key 중 가장 적합한 값을 사용한다.
- 각 기능 섹션마다 NNN은 001부터 다시 시작한다.
- 기능 섹션 외의 불필요한 텍스트는 넣지 말라.

[기능 힌트 목록]
{hints_md}

[소스코드]
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
                    }
                )
                result = response.json()["choices"][0]["message"]["content"]
                st.session_state.llm_result = result

                try:
                    normalized_md, groups = rebuild_normalized_markdown(result, st.session_state.get("feature_hints"))
                    st.session_state.normalized_markdown = normalized_md
                    st.session_state.parsed_groups = groups if groups else None
                    st.session_state.parsed_df = concat_groups_for_view(groups) if groups else None
                except Exception:
                    st.session_state.normalized_markdown = result
                    st.session_state.parsed_groups = None
                    st.session_state.parsed_df = None

                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.last_model = model
                st.session_state.last_role = qa_role
                st.session_state["is_loading"] = False

    if st.session_state.llm_result:
        st.success("✅ 테스트케이스 생성 완료!")
        st.markdown("## 📋 생성된 테스트케이스")
        st.markdown(
            '<small style="color:#000">'
            '아래는 제공된 소스코드를 분석한 후, 기능 단위의 테스트 시나리오를 기반으로 작성한 테스트 케이스입니다. '
            '각 테스트 케이스는 기능 설명, 입력값, 예상 결과, 그리고 우선순위를 포함합니다.'
            '</small>',
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.normalized_markdown or st.session_state.llm_result)

        st.markdown("---")
        st.markdown("### 설명")
        try:
            groups_for_desc = st.session_state.parsed_groups
            if not groups_for_desc:
                md = st.session_state.get("normalized_markdown") or st.session_state.get("llm_result") or ""
                groups_for_desc = group_tables_and_renumber(md)
            dynamic_md = build_dynamic_explanations(groups_for_desc or {})
            st.markdown(dynamic_md)
        except Exception as _e:
            st.caption("설명 생성 중 경고: 동적 요약에 실패하여 기본 안내만 표시합니다.")
            st.markdown("_기능별 테이블을 기준으로 우선순위 분포와 요약을 제공합니다._")

    # (항상 노출) 엑셀 다운로드
    excel_bytes = None
    try:
        bio = io.BytesIO()
        if st.session_state.get("parsed_groups"):
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                for key, df in st.session_state.parsed_groups.items():
                    df_out = _ensure_priorities(df)
                    sheet = re.sub(r"[^A-Za-z0-9가-힣_ -]", "", key)[:31] or "General"
                    df_out.to_excel(writer, index=False, sheet_name=sheet)
            excel_bytes = bio.getvalue()
        elif st.session_state.get("parsed_df") is not None:
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                pd.DataFrame(_ensure_priorities(st.session_state.parsed_df)).to_excel(writer, index=False, sheet_name="테스트케이스")
            excel_bytes = bio.getvalue()
        elif st.session_state.get("normalized_markdown") or st.session_state.get("llm_result"):
            md = st.session_state.get("normalized_markdown") or st.session_state.get("llm_result") or ""
            groups = group_tables_and_renumber(md)
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                if groups:
                    for key, df in groups.items():
                        df_out = _ensure_priorities(df)
                        sheet = re.sub(r"[^A-Za-z0-9가-힣_ -]", "", key)[:31] or "General"
                        df_out.to_excel(writer, index=False, sheet_name=sheet)
                else:
                    pd.DataFrame(columns=["TC ID","기능 설명","입력값","예상 결과","우선순위"]).to_excel(
                        writer, index=False, sheet_name="테스트케이스"
                    )
            excel_bytes = bio.getvalue()
        else:
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                pd.DataFrame(columns=["TC ID","기능 설명","입력값","예상 결과","우선순위"]).to_excel(
                    writer, index=False, sheet_name="테스트케이스"
                )
            excel_bytes = bio.getvalue()
    except Exception:
        excel_bytes = build_sample_tc_excel()

    st.download_button("⬇️ 엑셀 다운로드", data=excel_bytes, file_name="테스트케이스.xlsx")

# ────────────────────────────────────────────────
# 📑 TAB 2: 테스트케이스 → 명세서 요약 (기존 유지)
# ────────────────────────────────────────────────
with tc_tab:
    st.subheader("📑 테스트케이스 기반 기능/요구사항 명세서 추출기")
    st.download_button(
        "⬇️ 샘플 테스트케이스 엑셀 다운로드",
        data=build_sample_tc_excel(),
        file_name="테스트케이스_샘플.xlsx",
        help="필수 컬럼( TC ID, 기능 설명, 입력값, 예상 결과, 우선순위 ) 포함"
    )

    tc_file = st.file_uploader("📂 테스트케이스 파일 업로드 (.xlsx, .csv)", type=["xlsx", "csv"], key="tc_file")
    summary_type = st.selectbox("📌 요약 유형", ["기능 명세서", "요구사항 정의서"], key="summary_type")

    if st.button("🚀 명세서 생성하기", disabled=st.session_state["is_loading"]) and tc_file:
        st.session_state["is_loading"] = True
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
                }
            )
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
        st.download_button("⬇️ 명세서 텍스트 다운로드", data=st.session_state.spec_result, file_name="기능_요구사항_명세서.txt")

# ────────────────────────────────────────────────
# 🐞 TAB 3: 에러 로그 → 재현 시나리오 (기존 유지)
# ────────────────────────────────────────────────
with log_tab:
    st.subheader("🐞 에러 로그 기반 재현 시나리오 생성기")

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

    log_file = st.file_uploader("📂 에러 로그 파일 업로드 (.log, .txt)", type=["log", "txt"], key="log_file")

    if not API_KEY:
        st.warning("🔐 OpenRouter API Key가 설정되지 않았습니다.")

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
                raw_log_cache, context_lines=5, keep_last_lines_if_empty=2000, char_budget=budget)
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
        st.download_button("⬇️ 시나리오 텍스트 다운로드", data=st.session_state.scenario_result, file_name="재현_시나리오.txt")
