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
                    # 상위 디렉터리명을 기능 후보로
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
                    feature_keys.add(_norm_key(stem))  # 파일명도 후보

                if ext in lang_map:
                    lang_counts[lang_map[ext]] += 1

                try:
                    with zf.open(n) as fh:
                        content = fh.read(100_000).decode("utf-8", errors="ignore")
                        # 함수/메서드
                        for pat in [
                            r"def\s+([a-zA-Z_]\w*)\s*\(",
                            r"function\s+([a-zA-Z_]\w*)\s*\(",
                            r"(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z_<>\[\]]+\s+([a-zA-Z_]\w*)\s*\("
                        ]:
                            top_functions += re.findall(pat, content)
                        # 클래스
                        for cpat in [
                            r"class\s+([A-Z][A-Za-z0-9_]*)",
                            r"(?:public|final|abstract)\s+class\s+([A-Z][A-Za-z0-9_]*)"
                        ]:
                            classes += re.findall(cpat, content)
                except Exception:
                    pass

        # 클래스/함수명도 기능 후보
        for name in classes[:80]:
            feature_keys.add(_norm_key(name))
        for fn in top_functions[:120]:
            feature_keys.add(_norm_key(fn))

    except zipfile.BadZipFile:
        pass

    # 너무 일반적인 키 제거
    generic = {"app","main","index","core","utils","common","service","controller","model","routes","handler","api","src","test","tests"}
    feature_keys = {k for k in feature_keys if k and k not in generic}

    return {
        "total_files": total_files,
        "lang_counts": lang_counts,
        "top_functions": top_functions[:200],
        "module_counts": module_counts,
        "sample_paths": sample_paths,
        # [ADD]
        "classes": classes[:200],
        "feature_keys": sorted(feature_keys)[:40],  # 프롬프트 부담 완화
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

# [ADD] 기능 힌트(aliases) 생성: 각 key에 대해 파일명/클래스/함수 파생 토큰 포함
def build_feature_hints(stats: dict) -> dict:
    keys = stats.get("feature_keys", []) or []
    aliases = defaultdict(set)

    # 원 키
    for k in keys:
        aliases[k].add(k)
        aliases[k].add(k.replace("-", ""))

    # 파일/클래스/함수에서 파생 토큰
    for name in (stats.get("classes") or []) + (stats.get("top_functions") or []):
        norm = _norm_key(name)
        if not norm:
            continue
        # 가장 유사한 키에 매핑(간단: 접두 일치/부분 일치)
        target = None
        for k in keys:
            if norm.startswith(k) or k.startswith(norm) or norm.replace("-","") in k.replace("-",""):
                target = k; break
        if target:
            aliases[target].update({norm, norm.replace("-", ""), name.lower()})

    # 디렉터리/파일 기반 키(이미 analyze에서 넣었음)
    return {k: sorted(v) for k, v in aliases.items()}

# [ADD] 힌트 기반 행→기능 키 추정
def _infer_key_from_row_with_hints(row: pd.Series, hints: dict) -> str:
    text = " ".join([str(row.get(c,"")) for c in ["TC ID","기능 설명","입력값","예상 결과"]]).lower()
    # TCID 접두 우선
    pref = _extract_prefix_from_tcid(str(row.get("TC ID","")))
    if pref:
        return pref
    # 힌트 토큰 스캔
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

# [FIX] 핵심: 원문을 기능별 + ID정규화 ‘원문형식’으로 재구성 (힌트 기반 강제 분리 포함)
def rebuild_normalized_markdown(md_text: str, feature_hints: dict | None) -> tuple[str, dict[str, pd.DataFrame]]:
    groups = group_tables_and_renumber(md_text)
    if not groups:
        tbls = _parse_md_tables_with_heading(md_text)
        if tbls:
            # 단일 테이블이거나 헤딩 매핑 실패 → 힌트 기반 강제 분리
            base_df = _normalize_headers(tbls[0][1])
            hints = feature_hints or {}
            groups = split_single_df_feature_aware(base_df, hints)
        else:
            return (md_text, {})

    # 원문 순서 보존
    ordered = []
    tbls2 = _parse_md_tables_with_heading(md_text)
    seen = set()
    for (heading, df) in tbls2:
        sheet_name, key_id = _normalize_feature_key(heading, df.iloc[0].to_dict() if len(df) else None)
        candidates = [k for k in groups.keys() if k.startswith(sheet_name)]
        name = candidates[0] if candidates else sheet_name
        if name in groups and name not in seen:
            ordered.append(name); seen.add(name)
    for name in groups.keys():
        if name not in seen:
            ordered.append(name)

    parts = []
    for name in ordered:
        df = groups[name]
        parts.append(f"## {name}")
        parts.append(_df_to_md_table(df))
        parts.append("")
    return ("\n".join(parts).strip(), groups)

# ────────────────────────────────────────────────
# [ADD] NEW: "함수명 분석 기반" 샘플 TC 생성기 (기존 유지)
# ────────────────────────────────────────────────
def make_tc_id_from_fn(fn: str, used_ids: set, seq: int | None = None) -> str:
    stop = {
        "get","set","is","has","have","do","make","build","create","update","insert","delete","remove","fetch","load","read","write",
        "put","post","patch","calc","compute","process","handle","run","exec","call","check","validate","convert","parse","format",
        "test","temp","main","init","start","stop","open","close","send","receive","retry","download","upload","save","add","sum","plus","div","divide"
    }
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
# [ADD] 동적 설명 생성을 위한 유틸 (우선순위 정규화/추론 포함)
# ────────────────────────────────────────────────
# [ADD] 우선순위 토큰 정규화
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

# [ADD] 행 단위 우선순위 추론
def _infer_priority_from_text(text: str) -> str:
    s = (text or "").lower()
    if any(k in s for k in ["zerodivision", "division by zero", "0으로", "error", "exception", "fatal", "권한", "unauthorized", "forbidden", "not found", "401", "403", "404", "timeout", "타임아웃", "invalid", "오류"]):
        return "High"
    if any(k in s for k in ["경계", "boundary", "edge", "최대", "최소", "음수", "소수", "float", "overflow", "underflow"]):
        return "Medium"
    return "Medium"

# [ADD] DF 전체에 대해 우선순위 정규화+추론 반영
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

# [FIX] 우선순위 분포 계산: 정규화/추론 이후 카운트
def _priority_counts(df: pd.DataFrame) -> dict:
    df2 = _ensure_priorities(df)
    vals = df2["우선순위"].astype(str).str.strip().str.title().tolist()
    c = Counter(vals)
    return {"High": c.get("High", 0), "Medium": c.get("Medium", 0), "Low": c.get("Low", 0)}

# [ADD] 기능명 → 한국어 설명 휴리스틱
def _feature_korean_desc(name: str, fallback_from_rows: str = "") -> str:
    n = (name or "").lower()
    if any(k in n for k in ["add", "sum", "plus"]):
        return "두 개의 수를 더해 결과를 반환합니다."
    if any(k in n for k in ["div", "divide"]):
        return "두 수를 나누어 결과를 반환하며, 0으로 나누는 경우 예외가 발생합니다."
    if any(k in n for k in ["email", "isemail", "validator", "validate"]):
        return "문자열이 유효한 이메일 형식인지 검사합니다."
    if "health" in n:
        return "헬스체크 엔드포인트의 가용성을 확인합니다."
    # [ADD] 적절한 휴리스틱이 없을 경우, 첫 행의 기능 설명을 짧게 차용
    fb = fallback_from_rows.strip()
    return fb if fb else f"‘{name}’ 기능의 핵심 동작을 검증합니다."

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

# [FIX] 실제로 화면에 넣을 동적 설명 마크다운 생성
#      (요청 반영) 출력 구성: 기능설명, 우선순위 분포, 요약  ― 그리고 헤더는 "Feature (총 N건)" 형태
def build_dynamic_explanations(groups: dict[str, pd.DataFrame]) -> str:
    if not groups:
        return "_설명을 생성할 데이터가 없습니다._"

    parts = []
    for feature_name, df in groups.items():
        df_norm = _ensure_priorities(df)

        # 기능 설명: 휴리스틱 + 첫 행 보조
        first_desc = str(df_norm.iloc[0]["기능 설명"]) if len(df_norm) else ""
        func_desc = _feature_korean_desc(feature_name, fallback_from_rows=first_desc)

        # 우선순위 분포
        pr = _priority_counts(df_norm)

        # 시나리오 성향(정상/예외/경계 비율)로 요약 문구 가변화
        buckets = Counter()
        for _, row in df_norm.iterrows():
            s = " ".join([str(row.get(c,"")) for c in ["기능 설명","입력값","예상 결과"]])
            buckets[_classify_scenario_bucket(s)] += 1

        endpoints = _extract_endpoints(" ".join(df_norm[["기능 설명","입력값","예상 결과"]].astype(str).fillna("").values.ravel().tolist()))
        total = len(df_norm)

        # [FIX] 헤더 포맷 변경: "Div (총 3건)" 형태 (tc 범위 제거)
        parts.append(f"#### {feature_name} (총 {total}건)")
        parts.append(f"- **기능 설명**: {func_desc}")
        parts.append(f"- **우선순위 분포**: High {pr['High']} · Medium {pr['Medium']} · Low {pr['Low']}")

        # [FIX] 우선순위 정리 섹션 삭제 (요청 반영)

        # [FIX] 요약(기능별 상황에 따라 다르게)
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
        parts.append("")  # spacing

    return "\n".join(parts).strip()

# ────────────────────────────────────────────────
# 🧪 TAB 1: 소스코드 → 테스트케이스 자동 생성기
# ────────────────────────────────────────────────
with code_tab:
    st.subheader("🧪 소스코드 기반 테스트케이스 자동 생성기")

    # (유지) 샘플 코드 ZIP만 제공
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

    # Auto-Preview(요약) & Sample TC (기존 유지) + [ADD] 기능힌트 생성
    code_bytes = None
    stats = {"total_files":0,"lang_counts":Counter(),"top_functions":[]}
    if uploaded_file:
        code_bytes = uploaded_file.getvalue()
        stats = analyze_code_zip(code_bytes)
        # [ADD] 기능 힌트 저장 (후처리 강제 분리용)
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

    # LLM 호출
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

                # [FIX] 프롬프트 보강 (표만 생성하도록 유도)
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
- 각 기능 섹션 아래에 생성된 테스트케이스에 관한 설명을 작성한다.

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

                # [FIX] 결과는 'LLM 원문(정규화)'만 표시: 힌트 기반 강제 분리/정규화 포함
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

    # [FIX] 결과 표시: 헤더 문구 + 동적 설명 섹션(요청 반영: 기능설명/우선순위 분포/요약만)
    if st.session_state.llm_result:
        st.success("✅ 테스트케이스 생성 완료!")
        # (요청1) 문구 변경
        st.markdown("## 📋 생성된 테스트케이스")
        # (요청2) 작은 글씨, 검정색 캡션 추가
        st.markdown(
            '<small style="color:#000">'
            '아래는 제공된 소스코드를 분석한 후, 기능 단위의 테스트 시나리오를 기반으로 작성한 테스트 케이스입니다. '
            '각 테스트 케이스는 기능 설명, 입력값, 예상 결과, 그리고 우선순위를 포함합니다.'
            '</small>',
            unsafe_allow_html=True
        )
        # [REQ] LLM 원문 출력: 정규화본 대신 LLM 결과 원문을 그대로 표시하여,
        # 과거 소스처럼 상황에 따라 설명이 자연스럽게 포함될 수 있도록 함
        st.markdown(st.session_state.llm_result)

        # [REQ] 설명 하드코딩 출력 제거: 고정 "설명" 섹션을 주석 처리하여
        # 테이블 아래에 고정 문구가 항상 붙는 동작을 비활성화

    # [FIX] (요청4) 무슨 일이 있어도 '엑셀 다운로드' 버튼은 항상 표시
    excel_bytes = None
    try:
        bio = io.BytesIO()
        if st.session_state.get("parsed_groups"):
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                for key, df in st.session_state.parsed_groups.items():
                    # 저장 전 우선순위 정규화로 일관성 개선
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
            # [REQ] 여러 시트 지원: XLSX는 모든 시트를 읽어 세로 결합
            if tc_file.name.endswith("csv"):
                df = pd.read_csv(tc_file)
            else:
                sheets_dict = pd.read_excel(tc_file, sheet_name=None)  # 모든 시트 로드
                frames = []
                for sheet_name, sdf in sheets_dict.items():
                    if isinstance(sdf, pd.DataFrame) and not sdf.empty:
                        frames.append(sdf)
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                
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
테스트케이스 파일이 여러 시트로 구성된 경우, **모든 시트의 내용을 종합적으로 참고하여** 아래 형식으로 요약하라.

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
