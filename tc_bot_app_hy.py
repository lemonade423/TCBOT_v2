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

# [ADD] 기능별 그룹 보관용 세션 키 (엑셀 시트 분리용)
if "parsed_groups" not in st.session_state:
    st.session_state["parsed_groups"] = None

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
# [ADD] 샘플 파일/샘플 TC 엑셀 빌더 (기존 유지)
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
# [ADD] 코드 ZIP 분석/프리뷰 유틸 (기존 유지)
# ────────────────────────────────────────────────
def analyze_code_zip(zip_bytes: bytes) -> dict:
    lang_map = {
        ".py": "Python", ".java": "Java", ".js": "JS", ".ts": "TS",
        ".cpp": "CPP", ".c": "C", ".cs": "CS"
    }
    lang_counts = Counter()
    top_functions = []
    total_files = 0
    module_counts = Counter()
    sample_paths = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = zf.namelist()
            total_files = len(names)
            sample_paths = names[:10]
            for n in names:
                parts = n.split("/")
                module = parts[0] if len(parts) > 1 else "(root)"
                if not n.endswith("/"):
                    module_counts[module] += 1
                ext = os.path.splitext(n)[1].lower()
                if ext in lang_map:
                    lang_counts[lang_map[ext]] += 1
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
        "top_functions": top_functions[:50],
        "module_counts": module_counts,
        "sample_paths": sample_paths
    }


def estimate_tc_count(stats: dict) -> int:
    files = max(0, stats.get("total_files", 0))
    langs = sum(stats.get("lang_counts", Counter()).values())
    funcs = len(stats.get("top_functions", []))
    estimate = int(files * 0.3 + langs * 0.7 + funcs * 0.9)
    return max(3, min(estimate, 300))

# ────────────────────────────────────────────────
# [ADD] LLM 결과 포맷(핵심): 기능별 테이블 분리 + 그룹별 TC ID 재넘버링 + 엑셀 시트 분리
# ────────────────────────────────────────────────

# [ADD] 코드펜스 제거(테이블 파싱 방해 방지)
def _strip_code_fences(md: str) -> str:
    return re.sub(r"```.*?```", "", md, flags=re.DOTALL)

# [ADD] 마크다운 테이블 + 직전 헤딩 매핑 추출
def _parse_md_tables_with_heading(md_text: str) -> list[tuple[str, pd.DataFrame]]:
    text = _strip_code_fences(md_text)
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines) - 1:
        header = lines[i].strip()
        sep = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if "|" in header and re.search(r"\|\s*:?-{2,}\s*\|", sep):
            feature_name = ""
            for back in range(1, 6):
                if i - back < 0:
                    break
                prev = lines[i - back].strip()
                m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", prev)
                if m:
                    feature_name = m.group(1); break
                m2 = re.match(r"^\s{0,3}\*\*(.+?)\*\*\s*$", prev)
                if m2:
                    feature_name = m2.group(1); break
                m3 = re.match(r"^\s*(기능|Feature)\s*[:：]\s*(.+?)\s*$", prev, flags=re.IGNORECASE)
                if m3:
                    feature_name = m3.group(2); break
            j = i + 2
            rows = [header, sep]
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
        else:
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

# [ADD] 기능 키 정규화(시트명/ID용)
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

# [ADD] TCID 접두 추출 (예: TC-AlarmMgr-001 → 'alarmmgr')
def _extract_prefix_from_tcid(tcid: str) -> str | None:
    m = re.match(r"(?i)^TC[-_]?([A-Za-z][A-Za-z0-9]+)-\d{2,4}$", str(tcid).strip())
    if m:
        return m.group(1).lower()
    return None

# [ADD] 기능 키 추정(단일 DF 강제 분리용): TCID 접두→기능설명 키워드→Fallback
def _infer_key_from_row(row: pd.Series) -> str:
    tcid = str(row.get("TC ID",""))
    feat = str(row.get("기능 설명",""))
    pref = _extract_prefix_from_tcid(tcid)
    if pref:
        return pref
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", feat)
    if tokens:
        return "-".join(tokens[:2]).lower()
    return "general"

# [ADD] 단일 DF → 기능별 분리
def split_single_df(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df2 = _normalize_headers(df).fillna("")
    df2["_k_"] = df2.apply(_infer_key_from_row, axis=1)
    groups = {}
    for k, sub in df2.groupby("_k_"):
        sub = sub.drop(columns=["_k_"]).reset_index(drop=True)
        sheet, key_id = _normalize_feature_key(k, sub.iloc[0].to_dict() if len(sub) else None)
        sub["TC ID"] = [f"tc-{key_id}-{i:03d}" for i in range(1, len(sub)+1)]
        groups[sheet[:31] or "General"] = sub
    return groups

# [ADD] 핵심: 문서 전체 → 기능별 그룹핑(테이블 경계 보존) + 그룹 내 tc-<key>-NNN 재부여
def group_tables_and_renumber(md_text: str) -> dict[str, pd.DataFrame]:
    tbls = _parse_md_tables_with_heading(md_text)
    if not tbls:
        return {}
    # 1) 헤딩 존재/부재에 관계없이, 표 단위로 우선 분리
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

# [ADD] 화면 표시용(결합 표): 보기 편하도록 기능컬럼 추가해 합쳐서 보여줌
def concat_groups_for_view(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame(columns=["기능","TC ID","기능 설명","입력값","예상 결과","우선순위"])
    view_rows = []
    for sheet, df in groups.items():
        df2 = df.copy()
        df2["기능"] = sheet
        view_rows.append(df2)
    return pd.concat(view_rows, ignore_index=True)[["기능","TC ID","기능 설명","입력값","예상 결과","우선순위"]]

# ────────────────────────────────────────────────
# [FIX] Auto-Preview(Sample TC) 생성기 — 다양성/중복방지 강화
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

# [FIX] 템플릿 다양화 + 결정적 변이(해시) + 중복방지 + 구체 입력/결과
def build_function_based_sample_tc(top_functions: list[str]) -> pd.DataFrame:
    # 분류
    def classify(fn: str) -> str:
        s = fn.lower()
        if any(k in s for k in ["add","sum","plus"]): return "add"
        if any(k in s for k in ["div","divide"]): return "div"
        if any(k in s for k in ["get","fetch","load","read","list","find"]): return "read"
        if any(k in s for k in ["save","create","update","insert","post","put","patch"]): return "write"
        if any(k in s for k in ["delete","remove"]): return "delete"
        if any(k in s for k in ["auth","login","signin","verify","token","oauth","jwt"]): return "auth"
        if any(k in s for k in ["email","validate","regex","check","phone","url"]): return "validate"
        if any(k in s for k in ["upload","download","request","client","socket","stream","io"]): return "io"
        return "default"

    # [ADD] 각 kind별 다양한 시나리오(입력/예상결과를 구체적으로)
    TEMPLATE_POOL = {
        "add": [
            ("{fn} 정상 합산", "a=10, b=20", "30 반환"),
            ("{fn} 음수/양수 혼합", "a=-5, b=8", "3 반환"),
            ("{fn} 실수 합산", "a=0.1, b=0.2", "부동소수 오차 허용 범위 내 0.3"),
            ("{fn} 대용량 정수", "a=10**9, b=10**9", "2*10**9 반환/오버플로우 없음"),
        ],
        "div": [
            ("{fn} 정상 나눗셈", "a=9, b=3", "3 반환"),
            ("{fn} 0 나눗셈 예외", "a=1, b=0", "ZeroDivisionError/HTTP 400"),
            ("{fn} 실수 나눗셈", "a=1, b=4", "0.25 반환(반올림 정책 확인)"),
        ],
        "read": [
            ("{fn} 페이지네이션 조회", "page=1, size=20", "20건 반환 및 next 링크 포함"),
            ("{fn} 필터 조건 조회", "status='ACTIVE'", "상태 일치 레코드만 반환"),
            ("{fn} 존재하지 않는 키", "id=999999", "404/빈 결과"),
        ],
        "write": [
            ("{fn} 신규 생성", "payload={'name':'A','value':1}", "201/ID 발급 & 영속"),
            ("{fn} 필수값 누락", "payload={'value':1}", "400/필수 필드 누락 메시지"),
            ("{fn} 중복 키 처리", "payload={'id':1,'name':'dup'}", "409/중복 충돌"),
        ],
        "delete": [
            ("{fn} 정상 삭제", "id=1 (존재)", "204/재조회 시 미존재"),
            ("{fn} 미존재 삭제", "id=999999", "404 또는 멱등 처리"),
        ],
        "auth": [
            ("{fn} 유효 토큰", "Authorization='Bearer VALID.JWT'", "200/권한 허용"),
            ("{fn} 만료 토큰", "Authorization='Bearer EXPIRED.JWT'", "401/토큰 만료"),
            ("{fn} 권한 부족", "Authorization='Bearer NO_SCOPE'", "403/권한 부족"),
        ],
        "validate": [
            ("{fn} 이메일 정상", "s='user@example.com'", "True 반환"),
            ("{fn} 이메일 이상", "s='no-at-symbol'", "False/규칙 위반"),
            ("{fn} URL 검증", "s='https://example.com/path?x=1'", "True/허용"),
            ("{fn} 전화번호 검증", "s='+82-10-1234-5678'", "지역 규칙에 맞게 True/False"),
        ],
        "io": [
            ("{fn} 업로드 성공", "file=1MB, timeout=5s", "200/무결성 해시 일치"),
            ("{fn} 다운로드 지연", "timeout=1s (지연 환경)", "타임아웃 후 재시도/백오프"),
            ("{fn} 스트림 중단", "연결 강제 종료", "부분 수신 처리 및 복구 로직"),
        ],
        "default": [
            ("{fn} 정상 시나리오", "유효 파라미터 1세트", "성공 코드/정상 반환"),
            ("{fn} 경계/비정상", "필수값 누락/타입 불일치", "명확한 오류 메시지/코드"),
        ]
    }

    used_ids: set[str] = set()
    used_titles: set[str] = set()
    rows: list[list[str]] = []
    kinds_added: set[str] = set()

    # 결정적 인덱스 선택(랜덤 금지, 함수명 해시 기반)
    def pick_indices(fn: str, pool_len: int, want: int = 2) -> list[int]:
        if pool_len == 0: return []
        # sha1 해시로 변이, 충돌 줄임
        h = int(sha1(fn.encode("utf-8")).hexdigest(), 16)
        base = h % pool_len
        step = max(1, (h // 997) % pool_len)
        idxs = []
        cur = base
        for _ in range(want * 3):  # 여유 루프로 중복 피하기
            if cur not in idxs:
                idxs.append(cur)
                if len(idxs) >= want:
                    break
            cur = (cur + step) % pool_len
        return idxs[:want]

    seq = 1
    for fn in top_functions:
        kind = classify(fn)
        if kind in kinds_added:
            continue
        pool = TEMPLATE_POOL.get(kind, TEMPLATE_POOL["default"])
        # 함수명을 템플릿에 주입하고, 해시 기반으로 서로 다른 1~2개 선택
        want_cnt = 2 if kind in {"add","div","write","auth"} else 1
        indices = pick_indices(fn, len(pool), want=want_cnt)
        added_local = 0
        for i in indices:
            title_t, inp_t, exp_t = pool[i]
            title = title_t.format(fn=fn)
            if title in used_titles:
                continue
            tcid = make_tc_id_from_fn(fn, used_ids, seq)
            seq += 1
            rows.append([tcid, title, inp_t, exp_t, "High" if kind in {"div","auth","write","delete","io","validate"} else "Medium"])
            used_titles.add(title)
            added_local += 1
            if len(rows) >= 3:
                break
        if added_local > 0:
            kinds_added.add(kind)
        if len(rows) >= 3:
            break

    # 후보가 부족하면 보강(기본/경계 조합) — 단, 모호 표현 금지
    if len(rows) == 0:
        tc1 = make_tc_id_from_fn("Bootstrap_Init", used_ids, 1)
        tc2 = make_tc_id_from_fn("CorePath_Error", used_ids, 2)
        rows = [
            [tc1, "애플리케이션 부팅 경로 확인", "config=default.yaml, ENV=dev", "초기 화면 렌더/로그인 버튼 노출", "Medium"],
            [tc2, "핵심 경로 오류 처리", "payload={'id':None}", "400/필수 필드 누락 메시지", "High"],
        ]
    elif len(rows) == 1:
        # 하나일 때는 동일 kind 템플릿에서 다른 케이스 추가(중복검사 포함)
        fn = top_functions[0] if top_functions else "DefaultCase"
        kind = classify(fn)
        pool = TEMPLATE_POOL.get(kind, TEMPLATE_POOL["default"])
        for i, (title_t, inp_t, exp_t) in enumerate(pool):
            title = title_t.format(fn=fn)
            if title in used_titles:
                continue
            tcid = make_tc_id_from_fn(fn, used_ids, seq); seq += 1
            rows.append([tcid, title, inp_t, exp_t, "High" if kind in {"div","auth","write","delete","io","validate"} else "Medium"])
            break

    return pd.DataFrame(rows[:3], columns=["TC ID","기능 설명","입력값","예상 결과","우선순위"])

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

    uploaded_file = st.file_uploader("📂 소스코드 zip 파일 업로드",
                                     type=["zip"],
                                     key="code_zip")

    def need_llm_call(uploaded_file, model, role):
        return uploaded_file and (st.session_state.last_uploaded_file
                                  != uploaded_file.name
                                  or st.session_state.last_model != model
                                  or st.session_state.last_role != role)

    qa_role = st.session_state.get("qa_role", "기능 QA")

    # Auto-Preview(요약) & Sample TC (기존 유지)
    code_bytes = None
    if uploaded_file:
        code_bytes = uploaded_file.getvalue()
        stats = analyze_code_zip(code_bytes)

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

            # [FIX] ▼ 기능 분리 보강: 1) 표+헤딩 분리 시도 → 2) 실패 시 단일 DF 강제 분리 ▼
            try:
                tbl_with_heading = _parse_md_tables_with_heading(result)
                if tbl_with_heading:
                    # 표가 여러 개인 경우: 표 단위로 분리(헤딩 유무와 무관)
                    groups = group_tables_and_renumber(result)
                else:
                    # 표 파싱이 아예 안 되었을 때: 파이프 라인/CSV 등은 기존 로직을 건드리지 않음
                    groups = {}

                # 단일 표/분리 실패 시: TCID prefix/기능설명 키워드로 강제 분할
                if not groups and tbl_with_heading:
                    # tbl_with_heading에 1개만 있는 경우
                    single_df = tbl_with_heading[0][1]
                    groups = split_single_df(single_df)

                st.session_state.parsed_groups = groups if groups else None
                st.session_state.parsed_df = concat_groups_for_view(groups) if groups else None
            except Exception:
                st.session_state.parsed_groups = None
                st.session_state.parsed_df = None
            # [FIX] ▲ 변경 끝 ▲

            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.last_model = model
            st.session_state.last_role = qa_role
        st.session_state["is_loading"] = False

    # 결과 표시(원문 + 기능별 표 미리보기)
    if st.session_state.llm_result:
        st.success("✅ 테스트케이스 생성 완료!")
        st.markdown("## 📋 생성된 테스트케이스 (LLM 원문)")
        st.markdown(st.session_state.llm_result)

    if st.session_state.parsed_groups:
        st.markdown("## 📦 기능별 테스트케이스 (테이블 분리 + 기능별 ID 재넘버링 반영)")
        for key, df in st.session_state.parsed_groups.items():
            st.markdown(f"#### 기능: `{key}`")
            st.dataframe(df, use_container_width=True)

    # [FIX] 엑셀 다운로드: 기능별 '시트' 분리(시트명=기능명). 그룹 없으면 단일 시트 폴백.
    if (st.session_state.parsed_groups or st.session_state.parsed_df is not None) and not need_llm_call(
            uploaded_file, model, qa_role):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            if st.session_state.parsed_groups:
                with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
                    for key, df in st.session_state.parsed_groups.items():
                        sheet = re.sub(r"[^A-Za-z0-9가-힣_ -]", "", key)[:31] or "General"
                        df.to_excel(writer, index=False, sheet_name=sheet)
            else:
                st.session_state.parsed_df.to_excel(tmp.name, index=False, sheet_name="테스트케이스")
            tmp.seek(0)
            st.download_button("⬇️ 엑셀 다운로드",
                               data=tmp.read(),
                               file_name="테스트케이스.xlsx")

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

    tc_file = st.file_uploader("📂 테스트케이스 파일 업로드 (.xlsx, .csv)",
                               type=["xlsx", "csv"],
                               key="tc_file")
    summary_type = st.selectbox("📌 요약 유형", ["기능 명세서", "요구사항 정의서"],
                                key="summary_type")

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

    log_file = st.file_uploader("📂 에러 로그 파일 업로드 (.log, .txt)",
                                type=["log", "txt"],
                                key="log_file")
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
