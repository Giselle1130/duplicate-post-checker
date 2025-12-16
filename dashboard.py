import re
import time
import traceback
from datetime import datetime, date as date_cls

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except Exception:
    USE_WDM = False

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="🏰 클랜/방송/디스코드 중복 게시글 체크", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")


# =========================
# 대상 게시판 고정
# =========================
CLUB_ID = 28866679
MENU_ID = 178

BASE_LIST_URL = (
    "https://cafe.naver.com/ArticleList.nhn"
    f"?search.clubid={CLUB_ID}"
    f"&search.menuid={MENU_ID}"
    "&search.boardtype=L"
)

ARTICLEID_RE = re.compile(r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))")
LINK_CSS = "a[href*='articleid='], a[href*='/articles/']"

DETAIL_DATE_SELECTORS = [
    "span.date",
    ".article_info .date",
    ".ArticleTopInfo__date",
    ".ArticleTopInfo .date",
    "p.date",
    "span._articleTime",
]


# =========================
# 유틸
# =========================
def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def kst_today() -> date_cls:
    if KST is None:
        return datetime.now().date()
    return datetime.now(KST).date()


def is_time_token(s: str) -> bool:
    return re.fullmatch(r"\d{1,2}:\d{2}", (s or "").strip()) is not None


def extract_time_token(text: str) -> str:
    m = re.search(r"\b(\d{1,2}:\d{2})\b", clean(text))
    return m.group(1) if m else ""


def extract_date_token(text: str) -> str:
    m = re.search(r"\b(20\d{2}\.\d{2}\.\d{2})\.?\b", clean(text))
    return m.group(1) if m else ""


def build_page_url(page: int) -> str:
    return f"{BASE_LIST_URL}&search.page={page}"


def parse_detail_datetime_text(raw: str):
    s = clean(raw)

    m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.\s*(\d{1,2}):(\d{2})", s)
    if m:
        y, mo, d, hh, mm = map(int, m.groups())
        return datetime(y, mo, d, hh, mm)

    m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.\s*(오전|오후)\s*(\d{1,2}):(\d{2})", s)
    if m:
        y, mo, d = map(int, m.group(1, 2, 3))
        ampm = m.group(4)
        hh = int(m.group(5))
        mm = int(m.group(6))
        if ampm == "오후" and hh != 12:
            hh += 12
        if ampm == "오전" and hh == 12:
            hh = 0
        return datetime(y, mo, d, hh, mm)

    return None


# =========================
# 제목 정규화/토큰
# =========================
STOPWORDS = {
    "steam", "kakao", "paragon", "pubg",
    "클랜", "클랜원", "모집", "환영", "가입",
    "디스코드", "discord", "서버",
    "초보", "신생", "친목", "경쟁", "직장인",
    "일반", "랭크", "랭겜", "스쿼드", "듀오", "솔로",
    "내전", "자유", "이벤트", "안내", "공지",
}


def normalize_title(raw: str) -> str:
    t = clean(raw)
    t = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", t)
    t = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", t)
    t = re.sub(r"\[[^\]]{1,30}\]", " ", t)
    t = re.sub(r"\bLv\.?\s*\d+\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}\s*~\s*\d{1,2}\b", " ", t)
    t = re.sub(r"\b\d{1,2}\s*세\b", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def tokenize(text: str):
    t = normalize_title(text)
    toks = re.findall(r"[a-z]+|[가-힣]+", t)
    toks = [x for x in toks if len(x) >= 2]
    toks = [x for x in toks if x not in STOPWORDS]
    return toks


# =========================
# Selenium
# =========================
def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    opts.page_load_strategy = "eager"

    if headless:
        opts.add_argument("--headless=new")

    # 이미지 차단
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    if USE_WDM:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
    else:
        driver = webdriver.Chrome(options=opts)

    driver.implicitly_wait(0.3)

    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
        )
    except Exception:
        pass

    return driver


def switch_to_cafe_main_iframe(driver) -> bool:
    try:
        driver.switch_to.default_content()
        iframes = driver.find_elements(By.ID, "cafe_main")
        if iframes:
            driver.switch_to.frame("cafe_main")
            return True
    except Exception:
        pass
    return False


def wait_list_loaded(driver):
    wait = WebDriverWait(driver, 20)

    def has_links(d):
        return len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0

    if switch_to_cafe_main_iframe(driver):
        try:
            wait.until(has_links)
            return
        except Exception:
            pass

    driver.switch_to.default_content()
    wait.until(has_links)


def is_notice_row(row_text: str, row_el) -> bool:
    t = clean(row_text)
    lines = [x.strip() for x in t.split("\n") if x.strip()]
    if any(x == "공지" for x in lines):
        return True
    try:
        if row_el is not None and len(row_el.find_elements(By.XPATH, ".//*[normalize-space()='공지']")) > 0:
            return True
    except Exception:
        pass
    return False


def pick_row_author(row_text: str, title: str) -> str:
    t = clean(row_text)
    lines = [x.strip() for x in t.split("\n") if x.strip()]
    lines = [x for x in lines if x != title]
    lines = [x for x in lines if not is_time_token(x)]
    lines = [x for x in lines if not re.fullmatch(r"20\d{2}\.\d{2}\.\d{2}\.?", x)]
    bad = ["조회", "좋아요", "댓글", "댓글수"]
    lines = [x for x in lines if not any(b in x for b in bad)]
    lines = [x for x in lines if x != "공지"]
    for x in lines:
        if 1 <= len(x) <= 30:
            return x
    return ""


def extract_article_id_from_href(href: str) -> str:
    m = ARTICLEID_RE.search(href or "")
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


def get_article_datetime_strict(driver, href: str, pause: float = 0.05):
    try:
        driver.get(href)
        time.sleep(pause)

        switch_to_cafe_main_iframe(driver)
        wait = WebDriverWait(driver, 12)

        for css in DETAIL_DATE_SELECTORS:
            try:
                el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
                dt = parse_detail_datetime_text(el.text)
                if dt:
                    return dt
            except Exception:
                continue

        src = driver.page_source
        m = re.search(r"(\d{4}\.\d{1,2}\.\d{1,2}\.\s*(?:오전|오후)\s*\d{1,2}:\d{2})", src)
        if m:
            dt = parse_detail_datetime_text(m.group(1))
            if dt:
                return dt
        m = re.search(r"(\d{4}\.\d{1,2}\.\d{1,2}\.\s*\d{1,2}:\d{2})", src)
        if m:
            dt = parse_detail_datetime_text(m.group(1))
            if dt:
                return dt

    except Exception:
        return None

    return None


# =========================
# 진행/중지/디버그를 위한 "쪼개기(스텝 실행)" 상태머신
# =========================
def log(msg: str):
    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def ensure_state():
    ss = st.session_state
    if "running" not in ss:
        ss.running = False
    if "phase" not in ss:
        ss.phase = "idle"  # idle | collect | validate | done
    if "driver" not in ss:
        ss.driver = None
    if "logs" not in ss:
        ss.logs = []
    if "candidates" not in ss:
        ss.candidates = {}
    if "collected" not in ss:
        ss.collected = {}
    if "page" not in ss:
        ss.page = 1
    if "no_match_pages" not in ss:
        ss.no_match_pages = 0
    if "validate_keys" not in ss:
        ss.validate_keys = []
    if "validate_i" not in ss:
        ss.validate_i = 0
    if "last_url" not in ss:
        ss.last_url = ""
    if "posts" not in ss:
        ss.posts = []


def reset_job():
    ss = st.session_state
    # 드라이버 정리
    try:
        if ss.driver is not None:
            ss.driver.quit()
    except Exception:
        pass

    ss.running = False
    ss.phase = "idle"
    ss.driver = None
    ss.logs = []
    ss.candidates = {}
    ss.collected = {}
    ss.page = 1
    ss.no_match_pages = 0
    ss.validate_keys = []
    ss.validate_i = 0
    ss.last_url = ""
    ss.posts = []


def start_job(target_date: date_cls, headless: bool, max_pages: int, stop_no_match_pages: int, pause: float):
    reset_job()
    ss = st.session_state
    ss.target_date = target_date
    ss.headless = headless
    ss.max_pages = int(max_pages)
    ss.stop_no_match_pages = int(stop_no_match_pages)
    ss.pause = float(pause)

    ss.driver = make_driver(headless=headless)
    ss.phase = "collect"
    ss.running = True
    log(f"시작: target_date={target_date} headless={headless}")


def stop_job():
    ss = st.session_state
    ss.running = False
    log("중지(사용자 요청)")


def finalize_job():
    ss = st.session_state
    df = pd.DataFrame(list(ss.collected.values()))
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
        if "date_detail" in df.columns:
            df = df.sort_values(by="date_detail", ascending=False)
    ss.posts = df.to_dict("records")
    ss.phase = "done"
    ss.running = False
    log(f"완료: 최종 {len(ss.posts)}개")


def step_collect():
    """
    한 번 실행에서 '페이지 몇 개'만 처리 (UI 멈춤 방지)
    """
    ss = st.session_state
    d = ss.driver

    today = kst_today()
    is_today = (ss.target_date == today)
    target_dot = ss.target_date.strftime("%Y.%m.%d")
    target_iso = ss.target_date.strftime("%Y-%m-%d")

    # 이번 스텝에서 처리할 페이지 수 (고정)
    pages_per_step = int(ss.pages_per_step)

    processed = 0
    while ss.page <= ss.max_pages and processed < pages_per_step and ss.running:
        url = build_page_url(ss.page)
        ss.last_url = url
        log(f"[목록] page={ss.page}")
        try:
            d.get(url)
            wait_list_loaded(d)
            time.sleep(ss.pause)

            rows = d.find_elements(By.CSS_SELECTOR, "tr")
            if len(rows) < 5:
                rows = d.find_elements(By.CSS_SELECTOR, "li") + rows

            page_candidate = 0

            for row in rows:
                try:
                    row_text = clean(row.text)
                    if not row_text:
                        continue
                    if is_notice_row(row_text, row):
                        continue

                    links = row.find_elements(By.CSS_SELECTOR, LINK_CSS)
                    if not links:
                        continue
                    a = links[0]
                    href = clean(a.get_attribute("href"))
                    if not href:
                        continue

                    article_id = extract_article_id_from_href(href)
                    if not article_id:
                        continue

                    title_raw = clean(a.text)
                    if not title_raw:
                        lines = [x.strip() for x in row_text.split("\n") if x.strip()]
                        title_raw = lines[0] if lines else ""
                    if not title_raw:
                        continue

                    hhmm = extract_time_token(row_text)
                    dot = extract_date_token(row_text)

                    # 후보 최소조건(속도용)
                    if is_today:
                        if not hhmm:
                            continue
                        date_raw = hhmm
                    else:
                        if hhmm:
                            continue
                        if not dot or dot != target_dot:
                            continue
                        date_raw = dot

                    if href not in ss.candidates:
                        ss.candidates[href] = {
                            "date": target_iso,
                            "date_raw": date_raw,
                            "author": pick_row_author(row_text, title_raw),
                            "title": title_raw,
                            "title_norm": normalize_title(title_raw),
                            "link": href,
                        }
                        page_candidate += 1

                except Exception:
                    continue

            if page_candidate == 0:
                ss.no_match_pages += 1
            else:
                ss.no_match_pages = 0

            # 조기 종료 조건
            if ss.no_match_pages >= ss.stop_no_match_pages:
                log("목록 조기 종료(연속 0페이지)")
                ss.page = ss.max_pages + 1
                break

        except Exception as e:
            log(f"목록 오류: {type(e).__name__}: {e}")

        ss.page += 1
        processed += 1

    # 목록이 끝나면 validate로 전환
    if ss.page > ss.max_pages or ss.no_match_pages >= ss.stop_no_match_pages:
        ss.validate_keys = list(ss.candidates.keys())
        ss.validate_i = 0
        ss.phase = "validate"
        log(f"상세 검증 단계로 전환: 후보 {len(ss.validate_keys)}개")


def step_validate():
    """
    한 번 실행에서 '게시글 몇 개'만 상세 검증
    """
    ss = st.session_state
    d = ss.driver

    per_step = int(ss.articles_per_step)

    processed = 0
    while ss.validate_i < len(ss.validate_keys) and processed < per_step and ss.running:
        href = ss.validate_keys[ss.validate_i]
        ss.last_url = href

        try:
            dt = get_article_datetime_strict(d, href, pause=ss.pause)

            # 못읽으면 버림 (섞임 방지)
            if dt and dt.date() == ss.target_date:
                base = ss.candidates[href]
                out = dict(base)
                out["date_detail"] = dt.strftime("%Y-%m-%d %H:%M")
                ss.collected[href] = out

        except Exception as e:
            log(f"상세 오류: {type(e).__name__}: {e}")

        ss.validate_i += 1
        processed += 1

    if ss.validate_i >= len(ss.validate_keys):
        finalize_job()


# =========================
# 중복/유사
# =========================
def compute_author_dups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "author", "count"])
    a = df.groupby(["date", "author"]).size().reset_index(name="count")
    return a[a["count"] >= 2].sort_values(by="count", ascending=False)


def compute_exact_dups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    return df[df.duplicated(subset=["date", "title_norm"], keep=False)].copy()


def compute_keyword_groups(df: pd.DataFrame, min_count: int = 2):
    if df.empty:
        return pd.DataFrame(columns=["keyword", "count", "examples"])

    tokens_list = [tokenize(t) for t in df["title"].fillna("").astype(str).tolist()]
    inv = {}
    for idx, toks in enumerate(tokens_list):
        for tok in set(toks):
            inv.setdefault(tok, []).append(idx)

    rows = []
    for kw, idxs in inv.items():
        if len(idxs) >= min_count:
            ex = [df.iloc[i]["title"] for i in idxs[:3]]
            rows.append({"keyword": kw, "count": len(idxs), "examples": " | ".join(ex)})

    out = pd.DataFrame(rows)
    return out.sort_values(by=["count", "keyword"], ascending=[False, True]) if not out.empty else out


def compute_ai_similar(df: pd.DataFrame, threshold: float = 0.78) -> pd.DataFrame:
    cols = ["title_a", "title_b", "similarity", "link_a", "link_b"]
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=cols)

    titles_raw = df["title"].fillna("").astype(str).tolist()
    titles = df["title_norm"].fillna("").astype(str).tolist()
    links = df["link"].fillna("").astype(str).tolist()

    vec_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    Xw = vec_w.fit_transform(titles)
    Mw = cosine_similarity(Xw)

    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xc = vec_c.fit_transform(titles)
    Mc = cosine_similarity(Xc)

    M = 0.55 * Mw + 0.45 * Mc

    rows = []
    n = len(titles)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(M[i, j])
            if s >= threshold:
                rows.append({
                    "title_a": titles_raw[i],
                    "title_b": titles_raw[j],
                    "similarity": round(s, 3),
                    "link_a": links[i],
                    "link_b": links[j],
                })

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(by="similarity", ascending=False) if not out.empty else out


# =========================
# UI
# =========================
ensure_state()

with st.expander("설정", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택(✅ 이 날짜만)", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=500, value=120, step=5)
    with c4:
        stop_no_match_pages = st.number_input("연속 0페이지면 종료", min_value=1, max_value=10, value=2, step=1)
    with c5:
        pause = st.number_input("페이지 대기(초)", min_value=0.05, max_value=2.00, value=0.12, step=0.01)

    c6, c7, c8 = st.columns([1, 1, 1])
    with c6:
        pages_per_step = st.number_input("한 번에 목록 페이지 처리", min_value=1, max_value=10, value=2, step=1)
    with c7:
        articles_per_step = st.number_input("한 번에 상세 글 검증", min_value=1, max_value=30, value=10, step=1)
    with c8:
        auto_run = st.checkbox("자동 진행(켜면 알아서 계속)", value=True)

    st.session_state.pages_per_step = int(pages_per_step)
    st.session_state.articles_per_step = int(articles_per_step)

st.divider()

btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 2])
with btn1:
    if st.button("▶ 시작", use_container_width=True):
        try:
            start_job(target_date, headless, int(max_pages), int(stop_no_match_pages), float(pause))
            st.rerun()
        except Exception:
            st.error("시작 오류")
            st.code(traceback.format_exc())

with btn2:
    if st.button("⏭ 진행(한 번)", use_container_width=True):
        st.session_state.running = True
        st.rerun()

with btn3:
    if st.button("⏹ 중지", use_container_width=True):
        stop_job()
        st.rerun()

with btn4:
    debug = st.checkbox("🪲 디버그 보기", value=False)

# 진행 표시
phase = st.session_state.phase
running = st.session_state.running

status = st.empty()
pbar1 = st.progress(0)
pbar2 = st.progress(0)

# 진행률 계산
if phase in ("collect", "validate", "done"):
    # 1) 목록 단계 진행률
    maxp = max(1, int(st.session_state.max_pages) if "max_pages" in st.session_state else int(max_pages))
    curp = min(maxp, max(1, int(st.session_state.page)))
    p1 = min(1.0, curp / maxp)
    pbar1.progress(int(p1 * 100))

    # 2) 상세 검증 단계 진행률
    total = max(1, len(st.session_state.validate_keys))
    done = min(total, int(st.session_state.validate_i))
    p2 = min(1.0, done / total)
    pbar2.progress(int(p2 * 100))

if phase == "idle":
    status.info("대기 중. ▶ 시작을 눌러줘.")
elif phase == "collect":
    status.info(
        f"목록 수집 중… page={st.session_state.page-1} / 후보={len(st.session_state.candidates)} "
        f"(마지막 URL: {st.session_state.last_url})"
    )
elif phase == "validate":
    status.info(
        f"상세 작성일 검증 중… {st.session_state.validate_i} / {len(st.session_state.validate_keys)} "
        f"(통과={len(st.session_state.collected)})"
    )
elif phase == "done":
    status.success(f"완료! 선택한 날짜 글만 {len(st.session_state.posts)}개")

# 디버그 로그
if debug:
    st.caption("DEBUG LOG")
    st.code("\n".join(st.session_state.logs[-200:]) if st.session_state.logs else "(로그 없음)")
    st.caption(f"last_url = {st.session_state.last_url}")

# 실제 작업 스텝 실행
if running and phase in ("collect", "validate"):
    try:
        if phase == "collect":
            step_collect()
        elif phase == "validate":
            step_validate()
    except Exception as e:
        log(f"치명 오류: {type(e).__name__}: {e}")
        st.session_state.running = False

    # 자동 진행이면 계속 rerun
    if auto_run and st.session_state.running and st.session_state.phase in ("collect", "validate"):
        time.sleep(0.15)  # UI 숨 쉴 틈
        st.rerun()

# 결과 표시
df = (
    pd.DataFrame(st.session_state.posts)
    if st.session_state.posts
    else pd.DataFrame(columns=["date", "date_raw", "date_detail", "author", "title", "title_norm", "link"])
)

keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

author_dups = compute_author_dups(df)
exact_dups = compute_exact_dups(df)
keyword_groups = compute_keyword_groups(df, min_count=int(keyword_min_count))
ai_similar = compute_ai_similar(df, threshold=float(sim_threshold))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 원본", "🚨 작성자 동일", "🧨 제목 동일", "🔎 키워드 중복", "🤖 AI 유사"])

with tab1:
    st.dataframe(df, use_container_width=True)

with tab2:
    st.dataframe(author_dups if not author_dups.empty else pd.DataFrame(), use_container_width=True)

with tab3:
    st.dataframe(exact_dups if not exact_dups.empty else pd.DataFrame(), use_container_width=True)

with tab4:
    st.dataframe(keyword_groups if not keyword_groups.empty else pd.DataFrame(), use_container_width=True)

with tab5:
    st.dataframe(ai_similar if not ai_similar.empty else pd.DataFrame(), use_container_width=True)
