import re
import time
import traceback
from datetime import datetime, date as date_cls
from urllib.parse import urljoin

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =========================
# Streamlit (반드시 최상단)
# =========================
st.set_page_config(
    page_title="🏰 클랜/방송/디스코드 중복 게시글 체크",
    layout="wide",
)

# ✅ UI 폭 1400px 고정(중앙)
st.markdown(
    """
    <style>
      .block-container { max-width: 1400px !important; padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TZ
# =========================
try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


def kst_today() -> date_cls:
    return datetime.now(KST).date() if KST else datetime.now().date()


def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


# =========================
# 대상 게시판
# =========================
CLUB_ID = 28866679
MENU_ID = 178
BASE = "https://cafe.naver.com"

BASE_LIST_URL = (
    "https://cafe.naver.com/ArticleList.nhn"
    f"?search.clubid={CLUB_ID}"
    f"&search.menuid={MENU_ID}"
    "&search.boardtype=L"
)

LINK_CSS = "a[href*='articleid='], a[href*='/articles/']"
ARTICLEID_RE = re.compile(r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))")

# 상세 작성일 후보 셀렉터 (UI 변화 대비)
DETAIL_DATE_SELECTORS = [
    "span.date",
    ".article_info .date",
    ".ArticleTopInfo__date",
    ".ArticleTopInfo .date",
    "p.date",
    "span._articleTime",
]


def build_page_url(page: int) -> str:
    return f"{BASE_LIST_URL}&search.page={page}"


def extract_article_id(href: str) -> str:
    m = ARTICLEID_RE.search(href or "")
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


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
# Selenium (Render 호환: Selenium Manager)
# =========================
def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    # ✅ 요청: 1400x900
    opts.add_argument("--window-size=1400,900")
    opts.page_load_strategy = "eager"

    if headless:
        opts.add_argument("--headless=new")

    # Render/리눅스 안정성
    opts.add_argument("--remote-debugging-port=9222")

    # 이미지 차단
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    driver = webdriver.Chrome(options=opts)  # ✅ webdriver_manager 사용 금지
    driver.implicitly_wait(0.2)
    return driver


def switch_to_cafe_iframe(driver) -> bool:
    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    # name 시도
    try:
        WebDriverWait(driver, 2).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "cafe_main")))
        return True
    except Exception:
        pass

    # id 시도
    try:
        driver.switch_to.default_content()
        iframes = driver.find_elements(By.ID, "cafe_main")
        if iframes:
            driver.switch_to.frame("cafe_main")
            return True
    except Exception:
        pass

    return False


def wait_any_links(driver, timeout=12) -> bool:
    wait = WebDriverWait(driver, timeout)

    def has_links(d):
        return len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0

    if switch_to_cafe_iframe(driver):
        try:
            wait.until(has_links)
            return True
        except Exception:
            pass

    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    try:
        wait.until(has_links)
        return True
    except Exception:
        return False


def collect_links_from_list_page(driver):
    """
    ✅ 목록에서 후보 링크만 최대한 강하게 수집 (날짜표시/시간표시 믿지 않음)
    """
    links = driver.find_elements(By.CSS_SELECTOR, LINK_CSS)
    uniq = {}

    for a in links:
        try:
            href = clean(a.get_attribute("href"))
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(BASE, href)

            aid = extract_article_id(href)
            if not aid:
                continue

            title = clean(a.text) or clean(a.get_attribute("title"))
            if not title:
                continue

            author = ""
            try:
                parent = a.find_element(By.XPATH, "./ancestor::*[self::tr or self::li][1]")
                parts = [x.strip() for x in clean(parent.text).split("\n") if x.strip()]
                parts = [p for p in parts if p != title and "조회" not in p and "댓글" not in p and "좋아요" not in p and p != "공지"]
                author = parts[0] if parts else ""
            except Exception:
                author = ""

            if aid not in uniq:
                uniq[aid] = (href, title, author)
        except Exception:
            continue

    return uniq


# =========================
# 상세 작성일 파싱 (선택 날짜 100% 확정)
# =========================
def parse_detail_datetime_text(raw: str):
    s = clean(raw)

    # 2025.12.16. 23:58
    m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.\s*(\d{1,2}):(\d{2})", s)
    if m:
        y, mo, d, hh, mm = map(int, m.groups())
        return datetime(y, mo, d, hh, mm)

    # 2025.12.16. 오후 11:58 / 오전 9:02
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


def get_article_datetime(driver, href: str, pause: float = 0.05):
    try:
        driver.get(href)
        time.sleep(pause)
        switch_to_cafe_iframe(driver)
        wait = WebDriverWait(driver, 10)

        for css in DETAIL_DATE_SELECTORS:
            try:
                el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
                dt = parse_detail_datetime_text(el.text)
                if dt:
                    return dt
            except Exception:
                continue

        # 최후: 소스에서 날짜패턴 찾기
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
# 수집: 목록 후보 -> 상세 검증(B)
# =========================
def collect_strict_by_date(
    target_date: date_cls,
    headless: bool,
    max_pages: int,
    stop_no_new_pages: int,
    pause: float,
    progress_bar,
):
    """
    ✅ 해결되는 문제:
    - 17일 선택시 16일 글 섞임 -> 상세 작성일로 제거
    - 16일 선택시 0개 -> 목록 표기와 무관하게 후보를 모아 상세에서 날짜 확정
    """
    driver = make_driver(headless=headless)
    seen_ids = set()
    candidates = {}  # aid -> base info
    no_new_pages = 0

    try:
        # 1) 목록 후보 모으기 (진행 0~40%)
        for page in range(1, int(max_pages) + 1):
            progress_bar.progress(min(0.40, 0.40 * (page / max_pages)))

            url = build_page_url(page)
            driver.get(url)
            ok = wait_any_links(driver, timeout=12)
            time.sleep(pause)

            if not ok:
                no_new_pages += 1
                if no_new_pages >= stop_no_new_pages:
                    break
                continue

            page_links = collect_links_from_list_page(driver)
            before = len(candidates)

            for aid, (href, title, author) in page_links.items():
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                candidates[aid] = {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "date_raw": f"p{page}",
                    "author": author,
                    "title": title,
                    "title_norm": normalize_title(title),
                    "link": href,
                }

            after = len(candidates)
            if after == before:
                no_new_pages += 1
            else:
                no_new_pages = 0

            if no_new_pages >= stop_no_new_pages:
                break

        # 2) 상세 작성일로 “선택 날짜만” 필터 (진행 40~100%)
        ids = list(candidates.keys())
        kept = []
        total = max(1, len(ids))

        for i, aid in enumerate(ids):
            # 진행바(상세구간)
            progress_bar.progress(0.40 + 0.60 * ((i + 1) / total))

            base = candidates[aid]
            dt = get_article_datetime(driver, base["link"], pause=pause)

            # 작성일을 못 읽으면 안전하게 제외(섞임 방지)
            if not dt:
                continue

            # ✅ 핵심: 선택한 날짜와 정확히 같은 것만
            if dt.date() != target_date:
                continue

            out = dict(base)
            out["date_detail"] = dt.strftime("%Y-%m-%d %H:%M")
            kept.append(out)

        df = pd.DataFrame(kept)
        if not df.empty:
            df = df.drop_duplicates(subset=["link"]).copy()
            df = df.sort_values(by="date_detail", ascending=False)

        return df

    finally:
        try:
            driver.quit()
        except Exception:
            pass


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
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")

with st.expander("설정", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택(선택 날짜만 100% 정확)", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=200, value=30, step=5)
    with c4:
        pause = st.number_input("페이지 대기(초)", min_value=0.03, max_value=1.50, value=0.10, step=0.01)

    stop_no_new_pages = st.number_input("연속 신규 0페이지면 종료", min_value=1, max_value=10, value=2, step=1)

c5, c6 = st.columns([1, 1])
with c5:
    keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
with c6:
    sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

st.divider()

progress = st.progress(0.0)

if "posts" not in st.session_state:
    st.session_state.posts = []

if st.button("수집 시작", use_container_width=True):
    st.session_state.posts = []
    progress.progress(0.0)
    try:
        df = collect_strict_by_date(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            stop_no_new_pages=int(stop_no_new_pages),
            pause=float(pause),
            progress_bar=progress,
        )
        st.session_state.posts = df.to_dict("records")
        progress.progress(1.0)
        st.success(f"수집 완료(선택 날짜만): {len(df)}개")
    except Exception:
        st.error("수집 오류")
        st.code(traceback.format_exc())

df = pd.DataFrame(st.session_state.posts) if st.session_state.posts else pd.DataFrame(
    columns=["date", "date_raw", "date_detail", "author", "title", "title_norm", "link"]
)

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
