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
from selenium.webdriver.support.ui import WebDriverWait

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


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


def parse_dot_date(s: str):
    try:
        return datetime.strptime(s, "%Y.%m.%d").date()
    except Exception:
        return None


def build_page_url(page: int) -> str:
    return f"{BASE_LIST_URL}&search.page={page}"


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
# Selenium (✅ 플랫폼(Render) 호환: webdriver_manager 제거)
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

    # Render/리눅스에서 안정성
    opts.add_argument("--remote-debugging-port=9222")

    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    # ✅ Selenium Manager 사용 (Chrome 버전에 맞는 드라이버 자동)
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(0.5)

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
    wait = WebDriverWait(driver, 25)

    def has_links_in_current_doc(d):
        return len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0

    if switch_to_cafe_main_iframe(driver):
        try:
            wait.until(has_links_in_current_doc)
            return
        except Exception:
            pass

    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    wait.until(has_links_in_current_doc)


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


def collect_by_paging(
    target_date: date_cls,
    headless: bool,
    max_pages: int,
    stop_no_match_pages: int,
    pause: float,
    progress_bar=None,   # ✅ 진행바만 업데이트
):
    today = kst_today()
    is_today = (target_date == today)
    target_dot = target_date.strftime("%Y.%m.%d")
    target_iso = target_date.strftime("%Y-%m-%d")

    driver = make_driver(headless=headless)
    collected = {}
    no_match_pages = 0

    try:
        for page in range(1, int(max_pages) + 1):
            if progress_bar is not None:
                # 0~1 범위
                progress_bar.progress(min(1.0, page / max_pages))

            driver.get(build_page_url(page))

            wait_list_loaded(driver)
            time.sleep(pause)

            rows = driver.find_elements(By.CSS_SELECTOR, "tr")
            if len(rows) < 5:
                rows = driver.find_elements(By.CSS_SELECTOR, "li") + rows

            page_matches = 0

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

                    collected[href] = {
                        "date": target_iso,
                        "date_raw": date_raw,
                        "author": pick_row_author(row_text, title_raw),
                        "title": title_raw,
                        "title_norm": normalize_title(title_raw),
                        "link": href,
                    }
                    page_matches += 1

                except Exception:
                    continue

            if page_matches > 0:
                no_match_pages = 0
            else:
                no_match_pages += 1

            if no_match_pages >= int(stop_no_match_pages):
                break

            time.sleep(pause)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    df = pd.DataFrame(list(collected.values()))
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
        df = df.sort_values(by="date_raw", ascending=False)
    return df.to_dict("records")


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

    tokens_list = [tokenize(t) for t in df["title"]]
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
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        # ✅ 기본 30
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=500, value=30, step=5)
    with c4:
        stop_no_match_pages = st.number_input("연속 0페이지면 종료", min_value=1, max_value=10, value=2, step=1)
    with c5:
        pause = st.number_input("페이지 대기(초)", min_value=0.05, max_value=2.00, value=0.15, step=0.05)

c6, c7 = st.columns([1, 1])
with c6:
    keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
with c7:
    sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

st.divider()

# ✅ 진행바만
progress_bar = st.progress(0)

if "posts" not in st.session_state:
    st.session_state.posts = []

if st.button("수집 시작", use_container_width=True):
    st.session_state.posts = []
    progress_bar.progress(0)
    try:
        posts = collect_by_paging(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            stop_no_match_pages=int(stop_no_match_pages),
            pause=float(pause),
            progress_bar=progress_bar,
        )
        st.session_state.posts = posts
        progress_bar.progress(1.0)
        st.success(f"수집 완료: {len(posts)}개")
    except Exception:
        st.error("수집 오류")
        st.code(traceback.format_exc())

df = (
    pd.DataFrame(st.session_state.posts)
    if st.session_state.posts
    else pd.DataFrame(columns=["date", "date_raw", "author", "title", "title_norm", "link"])
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
