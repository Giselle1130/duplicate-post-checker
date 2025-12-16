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


# =========================
# 기본
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


def build_page_url(page: int) -> str:
    return f"{BASE_LIST_URL}&search.page={page}"


def extract_article_id(href: str) -> str:
    m = ARTICLEID_RE.search(href or "")
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


def switch_to_cafe_iframe(driver) -> bool:
    """cafe_main iframe 안에 목록이 있는 경우가 많아서 안전하게 전환"""
    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    # NAME 우선
    try:
        driver.switch_to.frame("cafe_main")
        return True
    except Exception:
        pass

    # ID 시도
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

    # iframe 안
    if switch_to_cafe_iframe(driver):
        try:
            wait.until(has_links)
            return True
        except Exception:
            pass

    # iframe 밖
    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    try:
        wait.until(has_links)
        return True
    except Exception:
        return False


# =========================
# 목록 텍스트에서 날짜/시간 추출
# =========================
RE_TIME = re.compile(r"\b(\d{1,2}:\d{2})\b")
RE_DOTDATE = re.compile(r"\b(20\d{2}\.\d{2}\.\d{2})\.?\b")


def extract_time_token(text: str) -> str:
    m = RE_TIME.search(clean(text))
    return m.group(1) if m else ""


def extract_dot_date(text: str) -> str:
    m = RE_DOTDATE.search(clean(text))
    return m.group(1) if m else ""


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

    # Render/리눅스 안정성
    opts.add_argument("--remote-debugging-port=9222")

    # 이미지 차단
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    # ✅ Selenium Manager 사용 (webdriver_manager 금지)
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(0.2)
    return driver


def collect_from_list_only(
    target_date: date_cls,
    headless: bool,
    max_pages: int = 30,
    pause: float = 0.08,
    stop_if_no_new_pages: int = 2,
    status_cb=None,   # 진행 숫자 출력용 콜백
):
    """
    ✅ 목록에서만 수집
    - 선택 날짜 필터는 목록에 찍힌 날짜/시간 기준으로만 적용
    - 페이지 중복/신규0 페이지가 연속이면 조기 종료
    """
    today = kst_today()
    is_today = (target_date == today)
    target_dot = target_date.strftime("%Y.%m.%d")
    target_iso = target_date.strftime("%Y-%m-%d")

    driver = make_driver(headless=headless)
    seen_ids = set()
    rows_out = []
    no_new_pages = 0

    try:
        for page in range(1, int(max_pages) + 1):
            url = build_page_url(page)

            if status_cb:
                status_cb(page, max_pages, len(seen_ids), url)

            driver.get(url)
            ok = wait_any_links(driver, timeout=12)
            time.sleep(pause)

            if not ok:
                # 링크가 없으면 끝
                no_new_pages += 1
                if no_new_pages >= stop_if_no_new_pages:
                    break
                continue

            # iframe 안/밖 어느 쪽이든 링크 수집
            # (wait_any_links에서 iframe 이동했을 수 있으니, 여기서도 현재 문서에서 바로 읽음)
            links = driver.find_elements(By.CSS_SELECTOR, LINK_CSS)

            page_new = 0
            for a in links:
                try:
                    href = clean(a.get_attribute("href"))
                    if not href:
                        continue
                    if href.startswith("/"):
                        href = urljoin(BASE, href)

                    aid = extract_article_id(href)
                    if not aid or aid in seen_ids:
                        continue

                    # 제목
                    title = clean(a.text) or clean(a.get_attribute("title"))
                    if not title:
                        continue

                    # 행 텍스트(작성자/날짜/시간이 여기에 들어있는 경우가 많음)
                    row_text = ""
                    author = ""
                    date_raw = ""

                    try:
                        parent = a.find_element(By.XPATH, "./ancestor::*[self::tr or self::li][1]")
                        row_text = clean(parent.text)
                        # 날짜/시간 토큰
                        hhmm = extract_time_token(row_text)
                        dot = extract_dot_date(row_text)

                        if is_today:
                            # 오늘: "시간표시"가 있는 행만 통과
                            if not hhmm:
                                continue
                            # 혹시 날짜가 같이 찍히면, 그 날짜가 target이 아니면 제외
                            if dot and dot != target_dot:
                                continue
                            date_raw = hhmm
                        else:
                            # 과거: 날짜표시가 target과 같은 것만 통과
                            if not dot or dot != target_dot:
                                continue
                            # 과거인데 시간만 찍혀있으면(일부 UI) 불확실 → 제외
                            if hhmm:
                                continue
                            date_raw = dot

                        # 작성자(가능하면)
                        parts = [x.strip() for x in row_text.split("\n") if x.strip()]
                        parts = [p for p in parts if p != title]
                        bad = ["조회", "좋아요", "댓글", "댓글수", "공지"]
                        parts = [p for p in parts if not any(b in p for b in bad)]
                        parts = [p for p in parts if not RE_TIME.fullmatch(p)]
                        parts = [p for p in parts if not RE_DOTDATE.fullmatch(p)]
                        author = parts[0] if parts else ""

                    except Exception:
                        # row_text 못 읽으면 필터 불가능 → 제외(섞임 방지)
                        continue

                    seen_ids.add(aid)
                    page_new += 1

                    rows_out.append({
                        "date": target_iso,
                        "date_raw": date_raw,     # 목록에 찍힌 값(시간/날짜)
                        "author": author,
                        "title": title,
                        "title_norm": normalize_title(title),
                        "link": href,
                    })

                except Exception:
                    continue

            if page_new == 0:
                no_new_pages += 1
            else:
                no_new_pages = 0

            if no_new_pages >= stop_if_no_new_pages:
                break

            time.sleep(pause)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    df = pd.DataFrame(rows_out)
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
    return df


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
with st.expander("설정", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=300, value=30, step=5)
    with c4:
        pause = st.number_input("페이지 대기(초)", min_value=0.00, max_value=2.00, value=0.08, step=0.01)

st.divider()

# 진행 텍스트(숫자만)
progress_box = st.empty()

def status_cb(page, max_pages, seen_cnt, url):
    # ✅ 숫자 텍스트만
    progress_box.info(f"진행: {page}/{max_pages} pages | 수집(중복제외): {seen_cnt} | URL: {url}")

if "posts" not in st.session_state:
    st.session_state.posts = []

if st.button("수집 시작", use_container_width=True):
    st.session_state.posts = []
    try:
        df = collect_from_list_only(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            pause=float(pause),
            stop_if_no_new_pages=2,
            status_cb=status_cb,
        )
        st.session_state.posts = df.to_dict("records")
        progress_box.success(f"완료: {len(df)}개 (목록 기준)")
    except Exception:
        progress_box.error("수집 오류")
        st.code(traceback.format_exc())

df = pd.DataFrame(st.session_state.posts) if st.session_state.posts else pd.DataFrame(
    columns=["date", "date_raw", "author", "title", "title_norm", "link"]
)

# 분석 옵션
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
