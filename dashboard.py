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
# Streamlit 기본 설정 (✅ 1번만!)
# =========================
st.set_page_config(page_title="🏰 클랜/방송/디스코드 중복 게시글 체크", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")


# =========================
# 대상 게시판 고정
# =========================
CLUB_ID = 28866679
MENU_ID = 178

# ✅ 안정적인 "클래식 목록"으로 진입
BASE_LIST_URL = (
    "https://cafe.naver.com/ArticleList.nhn"
    f"?search.clubid={CLUB_ID}"
    f"&search.menuid={MENU_ID}"
    "&search.boardtype=L"
)

ARTICLEID_RE = re.compile(r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))")
LINK_CSS = "a[href*='articleid='], a[href*='/articles/']"

# 상세에서 작성일 후보 셀렉터들 (UI가 바뀌어도 버티게 여러 개)
DETAIL_DATE_SELECTORS = [
    "span.date",                 # 구형
    ".article_info .date",
    ".ArticleTopInfo__date",     # 신형
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


def parse_detail_datetime_text(raw: str):
    """
    상세의 작성일 텍스트를 최대한 안전하게 파싱
    예)
    - 2025.12.16. 23:58
    - 2025.12.16. 오후 11:58
    """
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


def get_article_datetime_strict(driver, href: str, pause: float = 0.05):
    """
    ✅ 핵심: 글 상세 들어가서 '진짜 작성일' 얻기
    - iframe 전환 포함
    - 여러 셀렉터 시도 + 소스 정규식 백업
    """
    try:
        driver.get(href)
        time.sleep(pause)

        # 상세도 cafe_main iframe인 경우가 많음
        switch_to_cafe_main_iframe(driver)
        wait = WebDriverWait(driver, 15)

        # 1) 셀렉터로 찾기
        for css in DETAIL_DATE_SELECTORS:
            try:
                el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
                dt = parse_detail_datetime_text(el.text)
                if dt:
                    return dt
            except Exception:
                continue

        # 2) 백업: page_source에서 날짜 패턴 찾기
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


def collect_by_paging(
    target_date: date_cls,
    headless: bool,
    max_pages: int,
    stop_no_match_pages: int,
    pause: float,
):
    """
    ✅ '선택한 날짜만' 100% 보장 버전
    1) 목록에서 후보 글 링크를 모음(속도 위해 최소 조건 적용)
    2) 각 글 상세에 들어가서 실제 작성일을 읽음
    3) target_date와 정확히 같은 글만 최종 수집
    """
    today = kst_today()
    is_today = (target_date == today)
    target_dot = target_date.strftime("%Y.%m.%d")
    target_iso = target_date.strftime("%Y-%m-%d")

    driver = make_driver(headless=headless)

    # 후보 / 최종
    candidates = {}  # link -> base info(제목/작성자/목록표시)
    collected = {}   # link -> final info

    no_match_pages = 0

    try:
        # -------------------------
        # 1) 목록에서 후보 수집
        # -------------------------
        for page in range(1, int(max_pages) + 1):
            driver.get(build_page_url(page))
            wait_list_loaded(driver)
            time.sleep(pause)

            rows = driver.find_elements(By.CSS_SELECTOR, "tr")
            if len(rows) < 5:
                rows = driver.find_elements(By.CSS_SELECTOR, "li") + rows

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

                    # ✅ 후보 최소조건 (속도용)
                    # - 오늘이면 "시간표시"가 있는 것만 후보
                    # - 과거면 "YYYY.MM.DD"가 target과 같은 것만 후보
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

                    if href not in candidates:
                        candidates[href] = {
                            "date": target_iso,           # 최종은 target_iso로 통일
                            "date_raw": date_raw,         # 목록표시(참고용)
                            "author": pick_row_author(row_text, title_raw),
                            "title": title_raw,
                            "title_norm": normalize_title(title_raw),
                            "link": href,
                        }
                        page_candidate += 1

                except Exception:
                    continue

            if page_candidate == 0:
                no_match_pages += 1
            else:
                no_match_pages = 0

            if no_match_pages >= int(stop_no_match_pages):
                break

            time.sleep(pause)

        # -------------------------
        # 2) 상세 들어가서 "진짜 작성일"로 최종 필터
        # -------------------------
        # (여기서부터는 다른 날짜 0개도 섞이면 안되니까 무조건 확인)
        for idx, (href, base) in enumerate(candidates.items(), start=1):
            dt = get_article_datetime_strict(driver, href, pause=pause)

            # 작성일을 못 읽으면 안전하게 버림 (섞이는 것 방지)
            if not dt:
                continue

            if dt.date() != target_date:
                continue

            out = dict(base)
            out["date_detail"] = dt.strftime("%Y-%m-%d %H:%M")
            collected[href] = out

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    df = pd.DataFrame(list(collected.values()))
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
        # detail 시간 기준 정렬
        if "date_detail" in df.columns:
            df = df.sort_values(by="date_detail", ascending=False)
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

    tokens_list = []
    for _, row in df.iterrows():
        tokens_list.append(tokenize(row["title"]))

    inv = {}
    for idx, toks in enumerate(tokens_list):
        for tok in set(toks):
            inv.setdefault(tok, []).append(idx)

    rows = []
    for kw, idxs in inv.items():
        if len(idxs) >= min_count:
            ex = [df.iloc[i]["title"] for i in idxs[:3]]
            rows.append({
                "keyword": kw,
                "count": len(idxs),
                "examples": " | ".join(ex),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["count", "keyword"], ascending=[False, True])


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
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=500, value=120, step=5)
    with c4:
        stop_no_match_pages = st.number_input("연속 0페이지면 종료", min_value=1, max_value=10, value=2, step=1)
    with c5:
        pause = st.number_input("페이지 대기(초)", min_value=0.05, max_value=2.00, value=0.12, step=0.01)

c6, c7 = st.columns([1, 1])
with c6:
    keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
with c7:
    sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

st.divider()

if st.button("수집 시작", use_container_width=True):
    st.session_state.posts = []
    try:
        posts = collect_by_paging(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            stop_no_match_pages=int(stop_no_match_pages),
            pause=float(pause),
        )
        st.session_state.posts = posts
        st.success(f"수집 완료: {len(posts)}개 (✅ 선택한 날짜만)")
    except Exception:
        st.error("수집 오류")
        st.code(traceback.format_exc())

df = (
    pd.DataFrame(st.session_state.posts)
    if "posts" in st.session_state and st.session_state.posts
    else pd.DataFrame(columns=["date", "date_raw", "date_detail", "author", "title", "title_norm", "link"])
)

author_dups = compute_author_dups(df)
exact_dups = compute_exact_dups(df)
keyword_groups = compute_keyword_groups(df, min_count=int(keyword_min_count))
ai_similar = compute_ai_similar(df, threshold=float(sim_threshold))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 원본", "🚨 작성자 동일", "🧨 제목 동일", "🔎 키워드 중복", "🤖 AI 유사"])

with tab1:
    st.dataframe(df, use_container_width=True)

with tab2:
    if author_dups.empty:
        st.info("해당 없음")
    else:
        st.dataframe(author_dups, use_container_width=True)

with tab3:
    if exact_dups.empty:
        st.info("해당 없음")
    else:
        st.dataframe(exact_dups, use_container_width=True)

with tab4:
    if keyword_groups.empty:
        st.info("해당 없음")
    else:
        st.dataframe(keyword_groups, use_container_width=True)

with tab5:
    if ai_similar.empty:
        st.info("해당 없음")
    else:
        st.dataframe(ai_similar, use_container_width=True)
