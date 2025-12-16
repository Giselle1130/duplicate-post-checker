import os
import re
import time
import traceback
from datetime import datetime, date as date_cls
from shutil import which

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait

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

# 글번호 추출: href / onclick / 데이터 다양한 케이스 지원
ARTICLEID_RE = re.compile(
    r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))|(?:articleid[=:]\s*['\"]?(\d+))",
    re.IGNORECASE,
)

# 목록에서 글 링크 후보 (href/onclick/data-articleid 포함)
LINK_CSS = (
    "a[href*='articleid='], a[href*='/articles/'], "
    "a[onclick*='articleid'], a[onclick*='/articles/'], "
    "a[data-articleid]"
)


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


def extract_date_token_any(text: str):
    """
    ✅ 핵심 수정:
    - 2025.12.16 / 2025.12.16. (연도 포함)
    - 12.16 / 12.16. (연도 없음) 모두 인식
    """
    t = clean(text)

    m1 = re.search(r"\b(20\d{2})\.(\d{2})\.(\d{2})\.?\b", t)
    if m1:
        y, mo, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        return date_cls(y, mo, d)

    m2 = re.search(r"\b(\d{2})\.(\d{2})\.?\b", t)
    if m2:
        mo, d = int(m2.group(1)), int(m2.group(2))
        return ("MD", mo, d)

    return None


def build_page_url(page: int) -> str:
    return f"{BASE_LIST_URL}&search.page={page}"


def canonical_article_link(article_id: str) -> str:
    # ✅ 같은 글이면 링크가 섞여도 동일하게 저장(중복 제거 안정)
    return f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"


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

    # 끝 댓글수 제거
    t = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", t)
    t = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", t)

    # [Steam] 같은 태그 제거
    t = re.sub(r"\[[^\]]{1,30}\]", " ", t)

    # LV / 나이/범위 패턴 제거
    t = re.sub(r"\bLv\.?\s*\d+\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}\s*~\s*\d{1,2}\b", " ", t)
    t = re.sub(r"\b\d{1,2}\s*세\b", " ", t)

    # url 제거
    t = re.sub(r"https?://\S+", " ", t)

    # 이모지/기호 제거 (한/영/숫자/공백만 유지)
    t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)

    # 숫자 단독 제거
    t = re.sub(r"\b\d+\b", " ", t)

    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def tokenize(text: str):
    t = normalize_title(text)
    toks = re.findall(r"[a-z]+|[가-힣]+", t)
    toks = [x for x in toks if len(x) >= 2]
    toks = [x for x in toks if x not in STOPWORDS]
    return toks


# =========================
# Chrome/Chromium 바이너리 탐지
# =========================
def _find_chrome_binary():
    # 1) 사용자가 환경변수로 지정한 경우
    env = os.environ.get("CHROME_BIN") or os.environ.get("GOOGLE_CHROME_BIN")
    if env and os.path.exists(env):
        return env

    # 2) PATH 탐색
    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
        p = which(name)
        if p:
            return p

    # 3) 흔한 설치 경로
    candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# =========================
# Selenium
# =========================
def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")

    # ✅ 속도/안정
    opts.page_load_strategy = "eager"

    if headless:
        opts.add_argument("--headless=new")
        # 일부 리눅스/WSL에서 도움이 되는 경우가 있음
        opts.add_argument("--remote-debugging-port=0")

    # ✅ 이미지 차단(속도↑)
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    # ✅ 자동화 탐지 완화(가능한 범위)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    # ✅ 크롬/크로미움 경로 지정(없으면 드라이버가 바로 죽는 환경이 많음)
    chrome_bin = _find_chrome_binary()
    if chrome_bin:
        opts.binary_location = chrome_bin

    try:
        if USE_WDM:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=opts)
        else:
            driver = webdriver.Chrome(options=opts)
    except Exception as e:
        msg = str(e)
        hint = [
            "크롬/크로미움 브라우저가 설치되어 있는지 확인해줘.",
            "Ubuntu/WSL이면: sudo apt update && sudo apt install -y google-chrome-stable (또는 chromium-browser)",
            "설치되어 있는데도 실패하면 CHROME_BIN 환경변수로 크롬 경로를 지정해줘. 예) export CHROME_BIN=/usr/bin/google-chrome",
        ]
        raise RuntimeError("ChromeDriver 실행 실패\n\n원인:\n" + msg + "\n\n해결:\n- " + "\n- ".join(hint))

    driver.implicitly_wait(0.5)

    # navigator.webdriver 숨김(가능한 범위)
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
        )
    except Exception:
        pass

    return driver


def switch_to_cafe_main_iframe(driver) -> bool:
    # 네이버 카페(클래식)는 cafe_main iframe에 목록이 뜨는 경우가 많음
    try:
        driver.switch_to.default_content()
        if driver.find_elements(By.ID, "cafe_main"):
            driver.switch_to.frame("cafe_main")
            return True
    except Exception:
        pass
    return False


def wait_list_loaded(driver):
    """
    ✅ 핵심: '글 링크'가 실제로 생길 때까지 기다림.
    - iframe 안/밖 모두 체크
    - link 패턴 다양한 케이스 허용
    """
    wait = WebDriverWait(driver, 25)

    def has_links_in_current_doc(d):
        return len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0

    # 1) iframe 먼저
    if switch_to_cafe_main_iframe(driver):
        try:
            wait.until(has_links_in_current_doc)
            return
        except Exception:
            pass

    # 2) default content에서 다시 체크
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
    # 날짜 토큰(연도 포함/미포함) 제거
    lines = [x for x in lines if not re.fullmatch(r"(20\d{2}\.\d{2}\.\d{2}\.?)|(\d{2}\.\d{2}\.?)", x)]
    bad = ["조회", "좋아요", "댓글", "댓글수"]
    lines = [x for x in lines if not any(b in x for b in bad)]
    lines = [x for x in lines if x != "공지"]
    for x in lines:
        if 1 <= len(x) <= 30:
            return x
    return ""


def extract_article_id(el) -> str:
    """
    ✅ href가 비어있거나 javascript인 경우도 커버:
    - data-articleid
    - href
    - onclick
    """
    try:
        da = clean(el.get_attribute("data-articleid"))
        if da.isdigit():
            return da
    except Exception:
        pass

    try:
        href = clean(el.get_attribute("href"))
        if href:
            m = ARTICLEID_RE.search(href)
            if m:
                return (m.group(1) or m.group(2) or m.group(3) or "").strip()
    except Exception:
        pass

    try:
        onclick = clean(el.get_attribute("onclick"))
        if onclick:
            m = ARTICLEID_RE.search(onclick)
            if m:
                return (m.group(1) or m.group(2) or m.group(3) or "").strip()
    except Exception:
        pass

    return ""


def collect_by_paging(
    target_date: date_cls,
    headless: bool,
    max_pages: int,
    stop_no_match_pages: int,
    pause: float,
):
    today = kst_today()
    is_today = (target_date == today)

    driver = make_driver(headless=headless)
    collected = {}
    no_match_pages = 0

    try:
        for page in range(1, int(max_pages) + 1):
            driver.get(build_page_url(page))

            wait_list_loaded(driver)
            time.sleep(pause)

            # ✅ table/tr 우선, 없으면 li도 보조
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

                    # 글번호 확보(핵심)
                    article_id = extract_article_id(a)
                    if not article_id:
                        article_id = extract_article_id(row)
                    if not article_id:
                        continue

                    # 제목
                    title_raw = clean(a.text)
                    if not title_raw:
                        lines = [x.strip() for x in row_text.split("\n") if x.strip()]
                        title_raw = lines[0] if lines else ""
                    if not title_raw:
                        continue

                    hhmm = extract_time_token(row_text)
                    dtok = extract_date_token_any(row_text)

                    if is_today:
                        # 오늘: 시간형만 수집
                        if not hhmm:
                            continue
                        date_raw = hhmm
                    else:
                        # 과거: 날짜형만 수집(연도 포함/미포함 모두 허용)
                        if hhmm:
                            continue

                        ok = False
                        date_raw = ""

                        if isinstance(dtok, date_cls):
                            ok = (dtok == target_date)
                            date_raw = dtok.strftime("%Y.%m.%d")
                        elif isinstance(dtok, tuple) and dtok[0] == "MD":
                            _, mo, d = dtok
                            try:
                                d_obj = date_cls(target_date.year, mo, d)
                                ok = (d_obj == target_date)
                                date_raw = d_obj.strftime("%Y.%m.%d")
                            except Exception:
                                ok = False

                        if not ok:
                            continue

                    link = canonical_article_link(article_id)

                    collected[link] = {
                        "date": target_date.strftime("%Y-%m-%d"),
                        "date_raw": date_raw,
                        "author": pick_row_author(row_text, title_raw),
                        "title": title_raw,
                        "title_norm": normalize_title(title_raw),
                        "link": link,
                    }
                    page_matches += 1

                except Exception:
                    continue

            if page_matches > 0:
                no_match_pages = 0
            else:
                no_match_pages += 1

            # ✅ 조기 종료
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
# 중복/유사 (캐시로 렉 완화)
# =========================
@st.cache_data(show_spinner=False)
def compute_author_dups_cached(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "author", "count"])
    a = df.groupby(["date", "author"]).size().reset_index(name="count")
    return a[a["count"] >= 2].sort_values(by="count", ascending=False)


@st.cache_data(show_spinner=False)
def compute_exact_dups_cached(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    return df[df.duplicated(subset=["date", "title_norm"], keep=False)].copy()


@st.cache_data(show_spinner=False)
def compute_keyword_groups_cached(df: pd.DataFrame, min_count: int = 2):
    """
    ✅ 같은 키워드가 2건 이상이면 중복으로 잡기(작성자 무관)
    """
    if df.empty:
        return pd.DataFrame(columns=["keyword", "count", "examples"])

    tokens_list = [tokenize(x) for x in df["title"].fillna("").astype(str).tolist()]

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


@st.cache_data(show_spinner=False)
def compute_ai_similar_cached(df: pd.DataFrame, threshold: float = 0.78, max_n: int = 250) -> pd.DataFrame:
    """
    ✅ 작성자 조건 제거: 선택 날짜 전체에서 유사도 비교
    ✅ 렉 방지: max_n개만 비교(최신순 head)
    """
    cols = ["title_a", "title_b", "similarity", "link_a", "link_b"]
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=cols)

    df2 = df.copy()
    if len(df2) > max_n:
        df2 = df2.head(max_n).copy()

    titles_raw = df2["title"].fillna("").astype(str).tolist()
    titles = df2["title_norm"].fillna("").astype(str).tolist()
    links = df2["link"].fillna("").astype(str).tolist()

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
st.set_page_config(page_title="menu=178 수집/중복", layout="wide")
st.title("🏰┃클랜/방송/디스코드(menu=178)")

with st.expander("설정", expanded=True):
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=500, value=120, step=5)
    with c4:
        stop_no_match_pages = st.number_input("연속 0페이지면 종료", min_value=1, max_value=10, value=2, step=1)
    with c5:
        pause = st.number_input("페이지 대기(초)", min_value=0.05, max_value=2.00, value=0.15, step=0.05)
    with c6:
        run_ai = st.checkbox("🤖 AI 유사도 계산(무거움)", value=False)

c7, c8, c9 = st.columns([1, 1, 1])
with c7:
    keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
with c8:
    sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)
with c9:
    ai_max_n = st.number_input("AI 비교 최대 글 수", min_value=50, max_value=800, value=250, step=50)

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
        st.success(f"수집 완료: {len(posts)}개")
    except Exception as e:
        st.error("수집 오류")
        st.code(str(e))
        st.code(traceback.format_exc())

df = (
    pd.DataFrame(st.session_state.posts)
    if "posts" in st.session_state and st.session_state.posts
    else pd.DataFrame(columns=["date", "date_raw", "author", "title", "title_norm", "link"])
)

author_dups = compute_author_dups_cached(df)
exact_dups = compute_exact_dups_cached(df)
keyword_groups = compute_keyword_groups_cached(df, min_count=int(keyword_min_count))

ai_similar = pd.DataFrame(columns=["title_a", "title_b", "similarity", "link_a", "link_b"])
if run_ai:
    ai_similar = compute_ai_similar_cached(df, threshold=float(sim_threshold), max_n=int(ai_max_n))

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
    if not run_ai:
        st.info("AI 유사도는 무거워서 기본 OFF야. 위 설정에서 체크하면 계산해.")
    if ai_similar.empty:
        st.info("해당 없음")
    else:
        st.dataframe(ai_similar, use_container_width=True)
