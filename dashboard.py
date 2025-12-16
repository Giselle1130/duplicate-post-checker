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


def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def kst_today() -> date_cls:
    # 서버시간 기준. 날짜선택 기반으로만 써도 되지만 일단 유지
    return datetime.now().date()


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


def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,900")
    opts.page_load_strategy = "eager"

    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--remote-debugging-port=0")

    # 배포 컨테이너에서 설치한 chromium 경로
    opts.binary_location = "/usr/bin/chromium"

    # 배포 컨테이너에서 설치한 chromedriver 경로
    service = Service(executable_path="/usr/bin/chromedriver")

    driver = webdriver.Chrome(service=service, options=opts)
    driver.implicitly_wait(0.5)
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


def extract_article_id_from_href(href: str) -> str:
    m = ARTICLEID_RE.search(href or "")
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


def collect_by_paging(target_date, headless, max_pages, pause, status_cb=None, prog_cb=None):
    today = kst_today()
    is_today = (target_date == today)
    target_dot = target_date.strftime("%Y.%m.%d")
    target_iso = target_date.strftime("%Y-%m-%d")

    driver = make_driver(headless=headless)
    collected = {}
    try:
        for page in range(1, int(max_pages) + 1):
            if status_cb:
                status_cb(f"페이지 {page}/{max_pages} 수집중…")
            if prog_cb:
                prog_cb(page / float(max_pages))

            driver.get(build_page_url(page))
            wait_list_loaded(driver)
            if pause > 0:
                time.sleep(pause)

            rows = driver.find_elements(By.CSS_SELECTOR, "tr")
            if len(rows) < 5:
                rows = driver.find_elements(By.CSS_SELECTOR, "li") + rows

            page_hit = 0
            oldest_seen = None

            for row in rows:
                try:
                    row_text = clean(row.text)
                    if not row_text or "공지" in row_text:
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

                    title = clean(a.text)
                    if not title:
                        lines = [x.strip() for x in row_text.split("\n") if x.strip()]
                        title = lines[0] if lines else ""
                    if not title:
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
                        d_obj = parse_dot_date(dot)
                        if d_obj:
                            oldest_seen = d_obj if (oldest_seen is None or d_obj < oldest_seen) else oldest_seen
                        date_raw = dot

                    link = f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"
                    collected[link] = {
                        "date": target_iso,
                        "date_raw": date_raw,
                        "author": "",
                        "title": title,
                        "title_norm": normalize_title(title),
                        "link": link,
                    }
                    page_hit += 1
                except Exception:
                    continue

            # 조기 종료
            if (not is_today) and oldest_seen and oldest_seen < target_date:
                break
            if (not is_today) and page >= 3 and page_hit == 0:
                break

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    df = pd.DataFrame(list(collected.values()))
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
        df = df.sort_values(by="date_raw", ascending=False)
    return df


def compute_exact_dups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    return df[df.duplicated(subset=["date", "title_norm"], keep=False)].copy()


def compute_keyword_groups(df: pd.DataFrame, min_count: int = 2):
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
            rows.append({"keyword": kw, "count": len(idxs), "examples": " | ".join(ex)})

    out = pd.DataFrame(rows)
    return out.sort_values(by=["count", "keyword"], ascending=[False, True]) if not out.empty else out


def compute_ai_similar(df: pd.DataFrame, threshold: float = 0.78, max_n: int = 250) -> pd.DataFrame:
    cols = ["title_a", "title_b", "similarity", "link_a", "link_b"]
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=cols)

    df2 = df.head(max_n).copy()
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
st.set_page_config(page_title="클랜/방송/디스코드 중복 게시글 체크", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")

with st.expander("설정", expanded=True):
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", 1, 500, 120, step=10)
    with c4:
        pause = st.number_input("대기(초)", 0.0, 2.0, 0.15, step=0.05)

sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)
keyword_min_count = st.number_input("키워드 중복 최소 건수", 2, 20, 2)

st.divider()

if st.button("수집 시작", use_container_width=True):
    try:
        status = st.empty()
        prog = st.progress(0.0)

        df = collect_by_paging(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            pause=float(pause),
            status_cb=lambda m: status.info(m),
            prog_cb=lambda v: prog.progress(min(max(v, 0.0), 1.0)),
        )

        status.empty()
        prog.empty()

        st.success(f"수집 완료: {len(df)}개")

        tab1, tab2, tab3, tab4 = st.tabs(["📌 원본", "🧨 제목 동일", "🔎 키워드 중복", "🤖 AI 유사"])

        with tab1:
            st.dataframe(df, use_container_width=True)

        with tab2:
            exact = compute_exact_dups(df)
            st.dataframe(exact, use_container_width=True) if not exact.empty else st.info("해당 없음")

        with tab3:
            kw = compute_keyword_groups(df, min_count=int(keyword_min_count))
            st.dataframe(kw, use_container_width=True) if not kw.empty else st.info("해당 없음")

        with tab4:
            sim = compute_ai_similar(df, threshold=float(sim_threshold))
            st.dataframe(sim, use_container_width=True) if not sim.empty else st.info("해당 없음")

    except Exception:
        st.error("수집 오류")
        st.code(traceback.format_exc())
