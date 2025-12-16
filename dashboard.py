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

# 링크는 케이스가 섞여서 둘 다 지원:
ARTICLEID_RE = re.compile(r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))")

# 목록에서 글 링크를 찾는 CSS
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


def extract_date_token_any(text: str):
    """
    목록 날짜 표기가 3종류로 나올 수 있어서 모두 대응:
      - 2025.12.16
      - 12.16
      - 2025.12.16. (끝 점)
    return: ("YMD", "2025.12.16") or ("MD", "12.16") or ("", "")
    """
    t = clean(text)
    m1 = re.search(r"\b(20\d{2}\.\d{2}\.\d{2})\.?\b", t)
    if m1:
        return ("YMD", m1.group(1))
    m2 = re.search(r"\b(\d{2}\.\d{2})\b", t)
    if m2:
        return ("MD", m2.group(1))
    return ("", "")


def parse_dot_ymd(s: str):
    try:
        return datetime.strptime(s, "%Y.%m.%d").date()
    except Exception:
        return None


def parse_dot_md(s: str, year: int):
    try:
        mm, dd = s.split(".")
        return date_cls(year, int(mm), int(dd))
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
# Selenium
# =========================
def guess_chrome_binary():
    # Render/Ubuntu에서 흔한 경로들
    candidates = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ]
    return candidates


def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()

    # ✅ 안정 옵션
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--lang=ko-KR")

    # ✅ 더 안정적으로 (Render에서 유용)
    opts.add_argument("--disable-background-networking")
    opts.add_argument("--disable-background-timer-throttling")
    opts.add_argument("--disable-renderer-backgrounding")
    opts.add_argument("--disable-features=Translate,BackForwardCache,AcceptCHFrame")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-notifications")

    # ✅ 속도/안정
    opts.page_load_strategy = "eager"

    if headless:
        opts.add_argument("--headless=new")

    # ✅ 이미지 차단(속도↑)
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    # ✅ 자동화 탐지 완화(가능한 범위)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    # ✅ 크롬 바이너리 경로 지정(있으면)
    for p in guess_chrome_binary():
        try:
            import os
            if os.path.exists(p):
                opts.binary_location = p
                break
        except Exception:
            pass

    # Selenium Manager가 드라이버를 알아서 맞춰줌(온라인 환경에서 안정)
    service = Service()
    driver = webdriver.Chrome(service=service, options=opts)
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
    # 날짜형 제거(둘 다)
    lines = [x for x in lines if not re.fullmatch(r"20\d{2}\.\d{2}\.\d{2}\.?", x)]
    lines = [x for x in lines if not re.fullmatch(r"\d{2}\.\d{2}", x)]
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


# =========================
# "한 페이지"만 수집 (중요: 긴 작업을 끊어서 실행)
# =========================
def collect_one_page(driver, target_date: date_cls, page: int, pause: float):
    """
    return:
      collected_dict (href -> row dict),
      page_matches (int),
      saw_any_row (bool)
    """
    today = kst_today()
    is_today = (target_date == today)
    target_iso = target_date.strftime("%Y-%m-%d")

    driver.get(build_page_url(page))
    wait_list_loaded(driver)
    time.sleep(pause)

    rows = driver.find_elements(By.CSS_SELECTOR, "tr")
    if len(rows) < 5:
        rows = driver.find_elements(By.CSS_SELECTOR, "li") + rows

    collected = {}
    page_matches = 0
    saw_any = False

    for row in rows:
        try:
            row_text = clean(row.text)
            if not row_text:
                continue
            saw_any = True
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

            # 제목
            title_raw = clean(a.text)
            if not title_raw:
                lines = [x.strip() for x in row_text.split("\n") if x.strip()]
                title_raw = lines[0] if lines else ""
            if not title_raw:
                continue

            hhmm = extract_time_token(row_text)
            dtype, dtoken = extract_date_token_any(row_text)

            # ✅ 날짜 매칭 로직 (오늘=시간, 과거=날짜(YYYY.MM.DD 또는 MM.DD))
            if is_today:
                if not hhmm:
                    continue
                date_raw = hhmm
            else:
                # 과거인데 시간만 있는 경우도 가끔 있음(최근글)
                # → MM.DD 표기까지 받아서 target_date와 매칭되면 통과
                d_obj = None
                if dtype == "YMD":
                    d_obj = parse_dot_ymd(dtoken)
                elif dtype == "MD":
                    d_obj = parse_dot_md(dtoken, target_date.year)

                if d_obj != target_date:
                    continue

                date_raw = dtoken

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

    return collected, page_matches, saw_any


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

    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xc = vec_c.fit_transform(titles)

    Mw = cosine_similarity(Xw)
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
# Streamlit (안 튕기게 구조)
# =========================
st.set_page_config(page_title="menu=178 수집/중복", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크 (menu=178)")

# ---- session init
if "running" not in st.session_state:
    st.session_state.running = False
if "driver" not in st.session_state:
    st.session_state.driver = None
if "collected" not in st.session_state:
    st.session_state.collected = {}   # href -> row dict
if "page" not in st.session_state:
    st.session_state.page = 1
if "no_match_pages" not in st.session_state:
    st.session_state.no_match_pages = 0
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "debug_events" not in st.session_state:
    st.session_state.debug_events = []


def debug_log(msg: str):
    st.session_state.debug_events.append(f"{datetime.now().strftime('%H:%M:%S')}  {msg}")
    st.session_state.debug_events = st.session_state.debug_events[-300:]


def stop_and_cleanup():
    st.session_state.running = False
    st.session_state.no_match_pages = 0
    st.session_state.page = 1
    # driver는 원하면 유지 가능하지만, 안정적으로는 닫는 편이 좋음
    try:
        if st.session_state.driver is not None:
            st.session_state.driver.quit()
    except Exception:
        pass
    st.session_state.driver = None


with st.expander("설정", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.2, 1.2, 1.2])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=500, value=120, step=5)
    with c4:
        stop_no_match_pages = st.number_input("연속 0페이지면 종료", min_value=1, max_value=10, value=3, step=1)
    with c5:
        pause = st.number_input("페이지 대기(초)", min_value=0.05, max_value=2.00, value=0.25, step=0.05)

    c6, c7, c8 = st.columns([1.2, 1.2, 1.2])
    with c6:
        pages_per_tick = st.number_input("한 번에 처리할 페이지(권장 1~3)", min_value=1, max_value=10, value=2, step=1)
    with c7:
        keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
    with c8:
        sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

st.divider()

btn1, btn2, btn3 = st.columns([1, 1, 1])
with btn1:
    start = st.button("수집 시작", use_container_width=True, disabled=st.session_state.running)
with btn2:
    stop = st.button("중지", use_container_width=True, disabled=not st.session_state.running)
with btn3:
    reset = st.button("초기화(데이터 삭제)", use_container_width=True)

if reset:
    stop_and_cleanup()
    st.session_state.collected = {}
    st.session_state.last_error = ""
    st.session_state.debug_events = []
    st.success("초기화 완료")

if stop:
    stop_and_cleanup()
    st.warning("중지됨")

if start:
    # 새 실행
    stop_and_cleanup()
    st.session_state.collected = {}
    st.session_state.last_error = ""
    st.session_state.running = True
    debug_log("START pressed")


# ---- Running loop (짧게 끊어서 실행)
progress_box = st.empty()
status_box = st.empty()

if st.session_state.running:
    try:
        if st.session_state.driver is None:
            debug_log("Creating driver...")
            st.session_state.driver = make_driver(headless=headless)
            debug_log("Driver created.")

        # 이번 tick에 몇 페이지 처리
        pages_done = 0
        tick_start = time.time()

        while pages_done < int(pages_per_tick) and st.session_state.page <= int(max_pages):
            p = st.session_state.page
            progress_box.info(f"수집 중... page={p} / max={int(max_pages)}  (현재 수집 {len(st.session_state.collected)}개)")
            debug_log(f"Collecting page {p}")

            collected, page_matches, saw_any = collect_one_page(
                st.session_state.driver, target_date=target_date, page=p, pause=float(pause)
            )

            # 병합
            for k, v in collected.items():
                st.session_state.collected[k] = v

            if page_matches > 0:
                st.session_state.no_match_pages = 0
            else:
                st.session_state.no_match_pages += 1

            # 조기 종료 조건
            if st.session_state.no_match_pages >= int(stop_no_match_pages):
                debug_log("Stop condition met: consecutive no-match pages")
                st.session_state.running = False
                break

            st.session_state.page += 1
            pages_done += 1

            # 너무 오래 붙잡지 않기(세션 리셋 방지)
            if time.time() - tick_start > 12:
                debug_log("Tick time budget reached, yielding to Streamlit rerun")
                break

        # 종료 조건: 최대 페이지 도달
        if st.session_state.page > int(max_pages):
            debug_log("Reached max_pages. Stopping.")
            st.session_state.running = False

        # 아직 running이면 자동으로 다음 tick 진행
        if st.session_state.running:
            status_box.warning("계속 수집 중... 잠시 후 자동으로 다음 페이지로 진행합니다.")
            time.sleep(0.2)
            st.rerun()
        else:
            # 끝났으면 드라이버 정리
            try:
                if st.session_state.driver is not None:
                    st.session_state.driver.quit()
            except Exception:
                pass
            st.session_state.driver = None
            status_box.success(f"수집 완료: {len(st.session_state.collected)}개")

    except Exception:
        st.session_state.last_error = traceback.format_exc()
        debug_log("ERROR: " + st.session_state.last_error.splitlines()[-1] if st.session_state.last_error else "ERROR")
        st.session_state.running = False
        try:
            if st.session_state.driver is not None:
                st.session_state.driver.quit()
        except Exception:
            pass
        st.session_state.driver = None
        st.error("수집 오류")
        st.code(st.session_state.last_error)

# ---- DataFrame
df = pd.DataFrame(list(st.session_state.collected.values()))
if not df.empty:
    df = df.drop_duplicates(subset=["link"]).copy()
    # date_raw가 시간/날짜 혼합이라 정렬은 문자열 기준
    df = df.sort_values(by="date_raw", ascending=False)

author_dups = compute_author_dups(df)
exact_dups = compute_exact_dups(df)
keyword_groups = compute_keyword_groups(df, min_count=int(keyword_min_count))
ai_similar = compute_ai_similar(df, threshold=float(sim_threshold))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📌 원본", "🚨 작성자 동일", "🧨 제목 동일", "🔎 키워드 중복", "🤖 AI 유사", "🧪 디버그"])

with tab1:
    st.dataframe(df, use_container_width=True)

with tab2:
    st.dataframe(author_dups if not author_dups.empty else pd.DataFrame(), use_container_width=True)
    if author_dups.empty:
        st.info("해당 없음")

with tab3:
    st.dataframe(exact_dups if not exact_dups.empty else pd.DataFrame(), use_container_width=True)
    if exact_dups.empty:
        st.info("해당 없음")

with tab4:
    st.dataframe(keyword_groups if not keyword_groups.empty else pd.DataFrame(), use_container_width=True)
    if keyword_groups.empty:
        st.info("해당 없음")

with tab5:
    st.dataframe(ai_similar if not ai_similar.empty else pd.DataFrame(), use_container_width=True)
    if ai_similar.empty:
        st.info("해당 없음")

with tab6:
    st.write("최근 디버그 이벤트(최대 300줄):")
    st.code("\n".join(st.session_state.debug_events[-300:]) if st.session_state.debug_events else "(없음)")
    if st.session_state.last_error:
        st.write("마지막 오류:")
        st.code(st.session_state.last_error)
