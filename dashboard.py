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
BASE = "https://cafe.naver.com"

# 클래식 목록
BASE_LIST_URL = (
    "https://cafe.naver.com/ArticleList.nhn"
    f"?search.clubid={CLUB_ID}"
    f"&search.menuid={MENU_ID}"
    "&search.boardtype=L"
)

# 글 링크 패턴 (둘 다)
LINK_CSS = "a[href*='articleid='], a[href*='/articles/']"
ARTICLEID_RE = re.compile(r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))")

# 상세에서 작성일 후보 셀렉터 (UI 변화 대비)
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


def build_page_url(page: int) -> str:
    return f"{BASE_LIST_URL}&search.page={page}"


def extract_article_id(href: str) -> str:
    m = ARTICLEID_RE.search(href or "")
    if not m:
        return ""
    return m.group(1) or m.group(2) or ""


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
# Selenium (Selenium Manager)
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

    # 리눅스/Render 안정성
    opts.add_argument("--remote-debugging-port=9222")

    # 이미지 차단
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    driver = webdriver.Chrome(options=opts)  # ✅ Selenium Manager
    driver.implicitly_wait(0.2)
    return driver


def switch_to_cafe_iframe(driver) -> bool:
    """
    cafe_main iframe이 ID/NAME 형태로 섞여 나와서 둘 다 시도.
    """
    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    # 1) NAME
    try:
        WebDriverWait(driver, 2).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "cafe_main")))
        return True
    except Exception:
        pass

    # 2) ID
    try:
        driver.switch_to.default_content()
        iframes = driver.find_elements(By.ID, "cafe_main")
        if iframes:
            driver.switch_to.frame("cafe_main")
            return True
    except Exception:
        pass

    return False


def wait_any_links(driver, timeout=12):
    """
    iframe 안/밖에서 글 링크가 생길 때까지 기다림.
    """
    wait = WebDriverWait(driver, timeout)

    def has_links_in_current(d):
        return len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0

    # iframe 먼저
    if switch_to_cafe_iframe(driver):
        try:
            wait.until(has_links_in_current)
            return True
        except Exception:
            pass

    # 밖에서 다시
    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    try:
        wait.until(has_links_in_current)
        return True
    except Exception:
        return False


def collect_links_from_current_page(driver):
    """
    목록 페이지에서 글 링크들을 최대한 단순/강하게 수집.
    row(tr/li) 구조 안 믿고 a 링크만 싹 긁음.
    """
    links = driver.find_elements(By.CSS_SELECTOR, LINK_CSS)
    out = []

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

            title = clean(a.text)
            if not title:
                # title이 a 밖에 있을 수 있음: 주변 텍스트로 보조
                title = clean(a.get_attribute("title"))

            # 작성자 추출은 환경마다 너무 흔들려서 "있으면 가져오고 없으면 공백"
            author = ""
            try:
                # 근처에 nickname 같은게 있으면 읽기
                parent = a.find_element(By.XPATH, "./ancestor::*[self::tr or self::li][1]")
                cand = parent.text.split("\n")
                cand = [c.strip() for c in cand if c.strip()]
                # 제목 제외하고 짧은 텍스트 중 1개 골라보기
                cand2 = [c for c in cand if c != title and len(c) <= 30 and "조회" not in c and "댓글" not in c]
                author = cand2[0] if cand2 else ""
            except Exception:
                author = ""

            out.append((aid, href, title, author))
        except Exception:
            continue

    # articleid 기준 unique
    uniq = {}
    for aid, href, title, author in out:
        if aid not in uniq:
            uniq[aid] = (href, title, author)
    return uniq


def get_article_datetime_strict(driver, href: str, pause: float = 0.05):
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

        # 최후의 수단: 소스에서 날짜 패턴 찾기
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
# 진행/중지/디버그 상태머신
# =========================
def ensure_state():
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("phase", "idle")  # idle | collect | validate | done
    ss.setdefault("driver", None)
    ss.setdefault("logs", [])
    ss.setdefault("page", 1)
    ss.setdefault("seen_ids", set())      # 페이지 중복 감지용
    ss.setdefault("candidates", {})       # articleid -> info
    ss.setdefault("validate_ids", [])
    ss.setdefault("validate_i", 0)
    ss.setdefault("collected", {})        # articleid -> info
    ss.setdefault("last_url", "")
    ss.setdefault("posts", [])


def log(msg: str):
    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def reset_job():
    ss = st.session_state
    try:
        if ss.driver is not None:
            ss.driver.quit()
    except Exception:
        pass

    ss.running = False
    ss.phase = "idle"
    ss.driver = None
    ss.logs = []
    ss.page = 1
    ss.seen_ids = set()
    ss.candidates = {}
    ss.validate_ids = []
    ss.validate_i = 0
    ss.collected = {}
    ss.last_url = ""
    ss.posts = []


def start_job(target_date: date_cls, headless: bool, max_pages: int, pause: float):
    reset_job()
    ss = st.session_state
    ss.target_date = target_date
    ss.headless = headless
    ss.max_pages = int(max_pages)
    ss.pause = float(pause)

    ss.driver = make_driver(headless=headless)
    ss.phase = "collect"
    ss.running = True
    log(f"시작: target_date={target_date} max_pages={max_pages}")


def stop_job():
    st.session_state.running = False
    log("중지(사용자 요청)")


def finalize_job():
    ss = st.session_state
    df = pd.DataFrame(list(ss.collected.values()))
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
        df = df.sort_values(by="date_detail", ascending=False)
    ss.posts = df.to_dict("records")
    ss.phase = "done"
    ss.running = False
    log(f"완료: 최종 {len(ss.posts)}개")


def step_collect():
    """
    목록은 '페이지 1~max_pages'까지만,
    page2부터 새 글이 하나도 안 늘어나면 즉시 종료.
    """
    ss = st.session_state
    d = ss.driver

    pages_per_step = int(ss.pages_per_step)
    processed = 0

    while ss.page <= ss.max_pages and processed < pages_per_step and ss.running:
        url = build_page_url(ss.page)
        ss.last_url = url
        log(f"[목록] page={ss.page}")

        try:
            d.get(url)
            ok = wait_any_links(d, timeout=12)
            time.sleep(ss.pause)

            if not ok:
                # 링크가 아예 안 뜨면 바로 종료(더 돌 의미 없음)
                log("목록에서 링크를 찾지 못함 → 종료")
                ss.page = ss.max_pages + 1
                break

            page_links = collect_links_from_current_page(d)
            page_ids = set(page_links.keys())

            # page2부터 "완전히 같은 목록"이면 더 돌 필요 없음
            if ss.page >= 2 and page_ids and page_ids.issubset(ss.seen_ids):
                log("새 글이 더 이상 없음(중복 페이지) → 종료")
                ss.page = ss.max_pages + 1
                break

            before = len(ss.candidates)
            for aid, (href, title, author) in page_links.items():
                ss.seen_ids.add(aid)
                if aid not in ss.candidates:
                    ss.candidates[aid] = {
                        "date": ss.target_date.strftime("%Y-%m-%d"),
                        "date_raw": f"page={ss.page}",
                        "author": author,
                        "title": title,
                        "title_norm": normalize_title(title),
                        "link": href,
                    }
            after = len(ss.candidates)
            log(f"후보 누적: {after} (이번 페이지 신규 {after - before})")

            # page2부터 신규가 0이면 종료 (무야 상황: 글 20개 미만 + 1페이지)
            if ss.page >= 2 and (after - before) == 0:
                log("page>=2 신규 0 → 종료")
                ss.page = ss.max_pages + 1
                break

        except Exception as e:
            log(f"목록 오류: {type(e).__name__}: {e}")
            ss.page = ss.max_pages + 1
            break

        ss.page += 1
        processed += 1

    # 목록 단계 끝 → 상세 검증
    if ss.page > ss.max_pages:
        ss.validate_ids = list(ss.candidates.keys())
        ss.validate_i = 0
        ss.phase = "validate"
        log(f"상세 검증 시작: 후보 {len(ss.validate_ids)}개")


def step_validate():
    """
    후보(대개 20개 미만)만 상세 진입해서 작성일 확인.
    선택 날짜와 정확히 같은 글만 통과.
    """
    ss = st.session_state
    d = ss.driver

    per_step = int(ss.articles_per_step)
    processed = 0

    while ss.validate_i < len(ss.validate_ids) and processed < per_step and ss.running:
        aid = ss.validate_ids[ss.validate_i]
        base = ss.candidates.get(aid)
        ss.validate_i += 1
        processed += 1
        if not base:
            continue

        href = base["link"]
        ss.last_url = href

        dt = get_article_datetime_strict(d, href, pause=ss.pause)

        # 작성일 못 읽으면 안전하게 버림 (다른날짜 섞임 방지)
        if not dt:
            continue

        if dt.date() != ss.target_date:
            continue

        out = dict(base)
        out["date_detail"] = dt.strftime("%Y-%m-%d %H:%M")
        ss.collected[aid] = out

    if ss.validate_i >= len(ss.validate_ids):
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
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택(✅ 이 날짜만)", value=kst_today())
    with c2:
        headless = st.checkbox("헤드리스", value=True)
    with c3:
        # 무야 상황(20개 미만/1페이지) 기준: 크게 돌 필요 없음
        max_pages = st.number_input("최대 페이지(추천 3~5)", min_value=1, max_value=30, value=5, step=1)
    with c4:
        pause = st.number_input("대기(초)", min_value=0.00, max_value=1.00, value=0.08, step=0.01)

    c5, c6, c7 = st.columns([1, 1, 1])
    with c5:
        pages_per_step = st.number_input("한 번에 목록 페이지 처리", min_value=1, max_value=5, value=1, step=1)
    with c6:
        articles_per_step = st.number_input("한 번에 상세 검증", min_value=1, max_value=30, value=10, step=1)
    with c7:
        auto_run = st.checkbox("자동 진행(켜면 계속)", value=True)

    st.session_state.pages_per_step = int(pages_per_step)
    st.session_state.articles_per_step = int(articles_per_step)

st.divider()

b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
with b1:
    if st.button("▶ 시작", use_container_width=True):
        try:
            start_job(target_date, headless, int(max_pages), float(pause))
            st.rerun()
        except Exception:
            st.error("시작 오류")
            st.code(traceback.format_exc())

with b2:
    if st.button("⏭ 진행(한 번)", use_container_width=True):
        st.session_state.running = True
        st.rerun()

with b3:
    if st.button("⏹ 중지", use_container_width=True):
        stop_job()
        st.rerun()

with b4:
    debug = st.checkbox("🪲 디버그 보기", value=False)

# 진행 표시
phase = st.session_state.phase
running = st.session_state.running

status = st.empty()
pbar1 = st.progress(0)
pbar2 = st.progress(0)

# 진행률(대략)
if phase in ("collect", "validate", "done"):
    # 목록
    maxp = max(1, int(st.session_state.get("max_pages", int(max_pages))))
    curp = min(maxp, max(1, int(st.session_state.page)))
    pbar1.progress(int(min(1.0, curp / maxp) * 100))
    # 상세
    total = max(1, len(st.session_state.validate_ids))
    done = min(total, int(st.session_state.validate_i))
    pbar2.progress(int(min(1.0, done / total) * 100))

if phase == "idle":
    status.info("대기 중. ▶ 시작을 눌러줘.")
elif phase == "collect":
    status.info(
        f"목록 수집 중… page={st.session_state.page} / 후보={len(st.session_state.candidates)} "
        f"(마지막 URL: {st.session_state.last_url})"
    )
elif phase == "validate":
    status.info(
        f"상세 작성일 검증 중… {st.session_state.validate_i} / {len(st.session_state.validate_ids)} "
        f"(통과={len(st.session_state.collected)})"
    )
elif phase == "done":
    status.success(f"완료! 선택한 날짜 글만 {len(st.session_state.posts)}개")

if debug:
    st.caption("DEBUG LOG (최근 200줄)")
    st.code("\n".join(st.session_state.logs[-200:]) if st.session_state.logs else "(로그 없음)")
    st.caption(f"last_url = {st.session_state.last_url}")

# 작업 스텝 실행
if running and phase in ("collect", "validate"):
    try:
        if phase == "collect":
            step_collect()
        elif phase == "validate":
            step_validate()
    except Exception as e:
        log(f"치명 오류: {type(e).__name__}: {e}")
        st.session_state.running = False

    if auto_run and st.session_state.running and st.session_state.phase in ("collect", "validate"):
        time.sleep(0.12)
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
