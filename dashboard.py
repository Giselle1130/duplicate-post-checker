import os
import re
import time
import traceback
from datetime import datetime, date as date_cls
from shutil import which

import pandas as pd
import streamlit as st

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait

from webdriver_manager.chrome import ChromeDriverManager

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

ARTICLEID_RE = re.compile(
    r"(?:[?&]articleid=(\d+))|(?:/articles/(\d+))|(?:articleid[=:]\s*['\"]?(\d+))",
    re.IGNORECASE,
)

LINK_CSS = (
    "a[href*='articleid='], a[href*='/articles/'], "
    "a[onclick*='articleid'], a[data-articleid]"
)


# =========================
# 유틸
# =========================
def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def kst_today() -> date_cls:
    return datetime.now(KST).date() if KST else datetime.now().date()


def extract_time_token(text: str) -> str:
    m = re.search(r"\b(\d{1,2}:\d{2})\b", clean(text))
    return m.group(1) if m else ""


def extract_date_token_any(text: str):
    """
    - 2025.12.16 / 2025.12.16. (연도 포함)
    - 12.16 / 12.16. (연도 없음)
    """
    t = clean(text)

    m1 = re.search(r"\b(20\d{2})\.(\d{2})\.(\d{2})\.?\b", t)
    if m1:
        return date_cls(int(m1[1]), int(m1[2]), int(m1[3]))

    m2 = re.search(r"\b(\d{2})\.(\d{2})\.?\b", t)
    if m2:
        return ("MD", int(m2[1]), int(m2[2]))

    return None


def canonical_article_link(article_id: str) -> str:
    # 같은 글은 항상 동일 링크로 저장
    return f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"


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


def find_chrome_binary():
    # Streamlit Cloud는 보통 /usr/bin/chromium
    env = os.environ.get("CHROME_BIN") or os.environ.get("GOOGLE_CHROME_BIN")
    if env and os.path.exists(env):
        return env

    for name in ["chromium", "google-chrome", "google-chrome-stable"]:
        p = which(name)
        if p:
            return p

    # fallback
    return "/usr/bin/chromium"


# =========================
# Selenium (핵심)
# =========================
def make_driver(headless=True) -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,900")
    opts.page_load_strategy = "eager"

    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--remote-debugging-port=0")

    # 이미지 차단(속도↑)
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    })

    # Chromium 바이너리 고정
    opts.binary_location = find_chrome_binary()

    # ✅✅✅ 가장 중요:
    # webdriver-manager가 내려준 "절대경로"를 Service에 박아서
    # 시스템에 남아있는 chromedriver(114)가 있어도 무시하게 만든다.
    driver_path = ChromeDriverManager().install()
    service = Service(executable_path=driver_path)

    driver = webdriver.Chrome(service=service, options=opts)
    driver.implicitly_wait(0.5)
    return driver


def wait_list_loaded(driver):
    wait = WebDriverWait(driver, 25)
    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0)


def extract_article_id(el) -> str:
    for attr in ["data-articleid", "href", "onclick"]:
        try:
            v = clean(el.get_attribute(attr))
            m = ARTICLEID_RE.search(v)
            if m:
                return (m.group(1) or m.group(2) or m.group(3) or "").strip()
        except Exception:
            pass
    return ""


# =========================
# 수집 (조기 종료 + 진행 표시)
# =========================
def collect_by_paging(target_date, headless, max_pages, pause, progress_cb=None, status_cb=None):
    driver = make_driver(headless=headless)
    collected = {}

    try:
        for page in range(1, int(max_pages) + 1):
            if status_cb:
                status_cb(f"페이지 로딩 중... {page}/{max_pages}")
            driver.get(f"{BASE_LIST_URL}&search.page={page}")
            wait_list_loaded(driver)

            if pause and pause > 0:
                time.sleep(float(pause))

            rows = driver.find_elements(By.CSS_SELECTOR, "tr")
            if not rows:
                rows = driver.find_elements(By.CSS_SELECTOR, "li")

            # 조기 종료 판단: 이 페이지에서 관측된 "가장 오래된 날짜"
            oldest_seen = None
            matched_this_page = 0

            for row in rows:
                text = clean(row.text)
                if not text:
                    continue
                if "공지" in text:
                    continue

                links = row.find_elements(By.CSS_SELECTOR, LINK_CSS)
                if not links:
                    continue

                a = links[0]
                article_id = extract_article_id(a)
                if not article_id:
                    continue

                title = clean(a.text)
                if not title:
                    # a.text가 비는 케이스 대비: row 첫 줄
                    lines = [x.strip() for x in text.split("\n") if x.strip()]
                    title = lines[0] if lines else ""
                if not title:
                    continue

                hhmm = extract_time_token(text)
                dtok = extract_date_token_any(text)

                # row_date 만들기(조기 종료/필터)
                row_date = None
                if isinstance(dtok, date_cls):
                    row_date = dtok
                elif isinstance(dtok, tuple) and dtok[0] == "MD":
                    _, m, d = dtok
                    try:
                        row_date = date_cls(target_date.year, m, d)
                    except Exception:
                        row_date = None

                if row_date:
                    oldest_seen = row_date if (oldest_seen is None or row_date < oldest_seen) else oldest_seen

                # ===== 필터링 =====
                if target_date == kst_today():
                    # 오늘: 시간형만
                    if not hhmm:
                        continue
                else:
                    # 과거: 날짜형만
                    if hhmm:
                        continue
                    if row_date != target_date:
                        continue

                link = canonical_article_link(article_id)
                collected[link] = {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "title": title,
                    "title_norm": normalize_title(title),
                    "link": link,
                }
                matched_this_page += 1

            # 진행 표시
            if progress_cb:
                progress_cb(min(page / float(max_pages), 1.0))

            # ✅ 조기 종료:
            # 과거 날짜 수집 시, 페이지의 oldest_seen가 target보다 더 과거면
            # 앞으로는 더 과거만 나오므로 중단
            if target_date != kst_today() and oldest_seen and oldest_seen < target_date:
                if status_cb:
                    status_cb(f"조기 종료: {oldest_seen} < {target_date} (더 과거로 내려감)")
                break

            # 또, 연속으로 매칭이 0이면 너무 깊이 내려간 거라 중단(속도↑)
            if target_date != kst_today() and matched_this_page == 0 and page >= 3:
                # 3페이지까지는 UI 흔들림 고려해서 봐주고, 그 이후 0이면 끊기
                if status_cb:
                    status_cb("조기 종료: 연속 매칭 0페이지")
                break

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    return list(collected.values())


# =========================
# UI
# =========================
st.set_page_config(page_title="클랜/방송/디스코드 중복 게시글 체크", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")

with st.expander("설정", expanded=True):
    target_date = st.date_input("날짜", value=kst_today())
    headless = st.checkbox("헤드리스", value=True)
    max_pages = st.number_input("최대 페이지", min_value=1, max_value=500, value=120, step=5)
    pause = st.number_input("대기(초)", min_value=0.0, max_value=2.0, value=0.15, step=0.05)

st.divider()

if st.button("수집 시작", use_container_width=True):
    try:
        prog = st.progress(0.0)
        status = st.empty()

        posts = collect_by_paging(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            pause=float(pause),
            progress_cb=lambda v: prog.progress(v),
            status_cb=lambda msg: status.info(msg),
        )

        df = pd.DataFrame(posts)
        status.empty()
        prog.empty()

        st.success(f"수집 완료: {len(posts)}개")
        if df.empty:
            st.info("해당 날짜로 필터링된 글이 없어요. (카페 목록 날짜 표기/시간 표기 확인 필요)")
        else:
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error("수집 오류")
        st.code(str(e))
        st.code(traceback.format_exc())
