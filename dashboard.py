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

from webdriver_manager.chrome import ChromeDriverManager

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


# =========================
# 대상 게시판
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
    t = clean(text)

    m1 = re.search(r"\b(20\d{2})\.(\d{2})\.(\d{2})\.?\b", t)
    if m1:
        return date_cls(int(m1[1]), int(m1[2]), int(m1[3]))

    m2 = re.search(r"\b(\d{2})\.(\d{2})\.?\b", t)
    if m2:
        return ("MD", int(m2[1]), int(m2[2]))

    return None


def canonical_article_link(article_id: str) -> str:
    return f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"


def normalize_title(raw: str) -> str:
    t = clean(raw)
    t = re.sub(r"\[[^\]]+\]", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# =========================
# Chromium 경로 찾기
# =========================
def find_chrome_binary():
    for name in ["chromium", "google-chrome", "google-chrome-stable"]:
        p = which(name)
        if p:
            return p
    return "/usr/bin/chromium"


# =========================
# Selenium (🔥 핵심)
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

    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
    })

    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    # 🔑 Chromium 바이너리 명시
    opts.binary_location = find_chrome_binary()

    # 🔑 webdriver-manager가 Chromium 143에 맞는 driver 자동 다운로드
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    driver.implicitly_wait(0.5)
    return driver


def wait_list_loaded(driver):
    wait = WebDriverWait(driver, 20)
    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, LINK_CSS)) > 0)


def extract_article_id(el) -> str:
    for attr in ["data-articleid", "href", "onclick"]:
        try:
            v = clean(el.get_attribute(attr))
            m = ARTICLEID_RE.search(v)
            if m:
                return m.group(1) or m.group(2) or m.group(3)
        except Exception:
            pass
    return ""


# =========================
# 수집
# =========================
def collect_by_paging(target_date, headless, max_pages, pause):
    driver = make_driver(headless)
    collected = {}

    try:
        for page in range(1, max_pages + 1):
            driver.get(f"{BASE_LIST_URL}&search.page={page}")
            wait_list_loaded(driver)
            time.sleep(pause)

            rows = driver.find_elements(By.CSS_SELECTOR, "tr") or driver.find_elements(By.CSS_SELECTOR, "li")

            for row in rows:
                text = clean(row.text)
                if not text or "공지" in text:
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
                    continue

                hhmm = extract_time_token(text)
                dtok = extract_date_token_any(text)

                if target_date == kst_today():
                    if not hhmm:
                        continue
                else:
                    if hhmm:
                        continue
                    if isinstance(dtok, date_cls):
                        if dtok != target_date:
                            continue
                    elif isinstance(dtok, tuple):
                        _, m, d = dtok
                        if date_cls(target_date.year, m, d) != target_date:
                            continue
                    else:
                        continue

                link = canonical_article_link(article_id)
                collected[link] = {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "title": title,
                    "title_norm": normalize_title(title),
                    "link": link,
                }

            time.sleep(pause)

    finally:
        driver.quit()

    return list(collected.values())


# =========================
# UI
# =========================
st.set_page_config(page_title="중복 게시글 탐지", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")

with st.expander("설정", expanded=True):
    target_date = st.date_input("날짜", value=kst_today())
    headless = st.checkbox("헤드리스", value=True)
    max_pages = st.number_input("최대 페이지", 1, 300, 120)
    pause = st.number_input("대기(초)", 0.05, 1.0, 0.15, 0.05)

if st.button("수집 시작", use_container_width=True):
    try:
        posts = collect_by_paging(
            target_date=target_date,
            headless=headless,
            max_pages=int(max_pages),
            pause=float(pause),
        )
        st.success(f"수집 완료: {len(posts)}개")
        st.dataframe(pd.DataFrame(posts), use_container_width=True)
    except Exception as e:
        st.error("수집 오류")
        st.code(str(e))
        st.code(traceback.format_exc())
