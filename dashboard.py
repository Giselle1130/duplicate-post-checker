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
