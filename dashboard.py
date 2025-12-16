import re
import time
from datetime import datetime, date as date_cls

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


# =====================
# 게시판 설정
# =====================
CLUB_ID = 28866679
MENU_ID = 178

BASE_URL = (
    "https://cafe.naver.com/ArticleList.nhn"
    f"?search.clubid={CLUB_ID}"
    f"&search.menuid={MENU_ID}"
    "&search.boardtype=L"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# =====================
# 유틸
# =====================
def kst_today():
    return datetime.now(KST).date() if KST else datetime.now().date()


def clean(t):
    return (t or "").replace("\u200b", "").strip()


def extract_time(text):
    m = re.search(r"\b\d{1,2}:\d{2}\b", text)
    return m.group(0) if m else None


def extract_date(text):
    m = re.search(r"(20\d{2}\.\d{2}\.\d{2})", text)
    if m:
        return datetime.strptime(m.group(1), "%Y.%m.%d").date()
    return None


# =====================
# 수집
# =====================
def collect_posts(target_date, max_pages, pause, status_cb=None):
    session = requests.Session()
    session.headers.update(HEADERS)

    results = []

    for page in range(1, max_pages + 1):
        if status_cb:
            status_cb(f"페이지 {page} 수집 중...")

        res = session.get(BASE_URL + f"&search.page={page}")
        soup = BeautifulSoup(res.text, "html.parser")

        iframe = soup.find("iframe", {"id": "cafe_main"})
        if not iframe:
            break

        iframe_url = "https://cafe.naver.com" + iframe["src"]
        iframe_html = session.get(iframe_url).text
        doc = BeautifulSoup(iframe_html, "html.parser")

        rows = doc.select("tr")
        page_hit = 0

        for r in rows:
            text = clean(r.get_text(" "))
            if not text or "공지" in text:
                continue

            time_token = extract_time(text)
            date_token = extract_date(text)

            if target_date == kst_today():
                if not time_token:
                    continue
            else:
                if time_token or date_token != target_date:
                    continue

            a = r.find("a", href=re.compile("articleid"))
            if not a:
                continue

            article_id = re.search(r"articleid=(\d+)", a["href"]).group(1)
            title = clean(a.get_text())

            results.append({
                "date": target_date.isoformat(),
                "title": title,
                "link": f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"
            })
            page_hit += 1

        if page_hit == 0:
            break

        time.sleep(pause)

    return results


# =====================
# UI
# =====================
st.set_page_config(page_title="중복 게시글 체크", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")

with st.expander("설정", expanded=True):
    target_date = st.date_input("날짜", value=kst_today())
    max_pages = st.number_input("최대 페이지", 1, 300, 120)
    pause = st.number_input("대기(초)", 0.0, 1.0, 0.1)

st.divider()

if st.button("수집 시작", use_container_width=True):
    status = st.empty()
    posts = collect_posts(
        target_date,
        int(max_pages),
        float(pause),
        status_cb=lambda m: status.info(m),
    )
    status.empty()

    df = pd.DataFrame(posts)
    st.success(f"수집 완료: {len(df)}개")
    st.dataframe(df, use_container_width=True)
