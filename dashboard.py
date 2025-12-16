import re
import time
import traceback
from datetime import datetime, date as date_cls

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 대상 게시판 고정
# =========================
CLUB_ID = 28866679
MENU_ID = 178
BASE_URL = f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/menus/{MENU_ID}?viewType=L&page="

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# =========================
# 유틸
# =========================
def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def get_soup(page: int):
    url = BASE_URL + str(page)
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# =========================
# 제목 정규화
# =========================
def normalize_title(t: str) -> str:
    t = clean(t)
    t = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", t)
    t = re.sub(r"\[[^\]]{1,30}\]", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# =========================
# 목록 수집 (Cloud OK)
# =========================
def collect_posts(max_pages: int):
    items = []

    for page in range(1, max_pages + 1):
        soup = get_soup(page)
        links = soup.select("a[href*='/articles/']")

        for a in links:
            title = clean(a.get_text())
            href = a.get("href", "")
            if not title or "/articles/" not in href:
                continue

            items.append({
                "title": title,
                "title_norm": normalize_title(title),
                "link": "https://cafe.naver.com" + href,
            })

        time.sleep(0.3)

    df = pd.DataFrame(items)
    return df.drop_duplicates(subset=["link"]).reset_index(drop=True)


# =========================
# AI 유사도
# =========================
def compute_ai_similar(df: pd.DataFrame, threshold: float):
    if len(df) < 2:
        return pd.DataFrame(columns=["title_a", "title_b", "similarity", "link_a", "link_b"])

    titles = df["title_norm"].tolist()
    raw = df["title"].tolist()
    links = df["link"].tolist()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vec.fit_transform(titles)
    M = cosine_similarity(X)

    rows = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if M[i, j] >= threshold:
                rows.append({
                    "title_a": raw[i],
                    "title_b": raw[j],
                    "similarity": round(float(M[i, j]), 3),
                    "link_a": links[i],
                    "link_b": links[j],
                })

    return pd.DataFrame(rows).sort_values("similarity", ascending=False)


# =========================
# UI
# =========================
st.set_page_config(page_title="중복 게시물 체크", layout="wide")
st.title("🏰 클랜 / 방송 / 디스코드 중복 체크")

max_pages = st.number_input("최대 페이지", 1, 100, 30)
threshold = st.slider("AI 유사도 기준", 0.5, 0.95, 0.78)

if st.button("수집 시작", use_container_width=True):
    try:
        df = collect_posts(int(max_pages))
        st.success(f"수집 완료: {len(df)}개")

        st.subheader("📌 원본")
        st.dataframe(df, use_container_width=True)

        ai = compute_ai_similar(df, float(threshold))
        st.subheader("🤖 AI 유사 중복")
        if ai.empty:
            st.info("중복 없음")
        else:
            st.dataframe(ai, use_container_width=True)

    except Exception:
        st.error("수집 오류")
        st.code(traceback.format_exc())
