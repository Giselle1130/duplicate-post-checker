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
BASE_LIST_URL = f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/menus/{MENU_ID}?viewType=L&page="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Referer": f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/menus/{MENU_ID}?viewType=L",
}


# =========================
# 유틸
# =========================
def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def kst_today() -> date_cls:
    # Cloud에서도 안전하게 "서버 시간 기준"으로 date 사용
    return datetime.now().date()


def extract_time_token(text: str) -> str:
    m = re.search(r"\b(\d{1,2}:\d{2})\b", clean(text))
    return m.group(1) if m else ""


def extract_date_token(text: str) -> str:
    # 2025.12.17 또는 2025.12.17.
    m = re.search(r"\b(20\d{2}\.\d{2}\.\d{2})\.?\b", clean(text))
    return m.group(1) if m else ""


def build_page_url(page: int) -> str:
    return BASE_LIST_URL + str(page)


def get_soup(page: int) -> BeautifulSoup:
    url = build_page_url(page)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


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
# 목록 수집 (Cloud OK / 날짜 필터)
# =========================
def collect_posts_by_date(target_date: date_cls, max_pages: int, pause: float, strict_date: bool):
    """
    - 오늘: 목록에 시간이 (HH:MM)로 나오므로, 시간 토큰이 있는 것만 모으기
    - 과거: 목록에 2025.12.17. 형태가 나오므로, 날짜 토큰이 target과 같으면 모으기
    - strict_date 체크 시:
        * 과거 날짜는 일치하는 글만 수집
        * 오늘은 HH:MM 있는 글만 수집
      (기본 ON 추천)
    """
    today = kst_today()
    is_today = (target_date == today)
    target_dot = target_date.strftime("%Y.%m.%d")
    target_iso = target_date.strftime("%Y-%m-%d")

    items = []

    for page in range(1, int(max_pages) + 1):
        soup = get_soup(page)

        # 목록에서 글 링크 후보 찾기
        anchors = soup.select("a[href*='/f-e/cafes/'][href*='/articles/'], a[href*='/articles/']")
        if not anchors:
            break

        for a in anchors:
            href = a.get("href") or ""
            title = clean(a.get_text(" ", strip=True))
            if not title:
                continue

            # article id 추출
            m = re.search(r"/articles/(\d+)", href)
            if not m:
                continue
            article_id = m.group(1)

            link = f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/articles/{article_id}?boardtype=L&menuid={MENU_ID}"

            # ✅ 날짜/시간은 링크 주변 텍스트에서 대충 긁기 (HTML 구조가 자주 바뀌어서 '느슨한 파싱')
            # 가장 실용적인 방식: anchor의 부모 텍스트에서 토큰 찾기
            context_text = ""
            try:
                context_text = clean(a.parent.get_text(" ", strip=True))
            except Exception:
                context_text = title

            hhmm = extract_time_token(context_text)
            dot = extract_date_token(context_text)

            if strict_date:
                if is_today:
                    # 오늘은 HH:MM 있는 글만
                    if not hhmm:
                        continue
                    date_raw = hhmm
                else:
                    # 과거는 날짜 토큰이 target과 같아야
                    if not dot or dot != target_dot:
                        continue
                    date_raw = dot
            else:
                # 느슨 모드: 토큰이 있으면 넣고, 없으면 빈 값
                date_raw = hhmm or dot or ""

            items.append({
                "date": target_iso,
                "date_raw": date_raw,
                "author": "",  # Cloud HTML만으로는 안정적으로 못 뽑아서 비움
                "title": title,
                "title_norm": normalize_title(title),
                "link": link,
            })

        time.sleep(float(pause))

    df = pd.DataFrame(items)
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).copy()
        df = df.sort_values(by="date_raw", ascending=False)
    return df.to_dict("records")


# =========================
# 중복/유사
# =========================
def compute_keyword_groups(df: pd.DataFrame, min_count: int = 2):
    if df.empty:
        return pd.DataFrame(columns=["keyword", "count", "examples"])

    tokens_list = []
    for _, row in df.iterrows():
        toks = tokenize(row["title"])
        tokens_list.append(toks)

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

    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xc = vec_c.fit_transform(titles)
    M = cosine_similarity(Xc)

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
st.set_page_config(page_title="클랜/방송/디스코드 중복 체크", layout="wide")
st.title("🏰 클랜 / 방송 / 디스코드 중복 체크 (Cloud 버전)")

with st.expander("설정", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜 선택", value=kst_today())
    with c2:
        max_pages = st.number_input("최대 페이지", min_value=1, max_value=200, value=30, step=5)
    with c3:
        pause = st.number_input("페이지 대기(초)", min_value=0.0, max_value=3.0, value=0.25, step=0.05)
    with c4:
        strict_date = st.checkbox("날짜/시간 엄격 비교(추천)", value=True)

keyword_min_count = st.number_input("키워드 중복 최소 건수", min_value=2, max_value=20, value=2, step=1)
sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

st.divider()

if st.button("수집 시작", use_container_width=True):
    st.session_state.posts = []
    try:
        posts = collect_posts_by_date(
            target_date=target_date,
            max_pages=int(max_pages),
            pause=float(pause),
            strict_date=bool(strict_date),
        )
        st.session_state.posts = posts
        st.success(f"수집 완료: {len(posts)}개")
    except Exception:
        st.error("수집 오류")
        st.code(traceback.format_exc())

df = pd.DataFrame(st.session_state.posts) if "posts" in st.session_state and st.session_state.posts else pd.DataFrame(
    columns=["date", "date_raw", "author", "title", "title_norm", "link"]
)

keyword_groups = compute_keyword_groups(df, min_count=int(keyword_min_count))
ai_similar = compute_ai_similar(df, threshold=float(sim_threshold))

tab1, tab2, tab3 = st.tabs(["📌 원본", "🔎 키워드 중복", "🤖 AI 유사"])

with tab1:
    st.dataframe(df, use_container_width=True)

with tab2:
    if keyword_groups.empty:
        st.info("해당 없음")
    else:
        st.dataframe(keyword_groups, use_container_width=True)

with tab3:
    if ai_similar.empty:
        st.info("해당 없음")
    else:
        st.dataframe(ai_similar, use_container_width=True)
