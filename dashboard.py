import re
import json
import time
from datetime import datetime, date as date_cls

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================
# 게시판 설정
# =====================
CLUB_ID = 28866679
MENU_ID = 178

# ✅ 새 UI(Next.js) 메뉴 URL
BASE_MENU_URL = f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/menus/{MENU_ID}?viewType=L&page="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://cafe.naver.com/",
}


# =====================
# 유틸
# =====================
def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def kst_today() -> date_cls:
    # Streamlit Cloud에서 timezone 꼬임 방지: 날짜 선택값 기준으로 써도 충분
    return datetime.now().date()


def canonical_article_link(article_id: str) -> str:
    return f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"


def parse_ymd(s: str) -> date_cls | None:
    # "2025-12-17", "2025.12.17", "2025/12/17"
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def parse_epoch_ms(ms) -> date_cls | None:
    try:
        ms = int(ms)
        return datetime.fromtimestamp(ms / 1000.0).date()
    except Exception:
        return None


# =====================
# 제목 정규화/토큰
# =====================
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
    toks = [x for x in toks if len(x) >= 2 and x not in STOPWORDS]
    return toks


# =====================
# Next.js(__NEXT_DATA__) 파서
# =====================
def deep_find_articles(obj, out: list):
    """
    JSON 안을 재귀로 훑어서:
      - articleId(또는 id) + title/subject 같은 필드가 있는 dict
    를 최대한 많이 수집
    """
    if isinstance(obj, dict):
        # 후보 1) 단일 article dict
        keys = set(obj.keys())
        has_id = ("articleId" in keys) or ("id" in keys and "articleId" not in keys)
        has_title = ("title" in keys) or ("subject" in keys) or ("articleTitle" in keys)
        if has_id and has_title:
            out.append(obj)

        # 후보 2) articles/list 같은 배열
        for v in obj.values():
            deep_find_articles(v, out)

    elif isinstance(obj, list):
        for it in obj:
            deep_find_articles(it, out)


def extract_from_next_data(html: str):
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag or not tag.string:
        return [], {"has_next_data": False, "next_data_head": (html[:400] if html else "")}

    try:
        data = json.loads(tag.string)
    except Exception:
        return [], {"has_next_data": True, "json_parse_failed": True}

    candidates = []
    deep_find_articles(data, candidates)

    return candidates, {
        "has_next_data": True,
        "candidates_found": len(candidates),
    }


def normalize_article_candidate(c: dict):
    """
    후보 dict에서 우리가 필요한 형태로 최대한 뽑기
    """
    article_id = c.get("articleId") or c.get("id")
    title = c.get("title") or c.get("subject") or c.get("articleTitle") or ""

    # 날짜 후보들(구조가 바뀔 수 있어서 여러 키 시도)
    d = (
        parse_ymd(str(c.get("writeDate") or c.get("createdDate") or c.get("date") or ""))
        or parse_epoch_ms(c.get("writeDateTime") or c.get("createdAt") or c.get("timestamp"))
    )

    author = (
        c.get("writerNick") or c.get("writerNickname") or c.get("writer") or c.get("nickname") or ""
    )

    return str(article_id or ""), str(title), d, str(author or "")


# =====================
# 수집
# =====================
def collect_posts(
    target_date: date_cls,
    max_pages: int,
    pause: float,
    debug: bool = False,
    status_cb=None,
    progress_cb=None,
):
    session = requests.Session()
    session.headers.update(HEADERS)

    results = []
    last_dbg = None

    for page in range(1, max_pages + 1):
        if status_cb:
            status_cb(f"페이지 {page}/{max_pages} 수집 중…")
        if progress_cb:
            progress_cb(page / float(max_pages))

        url = BASE_MENU_URL + str(page)
        r = session.get(url, timeout=25)
        r.encoding = "utf-8"

        candidates, dbg = extract_from_next_data(r.text)
        if debug:
            last_dbg = {"status": r.status_code, "url": url, **dbg}

        if not candidates:
            break

        page_hit = 0
        oldest_seen = None

        for c in candidates:
            article_id, title, d, author = normalize_article_candidate(c)
            if not article_id or not title:
                continue
            if not d:
                # 날짜 못뽑는 후보는 스킵
                continue

            oldest_seen = d if (oldest_seen is None or d < oldest_seen) else oldest_seen

            if d != target_date:
                continue

            results.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "date_raw": d.strftime("%Y.%m.%d"),
                "author": author,
                "title": title,
                "title_norm": normalize_title(title),
                "link": canonical_article_link(article_id),
            })
            page_hit += 1

        # ✅ 조기 종료: 더 과거로 내려갔으면 stop
        if oldest_seen and oldest_seen < target_date:
            break

        # ✅ 너무 오래 도는 것 방지
        if page >= 3 and page_hit == 0:
            break

        if pause > 0:
            time.sleep(pause)

    df = pd.DataFrame(results).drop_duplicates(subset=["link"]).copy()
    return df, last_dbg


# =====================
# 중복/유사(탭)
# =====================
def compute_author_dups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "author", "count"])
    a = df.groupby(["date", "author"]).size().reset_index(name="count")
    return a[a["count"] >= 2].sort_values(by="count", ascending=False)


def compute_exact_dups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    return df[df.duplicated(subset=["date", "title_norm"], keep=False)].copy()


def compute_keyword_groups(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
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


# =====================
# UI
# =====================
st.set_page_config(page_title="클랜/방송/디스코드 중복 게시글 체크", layout="wide")
st.title("🏰 클랜/방송/디스코드 중복 게시글 체크")

with st.expander("설정", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1])
    with c1:
        target_date = st.date_input("날짜", value=datetime.now().date())
    with c2:
        max_pages = st.number_input("최대 페이지", 1, 500, 120, step=10)
    with c3:
        pause = st.number_input("대기(초)", 0.0, 2.0, 0.10, step=0.05)
    with c4:
        debug = st.checkbox("디버그 표시", value=True)
    with c5:
        run_ai = st.checkbox("🤖 AI 유사도 계산(무거움)", value=False)

c6, c7 = st.columns([1, 1])
with c6:
    keyword_min_count = st.number_input("키워드 중복 최소 건수", 2, 20, 2)
with c7:
    sim_threshold = st.slider("AI 유사도 기준", 0.50, 0.99, 0.78, 0.01)

st.divider()

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["date", "date_raw", "author", "title", "title_norm", "link"])
if "dbg" not in st.session_state:
    st.session_state.dbg = None

if st.button("수집 시작", use_container_width=True):
    status = st.empty()
    prog = st.progress(0.0)

    df, dbg = collect_posts(
        target_date=target_date,
        max_pages=int(max_pages),
        pause=float(pause),
        debug=bool(debug),
        status_cb=lambda m: status.info(m),
        progress_cb=lambda v: prog.progress(min(max(v, 0.0), 1.0)),
    )

    st.session_state.df = df
    st.session_state.dbg = dbg

    status.empty()
    prog.empty()

    st.success(f"수집 완료: {len(df)}개")

df = st.session_state.df

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📌 원본", "🚨 작성자 동일", "🧨 제목 동일", "🔎 키워드 중복", "🤖 AI 유사", "🛠 디버그"])

with tab1:
    st.dataframe(df, use_container_width=True)

with tab2:
    author_dups = compute_author_dups(df)
    st.dataframe(author_dups, use_container_width=True) if not author_dups.empty else st.info("해당 없음")

with tab3:
    exact_dups = compute_exact_dups(df)
    st.dataframe(exact_dups, use_container_width=True) if not exact_dups.empty else st.info("해당 없음")

with tab4:
    kw = compute_keyword_groups(df, min_count=int(keyword_min_count))
    st.dataframe(kw, use_container_width=True) if not kw.empty else st.info("해당 없음")

with tab5:
    if not run_ai:
        st.info("AI 유사도는 무거워서 기본 OFF야. 설정에서 체크하면 계산해.")
    else:
        sim = compute_ai_similar(df, threshold=float(sim_threshold))
        st.dataframe(sim, use_container_width=True) if not sim.empty else st.info("해당 없음")

with tab6:
    dbg = st.session_state.dbg
    if not dbg:
        st.info("디버그 정보 없음")
    else:
        st.write(dbg)
        st.caption("candidates_found가 0이면 Next 데이터 구조가 바뀐 것. 그때는 dbg 캡처로 구조 맞춰서 다시 추출하면 돼.")
