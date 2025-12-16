import re
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

BASE_LIST_URL = (
    "https://cafe.naver.com/ArticleList.nhn"
    f"?search.clubid={CLUB_ID}"
    f"&search.menuid={MENU_ID}"
    "&search.boardtype=L"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://cafe.naver.com/",
}

ARTICLEID_RE = re.compile(r"(?:[?&]articleid=(\d+))", re.IGNORECASE)


# =====================
# 유틸
# =====================
def clean(x: str) -> str:
    return (x or "").replace("\u200b", "").strip()


def kst_today() -> date_cls:
    # Streamlit Cloud에선 서버 시간 꼬일 수 있어서 "date_input 기준"으로만 판단해도 됨
    return datetime.now().date()


def extract_time_token(text: str) -> str:
    m = re.search(r"\b(\d{1,2}:\d{2})\b", clean(text))
    return m.group(1) if m else ""


def extract_date_any(text: str, assume_year: int) -> date_cls | None:
    """
    지원:
    - 2025.12.16 / 2025.12.16.
    - 12.16 / 12.16.
    """
    t = clean(text)

    m1 = re.search(r"\b(20\d{2})\.(\d{2})\.(\d{2})\.?\b", t)
    if m1:
        try:
            return date_cls(int(m1.group(1)), int(m1.group(2)), int(m1.group(3)))
        except Exception:
            return None

    m2 = re.search(r"\b(\d{2})\.(\d{2})\.?\b", t)
    if m2:
        try:
            return date_cls(assume_year, int(m2.group(1)), int(m2.group(2)))
        except Exception:
            return None

    return None


def canonical_article_link(article_id: str) -> str:
    return f"https://cafe.naver.com/ca-fe/cafes/{CLUB_ID}/articles/{article_id}"


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
# HTML 파싱(iframe/비iframe 둘 다)
# =====================
def _parse_rows_from_doc(doc: BeautifulSoup) -> list[dict]:
    """
    doc(iframe 안쪽 문서)에서 글 row들을 최대한 넓게 파싱
    """
    items = []

    # 1) table row 기반
    rows = doc.select("tr")
    for r in rows:
        text = clean(r.get_text("\n"))
        if not text or "공지" in text:
            continue

        a = r.select_one("a[href*='articleid=']")
        if not a:
            continue

        href = a.get("href") or ""
        m = ARTICLEID_RE.search(href)
        if not m:
            continue
        article_id = m.group(1)

        title = clean(a.get_text())
        if not title:
            # 제목이 비면 row 첫 줄
            lines = [x.strip() for x in text.split("\n") if x.strip()]
            title = lines[0] if lines else ""
        if not title:
            continue

        items.append({
            "row_text": text,
            "article_id": article_id,
            "title": title,
            "href": href,
        })

    # 2) 혹시 tr이 비면 a태그 기반(백업)
    if not items:
        for a in doc.select("a[href*='articleid=']"):
            href = a.get("href") or ""
            m = ARTICLEID_RE.search(href)
            if not m:
                continue
            title = clean(a.get_text())
            if not title:
                continue
            items.append({
                "row_text": title,
                "article_id": m.group(1),
                "title": title,
                "href": href,
            })

    return items


def fetch_frame_doc(session: requests.Session, page: int, debug: bool = False):
    """
    1) wrapper 페이지 요청
    2) iframe(cafe_main) src 찾기
    3) iframe 문서 요청
    4) 실패하면 wrapper 자체에서 row 파싱 시도
    """
    url = f"{BASE_LIST_URL}&search.page={page}"
    r = session.get(url, timeout=20)
    r.encoding = "utf-8"
    wrapper = BeautifulSoup(r.text, "html.parser")

    iframe = wrapper.find("iframe", {"id": "cafe_main"})
    if iframe and iframe.get("src"):
        src = iframe["src"]
        if src.startswith("/"):
            frame_url = "https://cafe.naver.com" + src
        elif src.startswith("http"):
            frame_url = src
        else:
            frame_url = "https://cafe.naver.com/" + src.lstrip("/")

        fr = session.get(frame_url, timeout=20)
        fr.encoding = "utf-8"
        frame_doc = BeautifulSoup(fr.text, "html.parser")

        if debug:
            return frame_doc, {
                "wrapper_status": r.status_code,
                "frame_status": fr.status_code,
                "frame_url": frame_url,
                "has_iframe": True,
                "wrapper_head": r.text[:300],
                "frame_head": fr.text[:300],
            }
        return frame_doc, None

    # iframe이 없으면 wrapper 자체를 frame처럼 취급해 파싱
    if debug:
        return wrapper, {
            "wrapper_status": r.status_code,
            "frame_status": None,
            "frame_url": None,
            "has_iframe": False,
            "wrapper_head": r.text[:500],
        }
    return wrapper, None


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
    assume_year = target_date.year
    is_today = (target_date == kst_today())

    last_debug = None

    for page in range(1, max_pages + 1):
        if status_cb:
            status_cb(f"페이지 {page}/{max_pages} 로딩/파싱 중…")
        if progress_cb:
            progress_cb(page / float(max_pages))

        doc, dbg = fetch_frame_doc(session, page, debug=debug)
        if dbg:
            last_debug = dbg

        items = _parse_rows_from_doc(doc)
        if not items:
            # 더 이상 글이 없거나 차단/다른 페이지
            break

        page_hit = 0
        oldest_seen = None

        for it in items:
            row_text = it["row_text"]
            title = it["title"]
            article_id = it["article_id"]

            hhmm = extract_time_token(row_text)
            row_date = extract_date_any(row_text, assume_year=assume_year)

            # oldest_seen(조기 종료용)
            if row_date:
                oldest_seen = row_date if (oldest_seen is None or row_date < oldest_seen) else oldest_seen

            if is_today:
                # 오늘: 시간형( HH:MM )이 있는 것만
                if not hhmm:
                    continue
            else:
                # 과거: 날짜형만
                if hhmm:
                    continue
                if row_date != target_date:
                    continue

            results.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "date_raw": hhmm if is_today else (row_date.strftime("%Y.%m.%d") if row_date else ""),
                "author": "",  # requests 파싱에선 author 안정적으로 뽑기 어려워 우선 비움(원하면 추가 가능)
                "title": title,
                "title_norm": normalize_title(title),
                "link": canonical_article_link(article_id),
            })
            page_hit += 1

        # ✅ 조기 종료: 과거 날짜 수집인데 page에 나온 가장 오래된 날짜가 target보다 과거면 stop
        if (not is_today) and oldest_seen and oldest_seen < target_date:
            break

        # ✅ 너무 깊이 내려가며 매칭이 0이면 stop (속도↑)
        if (not is_today) and page >= 3 and page_hit == 0:
            break

        if pause > 0:
            time.sleep(pause)

    df = pd.DataFrame(results).drop_duplicates(subset=["link"]).copy()

    return df, last_debug


# =====================
# 중복/유사(탭 복구)
# =====================
def compute_author_dups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "author" not in df.columns:
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

    df2 = df.copy()
    if len(df2) > max_n:
        df2 = df2.head(max_n).copy()

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
        debug = st.checkbox("디버그 표시(0개일 때 원인 확인)", value=True)
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
    st.dataframe(author_dups, use_container_width=True) if not author_dups.empty else st.info("해당 없음 (author가 비어있으면 이 탭은 비게 돼)")

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
        st.info("디버그 정보 없음 (아직 수집을 안 했거나, debug 체크가 꺼져있음)")
    else:
        st.write(dbg)
        st.caption("wrapper_head / frame_head에 차단/보안/로그인 유도 문구가 보이면 requests가 막힌 상태일 수 있어.")
