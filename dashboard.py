import re
import time
from datetime import datetime, date as date_cls, timedelta
from urllib.parse import urljoin

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------
# KST (배포 안전)
# ---------------------
try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


def kst_now() -> datetime:
    if KST:
        return datetime.now(KST)
    return datetime.utcnow() + timedelta(hours=9)


def kst_today() -> date_cls:
    return kst_now().date()


# =====================
# 게시판 고정
# =====================
CLUB_ID = 28866679
MENU_ID = 178
BASE_URL = f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/menus/{MENU_ID}?viewType=L&page="

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}


# =====================
# ✅ 네이버 로그인(쿠키) 적용 헤더
# =====================
def get_headers() -> dict:
    headers = BASE_HEADERS.copy()

    # 1) secrets 우선 (Render 환경변수/Streamlit secrets로 넣을 수 있음)
    nid_aut = st.secrets.get("NID_AUT", "") if hasattr(st, "secrets") else ""
    nid_ses = st.secrets.get("NID_SES", "") if hasattr(st, "secrets") else ""

    # 2) UI 입력값(세션) 있으면 덮어쓰기
    cookie = st.session_state.get("naver_cookie", "")
    if cookie:
        headers["Cookie"] = cookie
        return headers

    # 3) secrets가 있으면 cookie 구성
    if nid_aut and nid_ses:
        headers["Cookie"] = f"NID_AUT={nid_aut}; NID_SES={nid_ses}"
        return headers

    # 4) 쿠키 없으면 그냥 반환 (배포에서는 막힐 수 있음)
    return headers


# =====================
# 날짜 텍스트 해석 (오늘/과거 통일)
# =====================
def infer_date_from_list_text(date_text: str) -> date_cls | None:
    s = (date_text or "").strip()

    # HH:MM => 오늘(KST)
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return kst_today()

    # YYYY.MM.DD => 해당 날짜
    m = re.match(r"^(\d{4})\.(\d{2})\.(\d{2})$", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date_cls(y, mo, d)
        except Exception:
            return None

    return None


def is_target_date(date_text: str, target_date: date_cls) -> bool:
    inferred = infer_date_from_list_text(date_text)
    return inferred == target_date


# =====================
# 텍스트 정규화
# =====================
def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_title(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"[^\w가-힣 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokens(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣 ]+", " ", s)
    return [p for p in s.split() if len(p) >= 2]


# =====================
# 목록/본문 수집
# =====================
def fetch_list_page(page: int):
    url = BASE_URL + str(page)
    res = requests.get(url, headers=get_headers(), timeout=25, allow_redirects=True)
    return url, res


def collect_article_list(target_date: date_cls, max_pages: int, debug: bool = False):
    articles = []
    debug_log = []

    for page in range(1, max_pages + 1):
        url, res = fetch_list_page(page)
        html = res.text or ""
        soup = BeautifulSoup(html, "html.parser")

        # 1) 우선 a.article
        links = soup.select("a.article")

        # 2) 없으면 /articles/ 포함 링크로 fallback
        if not links:
            links = [a for a in soup.select("a[href]") if "/articles/" in (a.get("href") or "")]

        if debug:
            sample_dates = [dt.get_text(strip=True) for dt in soup.select("td.td_date")[:10]]
            debug_log.append(
                {
                    "page": page,
                    "status": res.status_code,
                    "final_url": res.url,
                    "found_links": len(links),
                    "sample_date_texts": ", ".join(sample_dates) if sample_dates else "(none)",
                    "html_head": html[:400].replace("\n", " "),
                }
            )

        # ✅ 1페이지부터 링크가 0개면: 차단/빈 HTML 가능성이 매우 큼 → 더 돌려도 소용 없음
        if page == 1 and len(links) == 0:
            break

        for a in links:
            href = a.get("href") or ""
            if not href:
                continue

            title = a.get_text(strip=True) or a.get("title", "") or ""

            # date/author는 보통 같은 tr 안에 있음
            date_text = ""
            author = ""

            row = a.find_parent("tr")
            if row:
                dt = row.select_one("td.td_date")
                if dt:
                    date_text = dt.get_text(strip=True)
                au = row.select_one("td.td_name")
                if au:
                    author = au.get_text(strip=True)

            # 날짜 못 잡았으면 주변에서 한번 더
            if not date_text:
                near = a.find_parent()
                if hasattr(near, "select_one"):
                    dt2 = near.select_one("td.td_date")
                    if dt2:
                        date_text = dt2.get_text(strip=True)

            # ✅ 날짜 필터
            if not is_target_date(date_text, target_date):
                continue

            full_url = urljoin("https://cafe.naver.com", href)

            articles.append(
                {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "date_raw": date_text,
                    "author": author,
                    "title": title,
                    "title_norm": normalize_title(title),
                    "link": full_url,
                }
            )

        time.sleep(0.25)

    return articles, debug_log


def fetch_content(url: str) -> str:
    try:
        res = requests.get(url, headers=get_headers(), timeout=25, allow_redirects=True)
        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        iframe = soup.select_one("iframe#cafe_main")
        if iframe and iframe.get("src"):
            iframe_url = urljoin("https://cafe.naver.com", iframe["src"])
            res2 = requests.get(iframe_url, headers=get_headers(), timeout=25, allow_redirects=True)
            if res2.status_code != 200:
                return ""
            soup = BeautifulSoup(res2.text, "html.parser")

        content = soup.select_one("div.se-main-container")
        if not content:
            content = soup.select_one("div#postViewArea") or soup.select_one("div.ContentRenderer")
        if not content:
            return ""

        return content.get_text(" ", strip=True)

    except Exception:
        return ""


# =====================
# 중복 판정
# =====================
def dup_by_author(df: pd.DataFrame):
    groups = df[df["author"].astype(str).str.len() > 0].groupby("author").indices
    pairs = []
    for author, idxs in groups.items():
        if len(idxs) >= 2:
            idxs = list(idxs)
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pairs.append((idxs[i], idxs[j], 1.0, f"작성자 동일: {author}"))
    return pairs


def dup_by_title(df: pd.DataFrame):
    groups = df.groupby("title_norm").indices
    pairs = []
    for t, idxs in groups.items():
        if t and len(idxs) >= 2:
            idxs = list(idxs)
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pairs.append((idxs[i], idxs[j], 1.0, "제목 동일"))
    return pairs


def dup_by_keywords(df: pd.DataFrame, jaccard_threshold: float = 0.6):
    token_sets = []
    for _, r in df.iterrows():
        token_sets.append(set(simple_tokens(f"{r.get('title','')} {r.get('content','')}")))

    pairs = []
    n = len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            score = len(a & b) / len(a | b) if (a | b) else 0.0
            if score >= jaccard_threshold:
                pairs.append((i, j, score, f"키워드 중복(Jaccard {score:.2f})"))
    return pairs


def dup_by_ai(df: pd.DataFrame, threshold: float = 0.7):
    texts = df["content"].fillna("").astype(str).tolist()
    try:
        vectorizer = TfidfVectorizer(min_df=2)
        tfidf = vectorizer.fit_transform(texts)
    except Exception:
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf = vectorizer.fit_transform(texts)

    sim = cosine_similarity(tfidf)
    pairs = []
    for i in range(len(sim)):
        for j in range(i + 1, len(sim)):
            if sim[i, j] >= threshold:
                pairs.append((i, j, float(sim[i, j]), f"AI 유사(cos {sim[i,j]:.2f})"))
    return pairs


def build_pairs_table(df: pd.DataFrame, pairs: list[tuple]):
    rows = []
    for i, j, score, reason in pairs:
        rows.append(
            {
                "A_idx": i,
                "A_title": df.loc[i, "title"],
                "A_author": df.loc[i, "author"],
                "A_link": df.loc[i, "link"],
                "B_idx": j,
                "B_title": df.loc[j, "title"],
                "B_author": df.loc[j, "author"],
                "B_link": df.loc[j, "link"],
                "score": round(float(score), 3),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


# =====================
# UI
# =====================
st.set_page_config(page_title="클랜/방송/디스코드 중복검사", layout="wide")
st.markdown("<style>.block-container{max-width:1400px;}</style>", unsafe_allow_html=True)

st.title("📌 클랜/방송/디스코드 중복검사")

# ✅ 로그인(쿠키) 입력 영역
st.subheader("🔐 네이버 로그인 (쿠키 입력)")
with st.expander("쿠키 입력 방법 / 입력칸 열기", expanded=True):
    st.markdown(
        """
**중요:** ID/비밀번호 입력이 아니라, **로그인된 쿠키 값만 붙여넣는 방식**이야.

### 쿠키 복사하는 방법 (PC 크롬 기준)
1) 네이버 카페에 접속해서 **로그인**
2) 키보드 `F12` (개발자도구) 열기
3) 위 탭에서 **Application(애플리케이션)** 선택  
   - 안 보이면 `>>` 누르고 찾기
4) 왼쪽 메뉴에서 **Cookies → https://cafe.naver.com**
5) 오른쪽 표에서 아래 2개를 찾아서 **Value를 복사**
   - `NID_AUT`
   - `NID_SES`

복사한 값은 아래 칸에 붙여넣기만 하면 돼.
        """
    )

    nid_aut_in = st.text_input("NID_AUT 값", type="password")
    nid_ses_in = st.text_input("NID_SES 값", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ 쿠키 저장"):
            if not nid_aut_in or not nid_ses_in:
                st.error("NID_AUT, NID_SES 둘 다 입력해야 해.")
            else:
                st.session_state["naver_cookie"] = f"NID_AUT={nid_aut_in}; NID_SES={nid_ses_in}"
                st.success("저장 완료! 이제 수집이 가능해.")
    with col2:
        if st.button("🧹 쿠키 삭제"):
            st.session_state.pop("naver_cookie", None)
            st.success("쿠키 삭제 완료.")

cookie_ready = bool(st.session_state.get("naver_cookie")) or (
    hasattr(st, "secrets") and st.secrets.get("NID_AUT", "") and st.secrets.get("NID_SES", "")
)

if not cookie_ready:
    st.warning("⚠️ Render 배포에서는 쿠키가 없으면 목록이 0개로 나올 수 있어. (네이버가 서버 IP를 막는 경우)")
st.divider()

# 상단 토글
colA, colB, colC, colD, colE = st.columns([1, 1, 1, 1, 1])
with colA:
    opt_original = st.toggle("📌 원본", value=True)
with colB:
    opt_author = st.toggle("🚨 작성자 동일", value=True)
with colC:
    opt_title = st.toggle("🧷 제목 동일", value=True)
with colD:
    opt_keyword = st.toggle("🔎 키워드 중복", value=False)
with colE:
    opt_ai = st.toggle("🤖 AI 유사", value=True)

st.divider()

left, right = st.columns([1, 1])
with left:
    target_date = st.date_input("📅 수집 날짜 선택 (KST 기준)", kst_today())
with right:
    max_pages = st.number_input("📄 최대 페이지 수", 1, 200, 10, 1)

with st.expander("⚙️ 중복 판정 옵션", expanded=False):
    ai_threshold = st.slider("🤖 AI 유사 임계치 (cosine)", 0.1, 0.99, 0.70, 0.01)
    kw_threshold = st.slider("🔎 키워드 중복 임계치 (Jaccard)", 0.1, 0.99, 0.60, 0.01)

with st.expander("🧪 디버그 (배포에서 0개면 확인)", expanded=False):
    debug_mode = st.checkbox("디버그 모드 켜기(페이지 상태/HTML 일부 표시)", value=False)

st.divider()

if "df" not in st.session_state:
    st.session_state["df"] = None

run = st.button("📥 게시글 수집 시작", type="primary")

if run:
    if not cookie_ready:
        st.error("Render 배포에서는 쿠키 없이 수집이 막힐 가능성이 매우 높아. 위에서 쿠키를 저장하고 다시 눌러줘.")
        st.stop()

    with st.spinner("게시글 목록 수집 중..."):
        articles, debug_log = collect_article_list(target_date, int(max_pages), debug=debug_mode)

    if debug_mode:
        st.subheader("🧪 디버그 로그")
        st.dataframe(pd.DataFrame(debug_log), use_container_width=True)

    if not articles:
        st.error("목록에서 해당 날짜 게시글을 찾지 못했어. (쿠키가 만료됐거나, 네이버가 차단/HTML 변경 가능)")
        st.stop()

    st.success(f"목록 수집 완료: {len(articles)}개")

    progress = st.progress(0.0)
    contents = []
    for i, art in enumerate(articles):
        contents.append(fetch_content(art["link"]))
        progress.progress((i + 1) / len(articles))
        time.sleep(0.15)

    df = pd.DataFrame(articles)
    df["content"] = contents
    st.session_state["df"] = df

df = st.session_state.get("df")
if df is not None:
    st.subheader("✅ 수집 결과")

    if opt_original:
        st.dataframe(df[["date", "date_raw", "author", "title", "title_norm", "link"]], use_container_width=True)

    all_pairs = []
    if opt_author:
        all_pairs += dup_by_author(df)
    if opt_title:
        all_pairs += dup_by_title(df)
    if opt_keyword:
        all_pairs += dup_by_keywords(df, float(kw_threshold))
    if opt_ai:
        all_pairs += dup_by_ai(df, float(ai_threshold))

    if not (opt_author or opt_title or opt_keyword or opt_ai):
        st.info("중복 기준 버튼을 하나 이상 켜줘.")
        st.stop()

    if all_pairs:
        merged = {}
        for i, j, score, reason in all_pairs:
            key = (min(i, j), max(i, j))
            merged.setdefault(key, {"score": 0.0, "reasons": []})
            merged[key]["score"] = max(merged[key]["score"], float(score))
            merged[key]["reasons"].append(reason)

        final_pairs = [(i, j, v["score"], " / ".join(v["reasons"])) for (i, j), v in merged.items()]
        result_df = build_pairs_table(df, final_pairs).sort_values(["score"], ascending=False)

        st.subheader("⚠️ 중복 의심 결과")
        st.dataframe(result_df, use_container_width=True)
    else:
        st.success("🎉 선택한 기준에서는 중복 의심이 없어!")
