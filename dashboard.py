import re
import time
from datetime import datetime, date as date_cls
from urllib.parse import urljoin

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    KST = None


# =====================
# 게시판 고정
# =====================
CLUB_ID = 28866679
MENU_ID = 178
BASE_URL = f"https://cafe.naver.com/f-e/cafes/{CLUB_ID}/menus/{MENU_ID}?viewType=L&page="

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# =====================
# 날짜 판별 (목록 기준)
# =====================
def is_target_date(date_text: str, target_date: date_cls) -> bool:
    """
    KST 기준:
    - 선택 날짜가 오늘이면: 'HH:MM' 형태만 수집
    - 그 외(과거 날짜)이면: 'YYYY.MM.DD' 정확히 일치만 수집
    """
    today_kst = datetime.now(KST).date() if KST else datetime.now().date()

    if target_date == today_kst:
        return bool(re.match(r"^\d{1,2}:\d{2}$", date_text))

    return date_text == target_date.strftime("%Y.%m.%d")


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
    """
    아주 단순 토큰화(한/영/숫자만 남기고 분리)
    - 키워드 중복용
    """
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣 ]+", " ", s)
    parts = [p for p in s.split() if len(p) >= 2]
    return parts


# =====================
# 목록 수집
# =====================
def collect_article_list(target_date: date_cls, max_pages: int = 30) -> list[dict]:
    articles = []
    target_str = target_date.strftime("%Y.%m.%d")
    for page in range(1, max_pages + 1):
        res = requests.get(BASE_URL + str(page), headers=HEADERS, timeout=20)
        if res.status_code != 200:
            break

        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("tr")

        stop_flag = False

        for row in rows:
            title_tag = row.select_one("a.article")
            date_tag = row.select_one("td.td_date")
            author_tag = row.select_one("td.td_name")  # 작성자(목록에 보이는 경우)

            if not title_tag or not date_tag:
                continue

            date_text = date_tag.get_text(strip=True)

            # ✅ 날짜 필터는 "목록에서만" 적용
            if not is_target_date(date_text, target_date):
                # 과거 날짜의 경우, 더 아래(더 옛날)로 내려가면 중단
                # date_text가 YYYY.MM.DD일 때만 비교
                if re.match(r"^\d{4}\.\d{2}\.\d{2}$", date_text) and date_text < target_str:
                    stop_flag = True
                continue

            article_url = urljoin("https://cafe.naver.com", title_tag.get("href", ""))
            title = title_tag.get_text(strip=True)
            author = author_tag.get_text(strip=True) if author_tag else ""

            articles.append(
                {
                    "title": title,
                    "title_norm": normalize_title(title),
                    "author": author,
                    "url": article_url,
                    "date": date_text,
                }
            )

        if stop_flag:
            break

        time.sleep(0.25)

    return articles


# =====================
# 본문 수집
# =====================
def fetch_content(url: str) -> str:
    try:
        res = requests.get(url, headers=HEADERS, timeout=25)
        if res.status_code != 200:
            return ""
        soup = BeautifulSoup(res.text, "html.parser")

        iframe = soup.select_one("iframe#cafe_main")
        if iframe and iframe.get("src"):
            iframe_url = urljoin("https://cafe.naver.com", iframe["src"])
            res2 = requests.get(iframe_url, headers=HEADERS, timeout=25)
            if res2.status_code != 200:
                return ""
            soup = BeautifulSoup(res2.text, "html.parser")

        content = soup.select_one("div.se-main-container")
        if not content:
            # 구형 에디터 fallback
            content = soup.select_one("div#postViewArea") or soup.select_one("div.ContentRenderer")

        if not content:
            return ""

        return content.get_text(" ", strip=True)

    except Exception:
        return ""


# =====================
# 중복 판정들
# =====================
def dup_by_author(df: pd.DataFrame):
    # 같은 작성자 그룹(2개 이상)
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
    """
    제목+본문 토큰으로 Jaccard 유사도
    """
    token_sets = []
    for _, r in df.iterrows():
        tokens = simple_tokens(f"{r.get('title','')} {r.get('content','')}")
        token_sets.append(set(tokens))

    pairs = []
    n = len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            score = inter / union if union else 0.0
            if score >= jaccard_threshold:
                pairs.append((i, j, score, f"키워드 중복(Jaccard {score:.2f})"))
    return pairs


def dup_by_ai(df: pd.DataFrame, threshold: float = 0.7):
    """
    TF-IDF cosine similarity
    """
    texts = df["content"].fillna("").astype(str).tolist()
    # 너무 짧은 텍스트가 많으면 min_df=1로 완화
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
    """
    pairs: (i, j, score, reason)
    """
    rows = []
    for i, j, score, reason in pairs:
        rows.append(
            {
                "A_idx": i,
                "A_title": df.loc[i, "title"],
                "A_author": df.loc[i, "author"],
                "A_url": df.loc[i, "url"],
                "B_idx": j,
                "B_title": df.loc[j, "title"],
                "B_author": df.loc[j, "author"],
                "B_url": df.loc[j, "url"],
                "score": round(float(score), 3),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="클랜/방송/디스코드 중복검사", layout="wide")

# ✅ 화면 폭/높이 느낌을 맞추는 CSS (강제는 아니고 최대한 근접)
st.markdown(
    """
<style>
.block-container {max-width: 1400px;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📌 클랜/방송/디스코드 중복검사")

# --- 상단 버튼(토글) 영역 ---
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

# --- 설정 영역 ---
left, right = st.columns([1, 1])

with left:
    target_date = st.date_input(
        "📅 수집 날짜 선택 (KST 기준)",
        datetime.now(KST).date() if KST else datetime.now().date(),
    )

with right:
    max_pages = st.number_input("📄 최대 페이지 수", min_value=1, max_value=200, value=30, step=1)

# 기준별 임계치 옵션
with st.expander("⚙️ 중복 판정 옵션", expanded=False):
    ai_threshold = st.slider("🤖 AI 유사 임계치 (cosine)", 0.1, 0.99, 0.70, 0.01)
    kw_threshold = st.slider("🔎 키워드 중복 임계치 (Jaccard)", 0.1, 0.99, 0.60, 0.01)

st.divider()

# 세션 초기화
if "df" not in st.session_state:
    st.session_state["df"] = None

run = st.button("📥 게시글 수집 시작", type="primary")

if run:
    # 1) 목록 수집
    with st.spinner("게시글 목록 수집 중..."):
        articles = collect_article_list(target_date, max_pages=int(max_pages))

    if not articles:
        st.error("목록에서 해당 날짜 게시글을 찾지 못했어. (날짜/페이지 설정 확인)")
        st.stop()

    st.success(f"목록 수집 완료: {len(articles)}개")

    # 2) 본문 수집
    progress = st.progress(0.0)
    contents = []
    for i, art in enumerate(articles):
        contents.append(fetch_content(art["url"]))
        progress.progress((i + 1) / len(articles))
        time.sleep(0.15)

    df = pd.DataFrame(articles)
    df["content"] = contents

    st.session_state["df"] = df

# --- 결과 표시 ---
df = st.session_state.get("df")

if df is not None:
    st.subheader("✅ 수집 결과")

    if opt_original:
        st.dataframe(
            df[["date", "author", "title", "url"]].copy(),
            use_container_width=True,
            hide_index=True,
        )

    # 선택된 기준으로 중복쌍 만들기
    all_pairs = []
    if opt_author:
        all_pairs += dup_by_author(df)
    if opt_title:
        all_pairs += dup_by_title(df)
    if opt_keyword:
        all_pairs += dup_by_keywords(df, jaccard_threshold=float(kw_threshold))
    if opt_ai:
        all_pairs += dup_by_ai(df, threshold=float(ai_threshold))

    # 기준이 하나도 선택 안 됐을 때
    if not (opt_author or opt_title or opt_keyword or opt_ai):
        st.info("중복 기준 버튼을 하나 이상 켜줘.")
        st.stop()

    # 결과 정리(같은 (i,j) 중복 reason 합치기)
    if all_pairs:
        merged = {}
        for i, j, score, reason in all_pairs:
            key = (min(i, j), max(i, j))
            if key not in merged:
                merged[key] = {"score": score, "reasons": [reason]}
            else:
                merged[key]["score"] = max(merged[key]["score"], score)
                merged[key]["reasons"].append(reason)

        final_pairs = []
        for (i, j), v in merged.items():
            final_pairs.append((i, j, v["score"], " / ".join(v["reasons"])))

        result_df = build_pairs_table(df, final_pairs).sort_values(["score"], ascending=False)

        st.subheader("⚠️ 중복 의심 결과")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    else:
        st.success("🎉 선택한 기준에서는 중복 의심이 없어!")
