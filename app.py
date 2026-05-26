import streamlit as st
import requests
from googleapiclient.discovery import build
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util
import yt_dlp

# ----------------------------------------------------------------------------
# API keys from Streamlit secrets (set in cloud dashboard)
# ----------------------------------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ----------------------------------------------------------------------------
# Cached model loader
# ----------------------------------------------------------------------------
@st.cache_resource
def load_relevance_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

relevance_model = load_relevance_model()

# ----------------------------------------------------------------------------
# Cached summary generator (only runs once per unique inputs)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_summary(concept, video_titles, article_titles):
    """Cached summary using Groq. Inputs are simple types for cache hash."""
    prompt = f"""Based on these resources, write a concise 3-4 sentence summary of the concept '{concept}'.
Resources: {', '.join(video_titles + article_titles)}
Summary:"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200
    }
    resp = requests.post(GROQ_URL, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return "Could not generate summary."

# ----------------------------------------------------------------------------
# Core AI functions (unchanged from your Phase 2)
# ----------------------------------------------------------------------------
def get_micro_concepts(syllabus_text, exam="UPSC"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""You are an expert exam coach. Break the following {exam} syllabus
into a detailed list of micro-concepts. A micro-concept is a tiny, self-contained
5‑minute study unit. Return ONLY a Python list of strings, like:
["Concept 1", "Concept 2", ...]

Syllabus:
{syllabus_text}
"""
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500
    }
    response = requests.post(GROQ_URL, headers=headers, json=data)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        import ast
        try:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                list_str = content[start:end+1]
                return ast.literal_eval(list_str)
        except:
            pass
        lines = [line.strip("-•* ") for line in content.split("\n") if line.strip()]
        return lines
    else:
        raise Exception(f"Groq API error: {response.text}")

def search_youtube(query, max_results=5):
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,
        'force_generic_extractor': False,
        'skip_download': True,
        'ignoreerrors': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            if info is None or 'entries' not in info:
                return []
            videos = []
            for entry in info['entries']:
                if entry is None:
                    continue
                videos.append({
                    'title': entry.get('title', 'No title'),
                    'videoId': entry.get('id', ''),
                    'url': entry.get('webpage_url', f"https://www.youtube.com/watch?v={entry.get('id', '')}"),
                    'description': entry.get('description', ''),
                    'publishedAt': entry.get('upload_date', '')
                })
            return videos
    except Exception as e:
        print(f"yt-dlp failed: {e}")
        return []

def search_web(query, max_results=5):
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                'title': r['title'],
                'link': r['href'],
                'snippet': r['body']
            })
        return results

def score_resources(concept, resources, resource_type='youtube'):
    concept_embedding = relevance_model.encode(concept, convert_to_tensor=True)
    scored = []
    for res in resources:
        text = res['title']
        if resource_type == 'youtube':
            text += ' ' + res.get('description', '')
        else:
            text += ' ' + res.get('snippet', '')
        emb = relevance_model.encode(text, convert_to_tensor=True)
        similarity = util.cos_sim(concept_embedding, emb).item()
        scored.append({**res, 'score': similarity})
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored

# ----------------------------------------------------------------------------
# Streamlit UI – stateful expanders with lazy summaries
# ----------------------------------------------------------------------------
st.set_page_config(page_title="GurukulAI", layout="wide")
st.title("📚 GurukulAI – Free AI Study Planner for Govt Exams")
st.markdown("Paste your syllabus and get micro‑concepts with embedded videos, articles & AI summary.")

exam = st.selectbox("Select Exam", ["APPSC","UPSC", "SSC CGL", "IBPS PO","TGPSC" "NEET", "JEE Main"])
user_text = st.text_area("Paste syllabus text here:", height=300)

# Session state initialization
if "concepts" not in st.session_state:
    st.session_state.concepts = None
if "expanded_concepts" not in st.session_state:
    st.session_state.expanded_concepts = set()         # indices of expanders that should be open
if "summary_requests" not in st.session_state:
    st.session_state.summary_requests = set()          # indices for which summary was requested
if "summaries" not in st.session_state:
    st.session_state.summaries = {}                    # index -> summary text

if st.button("✨ Generate Study Plan"):
    if user_text.strip() == "":
        st.warning("Please paste some syllabus content first.")
    else:
        with st.spinner("AI breaking down syllabus..."):
            try:
                st.session_state.concepts = get_micro_concepts(user_text, exam)
                st.session_state.expanded_concepts = set()
                st.session_state.summary_requests = set()
                st.session_state.summaries = {}
            except Exception as e:
                st.error(f"Error generating concepts: {e}")
                st.session_state.concepts = None

if st.session_state.concepts:
    concepts = st.session_state.concepts
    st.success(f"Found {len(concepts)} micro‑concepts!")

    for i, concept in enumerate(concepts, 1):
        # Determine if this expander should be open (from previous summary click)
        is_expanded = i in st.session_state.expanded_concepts

        with st.expander(f"📌 {i}. {concept}", expanded=is_expanded):
            col1, col2 = st.columns(2)
            ranked_videos = []
            ranked_pages = []

            with col1:
                st.markdown("**🎥 Top 5 Videos**")
                with st.spinner("Fetching videos..."):
                    try:
                        videos = search_youtube(concept)
                        ranked_videos = score_resources(concept, videos, 'youtube')
                        ranked_videos = [v for v in ranked_videos if v['score'] > 0.2]
                        ranked_videos.sort(key=lambda x: x['publishedAt'], reverse=True)
                        for v in ranked_videos[:5]:
                            st.video(v['url'])
                            st.markdown(f"📺 [Watch on YouTube]({v['url']})")
                    except Exception as e:
                        st.warning(f"Couldn't load videos: {e}")

            with col2:
                st.markdown("**🌐 Top 5 Articles**")
                with st.spinner("Fetching articles..."):
                    try:
                        pages = search_web(concept)
                        ranked_pages = score_resources(concept, pages, 'web')
                        for p in ranked_pages[:5]:
                            st.markdown(f"- [{p['title']}]({p['link']})")
                    except Exception as e:
                        st.warning(f"Couldn't load articles: {e}")

            # ---- Lazy Summary (state‑preserving) ----
            st.markdown("---")
            # Button that triggers summary generation and records the expander state
            if st.button(f"🧠 Generate Summary for this concept", key=f"summary_btn_{i}"):
                # When clicked, add this concept's expander to the "open" set
                st.session_state.expanded_concepts.add(i)
                # Mark that summary was requested for this concept
                st.session_state.summary_requests.add(i)

            # If summary was requested, generate (if not already cached in session) and display
            if i in st.session_state.summary_requests:
                if i not in st.session_state.summaries:
                    with st.spinner("Generating summary..."):
                        # Prepare simple inputs for caching
                        vid_titles = [v['title'] for v in ranked_videos[:3]]
                        art_titles = [a['title'] for a in ranked_pages[:3]]
                        if vid_titles or art_titles:
                            summary = generate_summary(concept, tuple(vid_titles), tuple(art_titles))
                            st.session_state.summaries[i] = summary
                        else:
                            st.session_state.summaries[i] = "Not enough resources to summarise."
                st.info(st.session_state.summaries[i])
