import os
import sys

# Must be set BEFORE importing crewai (turns off telemetry, avoids network noise/errors)
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"

# Streamlit Cloud safety: make sure a modern sqlite3 is used (needed by chromadb inside crewai)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# DuckDuckGo library: new name is "ddgs", old name is "duckduckgo_search"
try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS


# ----------------------------------------------------------------------------
# DuckDuckGo search (plain Python, no API key, no LLM tool-calling needed)
# ----------------------------------------------------------------------------
def gather_search_results(query: str) -> str:
    parts = []
    try:
        results = DDGS().text(query, max_results=6) or []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            link = r.get("href", r.get("url", ""))
            body = (r.get("body", "") or "")[:350]
            parts.append(f"[Web {i}] {title}\nURL: {link}\n{body}")
    except Exception as e:
        parts.append(f"(Web search failed: {e})")
    try:
        news = DDGS().news(query, max_results=4) or []
        for i, r in enumerate(news, 1):
            title = r.get("title", "")
            link = r.get("url", r.get("href", ""))
            source = r.get("source", "")
            date = r.get("date", "")
            body = (r.get("body", "") or "")[:250]
            parts.append(f"[News {i}] {title} ({source}, {date})\nURL: {link}\n{body}")
    except Exception:
        pass
    return "\n\n".join(parts) if parts else "No results found."


# ----------------------------------------------------------------------------
# Crew builder
# ----------------------------------------------------------------------------
def run_search_agent(topic: str, search_results: str, api_key: str, model_name: str) -> str:

    llm = LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=0.2,
        max_tokens=3000,
    )

    researcher = Agent(
        role="Web Researcher",
        goal="Extract the key facts about the topic from the provided DuckDuckGo search results",
        backstory=(
            "You are a careful researcher. You only use the search results you are given, "
            "compare sources, and never invent facts or links."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    writer = Agent(
        role="Report Writer",
        goal="Turn research findings into a clear, well-organized answer",
        backstory=(
            "You write concise, easy-to-read summaries and always list the "
            "source links that were found by the researcher."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    research_task = Task(
        description=(
            "Topic: " + topic + "\n\n"
            "Here are the DuckDuckGo search results:\n\n" + search_results + "\n\n"
            "Pick the most relevant and reliable facts from these results."
        ),
        expected_output="Bullet-point findings, each with its source URL.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the research findings, write a clear answer about: " + topic + ". "
            "Start with a short summary, then key points, then a 'Sources' list with URLs."
        ),
        expected_output="A markdown answer with summary, key points and sources.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        memory=False,
        verbose=False,
    )

    result = crew.kickoff()
    return getattr(result, "raw", str(result))


FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_models(key: str):
    """Ask Groq which chat models exist right now, so retired models never break the app."""
    if not key:
        return FALLBACK_MODELS
    try:
        import requests
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        ids = [m["id"] for m in r.json().get("data", [])]
        skip = ("whisper", "orpheus", "guard", "safeguard", "compound", "tts", "embed")
        ids = [i for i in ids if not any(w in i for w in skip)]
        preferred = [m for m in FALLBACK_MODELS if m in ids]
        rest = sorted(i for i in ids if i not in preferred)
        return (preferred + rest) or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Search Agent", page_icon="🔎", layout="wide")

CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    padding: 2.2rem 2rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35);
}
.hero h1 { color: white; margin: 0; font-size: 2.4rem; }
.hero p { color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.05rem; }
.badges { margin-top: 0.9rem; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 0.25rem 0.8rem; border-radius: 999px;
    margin: 0 0.25rem; font-size: 0.85rem;
}
.answer-card {
    background: rgba(124, 58, 237, 0.06);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-left: 6px solid #7c3aed;
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    margin-top: 1rem;
}
div.stButton > button {
    border-radius: 12px;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>🔎 AI Search Agent</h1>
        <p>Ask anything. Two AI agents search the web and write you a sourced answer.</p>
        <div class="badges">
            <span class="badge">CrewAI</span>
            <span class="badge">DuckDuckGo</span>
            <span class="badge">Groq</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Get the Groq key from Streamlit secrets first, otherwise from the sidebar
secret_key = ""
try:
    secret_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    secret_key = ""

if "history" not in st.session_state:
    st.session_state.history = []
if "query" not in st.session_state:
    st.session_state.query = ""


def set_query(text):
    st.session_state.query = text


with st.sidebar:
    st.header("⚙️ Settings")
    if secret_key:
        st.success("Groq API key loaded")
        api_key = secret_key
    else:
        api_key = st.text_input("Groq API key", type="password", help="Get it free at console.groq.com/keys")
    model_name = st.selectbox("Groq model", get_models(api_key), index=0)

    st.divider()
    st.subheader("🕘 Recent searches")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-8:]):
            st.button(item, key=f"hist_{item}", on_click=set_query, args=(item,), use_container_width=True)
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Your searches will appear here.")

st.text_input(
    "What do you want to know?",
    key="query",
    placeholder="e.g. Latest developments in solar energy",
)

st.caption("Try an example:")
examples = [
    "Latest AI news this week",
    "Best programming languages to learn in 2026",
    "How does solar energy work?",
    "Top tourist places in Pakistan",
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    col.button(ex, key=f"ex_{ex}", on_click=set_query, args=(ex,), use_container_width=True)

if st.button("🚀 Search", type="primary", use_container_width=True):
    query = st.session_state.query.strip()
    if not api_key:
        st.error("Please add your Groq API key (sidebar or Streamlit secrets).")
    elif not query:
        st.warning("Please type something to search.")
    else:
        try:
            with st.status("Working on it...", expanded=True) as status:
                st.write("🔎 Searching DuckDuckGo...")
                search_results = gather_search_results(query)
                st.write("🤖 Agents are analyzing and writing...")
                answer = run_search_agent(query, search_results, api_key.strip(), model_name)
                status.update(label="Done!", state="complete", expanded=False)

            if query not in st.session_state.history:
                st.session_state.history.append(query)

            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.download_button(
                "⬇️ Download answer (.md)",
                data=answer,
                file_name="search_answer.md",
                mime="text/markdown",
                use_container_width=True,
            )
            with st.expander("📚 Raw DuckDuckGo results"):
                st.text(search_results)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("If this is a rate limit error, wait a minute and try again, or switch the Groq model.")

st.markdown("---")
st.caption("Built with CrewAI, DuckDuckGo and Groq · Streamlit")
