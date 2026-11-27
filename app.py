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
def run_search_agent(topic: str, api_key: str, model_name: str) -> str:
    search_results = gather_search_results(topic)

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


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="CrewAI Search Agent", page_icon="🔎", layout="centered")
st.title("🔎 CrewAI Search Agent")
st.caption("CrewAI + DuckDuckGo search + Groq LLM")


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

# Get the Groq key from Streamlit secrets first, otherwise from the sidebar
secret_key = ""
try:
    secret_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    secret_key = ""

with st.sidebar:
    st.header("Settings")
    if secret_key:
        st.success("Groq API key loaded from secrets")
        api_key = secret_key
    else:
        api_key = st.text_input("Groq API key", type="password", help="Get it free at console.groq.com/keys")
    model_name = st.selectbox("Groq model", get_models(api_key), index=0)

query = st.text_input("What do you want to search?", placeholder="e.g. Latest developments in solar energy")

if st.button("Search", type="primary"):
    if not api_key:
        st.error("Please add your Groq API key (sidebar or Streamlit secrets).")
    elif not query.strip():
        st.warning("Please type something to search.")
    else:
        with st.spinner("Agents are searching and writing..."):
            try:
                answer = run_search_agent(query.strip(), api_key.strip(), model_name)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("If this is a rate limit error, wait a minute and try again, or switch the Groq model.")
