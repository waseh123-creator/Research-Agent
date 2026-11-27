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
from crewai.tools import tool

# DuckDuckGo library: new name is "ddgs", old name is "duckduckgo_search"
try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS


# ----------------------------------------------------------------------------
# DuckDuckGo tools (no API key needed)
# ----------------------------------------------------------------------------
@tool("DuckDuckGo Web Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input is a plain-text search query.
    Returns titles, links and short snippets of the top results."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        link = r.get("href", r.get("url", ""))
        body = (r.get("body", "") or "")[:300]
        lines.append(f"{i}. {title}\n   URL: {link}\n   {body}")
    return "\n\n".join(lines)


@tool("DuckDuckGo News Search")
def news_search(query: str) -> str:
    """Search recent news with DuckDuckGo. Input is a plain-text search query.
    Returns headlines, sources, dates and links."""
    try:
        results = DDGS().news(query, max_results=5)
    except Exception as e:
        return f"News search failed: {e}"
    if not results:
        return "No news found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        link = r.get("url", r.get("href", ""))
        source = r.get("source", "")
        date = r.get("date", "")
        body = (r.get("body", "") or "")[:250]
        lines.append(f"{i}. {title} ({source}, {date})\n   URL: {link}\n   {body}")
    return "\n\n".join(lines)


# ----------------------------------------------------------------------------
# Crew builder
# ----------------------------------------------------------------------------
def run_search_agent(topic: str, api_key: str, model_name: str) -> str:
    llm = LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=0.2,
        max_tokens=1500,
    )

    researcher = Agent(
        role="Web Researcher",
        goal="Find accurate, up-to-date information about: {topic}",
        backstory=(
            "You are a careful researcher. You search the web with DuckDuckGo, "
            "compare sources, and never invent facts or links."
        ),
        tools=[web_search, news_search],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=5,
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
            "Research the topic: {topic}. Use the search tools (1 to 3 searches max). "
            "Collect the key facts and the source URLs."
        ),
        expected_output="Bullet-point findings with a source URL for each point.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the research findings, write a clear answer about: {topic}. "
            "Start with a short summary, then key points, then a 'Sources' list."
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

    result = crew.kickoff(inputs={"topic": topic})
    return getattr(result, "raw", str(result))


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="CrewAI Search Agent", page_icon="🔎", layout="centered")
st.title("🔎 CrewAI Search Agent")
st.caption("CrewAI + DuckDuckGo search + Groq LLM")

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
    model_name = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
    )

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
