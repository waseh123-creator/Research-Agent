# CrewAI Search Agent

Search agent built with CrewAI, DuckDuckGo (no key needed) and Groq (free API key).

## Deploy
1. Upload `app.py`, `requirements.txt`, `README.md` to a GitHub repo (root folder).
2. On share.streamlit.io: New app -> pick repo -> main file `app.py`.
3. Advanced settings -> Python version **3.11** -> Secrets:

```
GROQ_API_KEY = "gsk_your_key_here"
```

4. Deploy.
