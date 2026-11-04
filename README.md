# IUB Policy Navigator

A Streamlit chatbot that retrieves IU policy passages, drafts an answer, and
checks it once against the same evidence. Conversation recall uses session
history without calling the API.

Memory belongs to the current browser session; reloading the page or opening a
new session starts a new conversation. Recall supports first, second, third,
previous, and all earlier user messages. Recall requests themselves are skipped
when selecting those messages. Policy follow-ups retain their original topic,
and answer generation receives the full saved conversation, not just two turns.

## Run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src/submit.py
```

Put your NRP token in the project `.env` as `NRP_API_KEY=your-token`.
The existing `OPENAI_API_KEY` setting is also supported and takes precedence.
Never commit `.env`.

The first policy question builds a local index in `.chroma/`. This can take
several minutes. Completed batches are saved, so retrying after an API failure
continues from the last completed batch. Later launches reuse the index;
changing the corpus or embedding model creates a fresh one.

Answers are limited to the bundled corpus and may be incomplete or outdated.
Review the displayed passages before relying on an answer.

## Check

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Tests cover indexing and recovery, recall, follow-ups, grounding failures,
source persistence, embedding order, and credential loading without live API calls.

The November 4, 2026 dependency audit found unpatched Chroma server advisories
([upstream issue](https://github.com/chroma-core/chroma/issues/7909)). This app
uses an embedded `PersistentClient` and does not expose the Chroma HTTP server.
The audit is not fully clean; reassess those advisories before deploying a
separate Chroma server. Fixable dependencies have minimum versions in
`requirements.txt`.
