# 🛒 Amazon Rufus Twin (V3)

A reverse-engineered, production-grade AI shopping assistant with verifiable source citations.

This application scrapes live Amazon product data and uses a sophisticated Hybrid Retrieval-Augmented Generation (RAG) pipeline to answer user queries with strict grounding, preventing LLM hallucinations.

## 🚀 Architecture Evolution

**V1: Base Retrieval (Stable Foundation)** 
* Engineered a reliable RAG pipeline using **FAISS (Dense Vector Search)** to retrieve semantic context from third-party scraped Amazon reviews and product specifications.
* Implemented basic LangChain LLM chains for Q&A.

**V2: Performance & Accuracy Sprint (UX & Retrieval Polish)**
* **Hybrid Search (Dense + Sparse):** Integrated `rank_bm25` alongside FAISS in an `EnsembleRetriever` (50/50 weight). This ensures exact-keyword queries (e.g., specific ASINs, "public review", or precise ingredients) are caught just as effectively as broad semantic concepts.
* **Token Streaming UI:** Upgraded the Streamlit frontend from static block-rendering to real-time chunk streaming (`st.write_stream`), drastically reducing perceived latency.
* **Resource Caching:** Implemented `@st.cache_resource` on the backend initialization to prevent memory reloading on every user interaction.

**V3: Production MVP (Current)**
* **Stealth LLM Extraction:** Replaced rigid third-party APIs (Apify) with a custom `cloudscraper` + `BeautifulSoup` pipeline to bypass basic WAFs. Fed raw DOM text into a strict, JSON-bound LLM (`gpt-3.5-turbo`) to enforce a highly predictable data schema.
* **Defensive Prompt Engineering:** Implemented a zero-temperature strict system prompt that forces structured Markdown table comparisons for similar products, and explicitly blocks cross-category hallucinations (e.g., attempting to compare a Bluetooth speaker to a Magnesium supplement).

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** Python, LangChain
* **Vector Store / Indexing:** FAISS, BM25 (`rank_bm25`)
* **LLM & Embeddings:** OpenAI (`gpt-3.5-turbo`, `text-embedding-ada-002`)
* **Data Extraction:** Cloudscraper, BeautifulSoup4 (Replaces Apify)

## 💻 Local Setup
1. Clone the repository: `git clone https://github.com/himanshu-kr7/rufus_twin.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your OpenAI API key:
   `OPENAI_API_KEY=your_openai_key`
4. Run the Extractor (Bypasses WAF & parses JSON): `python data_extractor.py`
5. Build the Hybrid Indices: `python data_ingestion.py`
6. Run the Application: `streamlit run app.py`

## 🔮 Next Sprint (V4 Roadmap)
* Implement LangChain's `create_history_aware_retriever` to support multi-turn conversational memory and complex pronoun resolution.