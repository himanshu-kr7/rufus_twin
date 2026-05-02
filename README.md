# 🛒 Amazon Rufus Twin

A reverse-engineered, production-grade AI shopping assistant with verifiable source citations. 

This application scrapes live Amazon product data and uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline to answer user queries with strict grounding, preventing LLM hallucinations.

## 🚀 Architecture Evolution

**V1: Base Retrieval (Stable Foundation)** 
* Engineered a reliable RAG pipeline using **FAISS (Dense Vector Search)** to retrieve semantic context from Apify-scraped Amazon reviews and product specifications.
* Implemented basic LangChain LLM chains for Q&A.

**V2: Performance & Accuracy Sprint (Current)**
* **Hybrid Search (Dense + Sparse):** Integrated `rank_bm25` alongside FAISS in an `EnsembleRetriever` (50/50 weight). This ensures exact-keyword queries (e.g., specific ASINs, "public review", or precise ingredients) are caught just as effectively as broad semantic concepts.
* **Token Streaming UI:** Upgraded the Streamlit frontend from static block-rendering to real-time chunk streaming (`st.write_stream`), drastically reducing perceived latency.
* **Strict Hallucination Guardrails:** Enforced explicit system prompts that block the LLM from providing unauthorized medical advice or fabricating features not present in the ingested JSON context.
* **Resource Caching:** Implemented `@st.cache_resource` on the backend initialization to prevent memory reloading on every user interaction.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** Python, LangChain
* **Vector Store / Indexing:** FAISS, BM25 (`rank_bm25`)
* **LLM & Embeddings:** OpenAI (`gpt-3.5-turbo`, `text-embedding-ada-002`)
* **Data Ingestion:** Apify (Amazon Crawler)

## 💻 Local Setup
1. Clone the repository: `git clone [https://github.com/himanshu-kr7/rufus_twin.git](https://github.com/himanshu-kr7/rufus_twin.git)`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your API keys:
   `OPENAI_API_KEY=your_openai_key`
   `APIFY_API_TOKEN=your_apify_key`
4. Build the Hybrid Indices: `python data_ingestion.py`
5. Run the Application: `streamlit run app.py`


## 🔮 Next Sprint (V3 Roadmap)
* Implement LangChain's `create_history_aware_retriever` to support multi-turn conversational memory and complex pronoun resolution.