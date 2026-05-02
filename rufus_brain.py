import os
import pickle
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_hybrid_retriever():
    """
    Loads and merges the local FAISS and BM25 stores into an EnsembleRetriever.
    Returns None if the indices are missing.
    """
    embeddings = OpenAIEmbeddings()
    
    try:
        # allow_dangerous_deserialization is required for local FAISS loading in LangChain >= 0.1.0
        vectorstore = FAISS.load_local(
            "vector_store", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    except Exception as e:
        print(f"[ERROR] FAISS index initialization failed: {e}")
        return None

    try:
        with open("retriever_store/bm25_retriever.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 3 
    except Exception as e:
        print(f"[ERROR] BM25 index initialization failed: {e}")
        return None

    # Merge retrievers: 50% semantic meaning, 50% exact keyword matching
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def get_rufus_chain():
    """
    Constructs the core RAG chain, binding the hybrid retriever to the LLM.
    Configured for token streaming to reduce perceived latency in the UI.
    """
    retriever = load_hybrid_retriever()
    if not retriever:
        return None

    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo", streaming=True)

    system_prompt = (
        "You are Rufus, Amazon's elite AI shopping assistant. "
        "Use the following pieces of retrieved product context to answer the user's question.\n"
        "Rules for your response:\n"
        "1. Be direct, helpful, and objective. Do not sound like a pushy salesperson.\n"
        "2. If the user asks about side effects, negative aspects, or complaints, tell them the truth based ONLY on the customer reviews provided.\n"
        "3. If you don't know the answer based on the context, say 'I don't have enough information on that specific detail.' DO NOT hallucinate external facts.\n"
        "4. Keep it concise, exactly how a modern AI assistant would reply on a mobile screen.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    # Diagnostic Health Check for local testing
    print("\n[INFO] Running Rufus Brain Diagnostic Check...")
    
    test_chain = get_rufus_chain()
    
    if test_chain:
        print("[SUCCESS] Rufus Brain backend initialized perfectly.")
        print("[INFO] Hybrid Retrievers (FAISS + BM25) loaded successfully.")
        print("[INFO] LLM Chain and system prompts are configured.")
        print("\n▶ To interact with Rufus, run the UI with: `streamlit run app.py`\n")
    else:
        print("\n[FAILED] Rufus Brain initialization failed.")
        print("[ACTION] Please ensure you have run `python data_ingestion.py` first to generate the local vector stores.\n")