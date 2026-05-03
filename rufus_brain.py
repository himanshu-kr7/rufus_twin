import os
import pickle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (OpenAI API Key)
load_dotenv()

def load_hybrid_retriever():
    """Loads both FAISS (Dense) and BM25 (Sparse) retrievers for Hybrid Search."""
    # 1. Load FAISS Vector Store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    # Set to 5 for the MVP demo to ensure all products can be listed
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Load BM25 Keyword Store
    with open("retriever_store/bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 5

    # 3. Combine into an Ensemble Retriever (50% Dense, 50% Sparse)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], 
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

def get_rufus_chain():
    """Processes the user query through the V3 Hybrid RAG pipeline with streaming."""
    
    # Initialize the LLM with streaming enabled
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
    
    
    # The V3 System Prompt: Optimized for formatting and engagement
    system_prompt = (
        "You are Rufus, an expert AI shopping assistant. "
        "Use the provided context to answer questions about any products in the database, "
        "including electronics (like Bluetooth speakers) and health supplements (like Magnesium). "
        "Compare features objectively when asked to help the user make a purchasing decision. "
        "TAXONOMY KNOWLEDGE: "
        "- Bluetooth Speakers: Bose SoundLink Flex, Anker Soundcore 2. "
        "- Health Supplements: Natural Vitality Calm, Doctor's Best Magnesium, Intelligent Labs MagEnhance. "
        "COMPARISON RULES: "
        "1. Same Category: When comparing products WITHIN the same category, always present the comparison using a clean Markdown table. "
        "2. Cross-Category: If a user asks to compare products ACROSS different categories (e.g., a speaker and a supplement), DO NOT generate a table. Politely inform them that these items belong to different categories and cannot be compared. "
        "ENGAGEMENT RULE: Always end your response with a brief, relevant follow-up question to help the user narrow down their choice. "
        "If the answer to the user's question is not in the context, say you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build the LangChain Expression Language (LCEL) Chain
    retriever = load_hybrid_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Return the chain itself so Streamlit can pull data from it token-by-token
    return rag_chain

if __name__ == "__main__":
    print("\n[INFO] Booting up Rufus Brain for terminal testing...")
    try:
        # Initialize the chain to ensure no syntax or loading errors
        rufus_chain = get_rufus_chain()
        print("[SUCCESS] Hybrid Retriever and LLM Chain loaded perfectly.")
        print("[SUCCESS] Brain test complete! Ready for Streamlit UI.\n")
        
    except Exception as e:
        print(f"\n[ERROR] Brain initialization failed: {e}")