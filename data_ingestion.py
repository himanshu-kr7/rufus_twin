import os
import json
import pickle
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Load environment variables
load_dotenv()

def build_hybrid_indices(json_file_path):
    """
    Constructs dual retrieval indices (Dense FAISS and Sparse BM25) from the extracted Amazon JSON data
    """
    print(f"\n[INFO] Processing data from {json_file_path} for Hybrid RAG...")
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    texts = []
    metadatas = []
    
    for item in data:
        # Extract fields based on the new V3 ScraperAPI structure
        title = item.get("title", "Unknown Title")
        url = item.get("url", "")
        price = item.get("price", "No price listed")
        description = item.get("description", "")
        features = item.get("features", [])
        
        # Extract ASIN dynamically from the URL for metadata tagging
        asin = url.split("/")[-1] if url else "Unknown"
        
        # Format features nicely whether it's a list or a string
        features_text = " ".join(features) if isinstance(features, list) else str(features)
        
        # 1. Isolate product specifications
        product_info = f"Product: {title}\nPrice: {price}\nASIN: {asin}\nFeatures: {features_text}\nDescription: {description}"
        texts.append(product_info)
        metadatas.append({"source": asin, "type": "specs"})
        
        # 2. Safety check for reviews (if you add them to the ScraperAPI pipeline later)
        reviews = item.get("reviews", [])
        for review in reviews:
            review_text = review if isinstance(review, str) else review.get("reviewText", "")
            if review_text:
                texts.append(f"Review for {title}: {review_text}")
                metadatas.append({"source": asin, "type": "review"})

    print(f"[INFO] Chunking {len(texts)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(texts, metadatas=metadatas)

    # --- Build Dense Vector Search (FAISS) ---
    print("[INFO] Building FAISS Vector Index...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_store")

    # --- Build Sparse Keyword Search (BM25) ---
    print("[INFO] Building BM25 Keyword Index...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    os.makedirs("retriever_store", exist_ok=True)
    with open("retriever_store/bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

    print("[SUCCESS] Hybrid indices built and saved successfully!\n")

if __name__ == "__main__":
    # V3 Pipeline: Directing to the newly generated dataset
    data_file = "amazon_raw_data_v3.json" 
    
    if not os.path.exists(data_file):
        print(f"[ERROR] '{data_file}' not found. Please run data_extractor.py first.")
    else:
        build_hybrid_indices(data_file)