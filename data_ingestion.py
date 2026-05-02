import os
import json
import pickle
from dotenv import load_dotenv
from apify_client import ApifyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

def scrape_amazon_data():
    """
    Triggers the Apify actor to scrape live Amazon product data.
    Bypasses standard bot-protections to retrieve structured JSON containing specs and reviews.
    """
    print("\n[INFO] Initializing Apify Client...")
    client = ApifyClient(APIFY_TOKEN)

    run_input = {
        "categoryOrProductUrls": [
            {"url": "https://www.amazon.com/dp/B0797GZQZN"}, 
            {"url": "https://www.amazon.com/dp/B000BD0RT0"}, 
            {"url": "https://www.amazon.com/dp/B00BPUY3W0"}
        ],
        "maxItemsPerStartUrl": 1,
        "useCaptchaSolver": True,
        "extractReviews": True,
        "maxReviews": 20 
    }

    print("[INFO] Scraping live Amazon data...")
    run = client.actor("junglee/amazon-crawler").call(run_input=run_input)
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    
    with open("amazon_raw_data.json", "w") as f:
        json.dump(dataset_items, f, indent=4)
    
    print(f"[SUCCESS] Extracted {len(dataset_items)} products.")
    return "amazon_raw_data.json"

def build_hybrid_indices(json_file_path):
    """
    Constructs dual retrieval indices (Dense FAISS and Sparse BM25) from the raw JSON data
    to support hybrid search capabilities in the RAG pipeline.
    """
    print("[INFO] Processing data for Hybrid RAG...")
    with open(json_file_path, "r") as f:
        data = json.load(f)
        
    texts = []
    metadatas = []
    
    for item in data:
        title = item.get("title", "")
        asin = item.get("asin", "")
        features = item.get("features", [])
        reviews = item.get("reviews", [])
        
        # Isolate product specifications
        product_info = f"Product: {title}\nASIN: {asin}\nFeatures: {' '.join(features)}"
        texts.append(product_info)
        metadatas.append({"source": asin, "type": "specs"})
        
        # Isolate individual customer reviews
        for review in reviews:
            review_text = review.get("reviewText", "")
            if review_text:
                texts.append(f"Review for {title}: {review_text}")
                metadatas.append({"source": asin, "type": "review"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(texts, metadatas=metadatas)

    # 1. Build Dense Vector Search (FAISS)
    print("[INFO] Building FAISS Vector Index...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_store")

    # 2. Build Sparse Keyword Search (BM25)
    print("[INFO] Building BM25 Keyword Index...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    os.makedirs("retriever_store", exist_ok=True)
    with open("retriever_store/bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

    print("[SUCCESS] Hybrid indices built and saved successfully!\n")

if __name__ == "__main__":
    # Toggle data_file to scrape_amazon_data() to fetch fresh data, 
    # otherwise default to the local frozen dataset to preserve Apify credits.
    data_file = "amazon_raw_data.json" 
    build_hybrid_indices(data_file)