import os
import json
from dotenv import load_dotenv
from apify_client import ApifyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

def scrape_amazon_data():
    """Scrapes Amazon product specifications and reviews using Apify."""
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

    print("[INFO] Scraping live Amazon data (this may take 60-90 seconds)...")
    run = client.actor("junglee/amazon-crawler").call(run_input=run_input)
    
    dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    print(f"[SUCCESS] Successfully extracted {len(dataset_items)} products from Apify.")
    
    with open("amazon_raw_data.json", "w") as f:
        json.dump(dataset_items, f, indent=4)
    
    return "amazon_raw_data.json"

def build_vector_store(json_file_path):
    """Processes raw JSON data, chunks it, and builds a FAISS vector index."""
    print("[INFO] Chunking data and generating OpenAI embeddings...")
    with open(json_file_path, "r") as f:
        data = json.load(f)
        
    texts = []
    metadatas = []
    
    for item in data:
        title = item.get("title", "")
        asin = item.get("asin", "")
        features = item.get("features", [])
        reviews = item.get("reviews", [])
        
        # Ingest Product Specifications
        product_info = f"Product: {title}\nASIN: {asin}\nFeatures: {' '.join(features)}"
        texts.append(product_info)
        metadatas.append({"source": asin, "type": "specs"})
        
        # Ingest Customer Reviews
        for review in reviews:
            review_text = review.get("reviewText", "")
            rating = review.get("rating", "")
            if review_text:
                texts.append(f"Review for {title} ({rating} stars): {review_text}")
                metadatas.append({"source": asin, "type": "review"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(texts, metadatas=metadatas)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    os.makedirs("vector_store", exist_ok=True)
    vectorstore.save_local("vector_store")
    print("[SUCCESS] Local FAISS Vector Store built and saved successfully!\n")

if __name__ == "__main__":
    data_file = scrape_amazon_data()
    build_vector_store(data_file)