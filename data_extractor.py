import os
import json
import time
import cloudscraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def extract_with_llm():
    print("\n[INFO] Initializing Stealth LLM Extractor...")
    
    # Initialize Cloudscraper (Bypasses basic anti-bot WAFs)
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
    
    # Initialize the LLM (Using gpt-3.5-turbo to save cost, but forcing JSON output)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind(response_format={"type": "json_object"})
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data extraction algorithm. Extract the product details from the raw HTML text provided. "
                   "You MUST return a valid JSON object with the following exact keys: 'title', 'price', 'description', and 'features' (an array of strings). "
                   "If you cannot find a price, set it to 'Currently unavailable'. Strip out all promotional garbage."),
        ("human", "ASIN: {asin}\n\nRAW TEXT: {raw_text}")
    ])
    
    extraction_chain = extraction_prompt | llm

# The 5 validated MVP products
    target_asins = [
        "B00BPUY3W0", "B000BD0RT0", "B073WZ9SM3", # Magnesium
        "B099T738ZC", "B01MTB55WH"                # Speakers
    ]

    extracted_data = []

    for asin in target_asins:
        url = f"https://www.amazon.com/dp/{asin}"
        print(f"[FETCHING] ASIN: {asin}...")
        
        try:
            # 1. Fetch raw DOM
            response = scraper.get(url, timeout=10)
            
            # Professional HTTP Status Handling
            if response.status_code == 404:
                print(f"  ⚠️  Product Not Found (Status: 404). Dead ASIN.")
                continue
            elif response.status_code in [403, 503]:
                print(f"  ❌ Blocked by Amazon WAF (Status: {response.status_code})")
                continue
            elif response.status_code != 200:
                print(f"  ❌ HTTP Error (Status: {response.status_code})")
                continue
                
            # 2. Clean HTML into raw text
            soup = BeautifulSoup(response.text, 'html.parser')
            raw_text = soup.get_text(separator=' ', strip=True)
            
            # Truncate text to fit within LLM context window
            truncated_text = raw_text[:8000]
            
            # 3. Pass to LLM for JSON structuring
            print(f"  🧠 Parsing with LangChain...")
            result = extraction_chain.invoke({"asin": asin, "raw_text": truncated_text})
            
            # Parse the JSON string returned by the LLM
            product_json = json.loads(result.content)
            product_json["url"] = url 
            product_json["asin"] = asin
            
            extracted_data.append(product_json)
            print(f"  ✅ Success: {product_json.get('title', 'Unknown')[:30]}...")
            
            # Sleep to prevent rate-limiting WAF bans
            time.sleep(3)
            
        except Exception as e:
            print(f"  ❌ Error processing {asin}: {e}")

    # Save final dataset
    output_file = "amazon_raw_data_v3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)
        
    print(f"\n[COMPLETE] Successfully engineered data for {len(extracted_data)}/{len(target_asins)} products.")
    print(f"[INFO] Saved to {output_file}. Run data_ingestion.py to update FAISS.")

if __name__ == "__main__":
    extract_with_llm()