import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_rufus_response(user_query):
    """Retrieves relevant context from FAISS and generates an LLM response with verifiable citations."""
    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = FAISS.load_local(
            "vector_store", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        return "Error: Vector store not found. Please run data_ingestion.py first.", []

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")

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

    response = rag_chain.invoke({"input": user_query})
    
    return response["answer"], response["context"]

if __name__ == "__main__":
    print("\n[INFO] 🤖 Rufus Brain Initialized. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
        print("[INFO] Rufus is thinking...")
        answer, sources = get_rufus_response(query)
        print(f"\nRufus: {answer}")
        print(f"\n[SUCCESS] Sources Cited: {len(sources)} documents")