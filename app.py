import streamlit as st
from rufus_brain import get_rufus_chain

st.set_page_config(page_title="Rufus Twin V2 | AI Shopping Assistant", page_icon="🛒", layout="centered")

st.title("🛒 Amazon Rufus Twin")
st.caption("A reverse-engineered AI shopping assistant with verifiable source citations.")

# Cache the LLM chain to prevent memory reloading on every user interaction
@st.cache_resource
def load_brain():
    return get_rufus_chain()

rag_chain = load_brain()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Rufus. Ask me anything about the Magnesium Supplements in my database, and I'll cite my sources!"}
    ]

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=100)
    st.markdown("### Rufus Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Rufus. Ask me anything about the Magnesium Supplements in my database, and I'll cite my sources!"}
        ]
        st.rerun()
        
# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources"):
                for idx, doc in enumerate(message["sources"]):
                    doc_type = doc.metadata.get('type', 'document').capitalize()
                    asin = doc.metadata.get('source', 'Unknown ASIN')
                    st.caption(f"{idx + 1}. {doc_type} (ASIN: {asin})")

if prompt := st.chat_input("Ask about specific features, exact prices, or reviews..."):
    if not rag_chain:
        st.error("System Error: Retrieval stores not initialized. Run data_ingestion.py first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        sources = []
        
        for chunk in rag_chain.stream({"input": prompt}):
            if "context" in chunk:
                sources = chunk["context"]
            
            if "answer" in chunk:
                full_answer += chunk["answer"]
                # Display text chunks instantly with a blinking cursor effect
                answer_placeholder.markdown(full_answer + "▌")
        
        answer_placeholder.markdown(full_answer)
        
        if sources:
            with st.expander("🔍 View Sources"):
                for idx, doc in enumerate(sources):
                    doc_type = doc.metadata.get('type', 'document').capitalize()
                    asin = doc.metadata.get('source', 'Unknown ASIN')
                    st.caption(f"{idx + 1}. {doc_type} (ASIN: {asin})")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_answer,
        "sources": sources
    })