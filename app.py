import streamlit as st
from rufus_brain import get_rufus_chain

st.set_page_config(page_title="Rufus Twin V3 | AI Shopping Assistant", page_icon="🛒", layout="centered")

st.title("🛒 Rufus Twin v3")
st.caption("A reverse-engineered AI shopping assistant with verifiable source citations.")

# Cache the brain so it doesn't reload the FAISS database on every single keystroke
@st.cache_resource
def load_brain():
    return get_rufus_chain()

rag_chain = load_brain()

# Define the onboarding message
greeting_msg = (
    "Hi! I'm Rufus Twin V3, an AI shopping assistant engineered for hallucination-free retrieval. \n\n"
    "My current demo database contains verifiable specs for **Bluetooth Speakers** and **Magnesium Supplements**. \n\n"
    "**Try asking me to:**\n"
    "* *'Compare the Bose and Anker speakers'*\n"
    "* *'Recommend a magnesium supplement for sleep'*"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": greeting_msg}
    ]

# Sidebar Controls
with st.sidebar:
    # Replaced the Amazon logo with a clean text header
    st.markdown("## 🛒 Rufus Twin V3")
    st.caption("Pixii Evaluation Environment")
    st.markdown("---")
    
    st.markdown("### Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": greeting_msg}
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

# Chat Input & Processing
if prompt := st.chat_input("Ask about specific features, exact prices, or reviews..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        sources = []
        
        # The Streaming Loop: Pulls tokens live from OpenAI
        for chunk in rag_chain.stream({"input": prompt}):
            if "context" in chunk:
                sources = chunk["context"]
            
            if "answer" in chunk:
                full_answer += chunk["answer"]
                # Display text chunks instantly with a blinking cursor effect
                answer_placeholder.markdown(full_answer + "▌")
        
        # Remove the cursor when finished
        answer_placeholder.markdown(full_answer)
        
        # Display Citations
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