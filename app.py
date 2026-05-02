import streamlit as st
from rufus_brain import get_rufus_response

st.set_page_config(page_title="Rufus Twin | AI Shopping Assistant", page_icon="🛒", layout="centered")

st.title("🛒 Amazon Rufus Twin")
st.caption("A reverse-engineered AI shopping assistant with verifiable source citations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Rufus. Ask me anything about the Magnesium Supplements in my database, and I'll cite my sources!"}
    ]

# Render previous chat messages and their expandable sources
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources"):
                for idx, doc in enumerate(message["sources"]):
                    doc_type = doc.metadata.get('type', 'document').capitalize()
                    asin = doc.metadata.get('source', 'Unknown ASIN')
                    st.caption(f"{idx + 1}. {doc_type} (ASIN: {asin})")

# Handle new user input
if prompt := st.chat_input("Ask about features, reviews, or side effects..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Rufus is analyzing product data..."):
            answer, sources = get_rufus_response(prompt)
            st.markdown(answer)
            
            with st.expander("🔍 View Sources"):
                for idx, doc in enumerate(sources):
                    doc_type = doc.metadata.get('type', 'document').capitalize()
                    asin = doc.metadata.get('source', 'Unknown ASIN')
                    st.caption(f"{idx + 1}. {doc_type} (ASIN: {asin})")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })