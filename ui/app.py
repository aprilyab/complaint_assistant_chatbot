import streamlit as st
from rag_pipeline import RAGPipeline  # Ensure this imports your Task 3 class

# Initialize only once using session_state
if "rag" not in st.session_state:
    VECTOR_DIR = r"C:\Users\user\Desktop\tasks\complaint_assistant_chatbot\vector_store"  # Adjust path if needed
    st.session_state.rag = RAGPipeline(vector_dir=VECTOR_DIR)

# Streamlit app title
st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="centered")
st.title(" CrediTrust Complaint Chatbot")

# Input area
user_input = st.text_input("Ask a question about customer complaints:")

# Submit and clear buttons
col1, col2 = st.columns([1, 1])
submit = col1.button("Ask")
clear = col2.button("Clear")

if submit and user_input.strip() != "":
    with st.spinner("Retrieving and generating answer..."):
        answer, sources = st.session_state.rag.answer_question(user_input)

        st.markdown("###  Answer")
        st.success(answer)

        st.markdown("### 🔍 Retrieved Sources")
        for i, src in enumerate(sources[:5]):
            st.markdown(f"**Source {i+1}:**")
            st.write(src)

elif clear:
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Created with  for Task 4 - Interactive RAG Chat Interface")

