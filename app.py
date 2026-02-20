import streamlit as st
from rag import get_hybrid_chain

st.set_page_config(page_title="CorpBrain RAG")
st.title("CorpBrain: Policy & Docs Assistant")

# Initialize chain only once
if "chain" not in st.session_state:
    with st.spinner("Initializing Hybrid Search Engine..."):
        st.session_state.chain = get_hybrid_chain()
    st.success("System Ready!")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = st.session_state.chain.invoke({"input": prompt})
            answer = response["answer"]
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in response["context"]]))
            
            full_response = f"{answer}\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
            st.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})