import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.title("Query Pipeline Demo")

# Initialize session state for conversation history
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    st.text(message)

# Query input
query = st.text_input("Enter your query:")

if st.button("Submit"):
    # Prepare the request
    payload = {
        "text": query,
        "conversation_id": st.session_state.conversation_id
    }

    # Send request to API
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.session_state.conversation_id = result['conversation_id']
        
        # Update chat history
        st.session_state.chat_history.append(f"You: {query}")
        st.session_state.chat_history.append(f"AI: {result['answer']}")
        
        # Display the answer
        st.text(f"AI: {result['answer']}")
    else:
        st.error(f"Error: {response.text}")

# Option to clear conversation
if st.button("Clear Conversation"):
    st.session_state.conversation_id = None
    st.session_state.chat_history = []
    st.rerun()