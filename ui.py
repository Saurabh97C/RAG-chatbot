
import streamlit as st
from app import get_answer

st.title("⚡ RAG Chatbot (NEC + Wattmonk)")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask something:")

if user_input:
    response = get_answer(user_input)

    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))

for sender, msg in st.session_state.messages:
    if sender == "You":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")
