import streamlit as st

from chat_tools import memory, qa

# Streamlit app UI
st.title("💬 Chat with your Docs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.memory = memory  # persist memory

user_input = st.chat_input("Ask me something...")

if user_input:
    # Run query
    result = qa({"question": user_input})
    answer = result["answer"]

    # Append to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker.lower()):
        st.markdown(msg)
