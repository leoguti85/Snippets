import streamlit as st
import random
import time
import requests


# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        data = {"query": prompt, "max_length": 100}

        url = "http://bridge:8000/generate"
        response = requests.post(url, json=data)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
