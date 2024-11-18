import streamlit as st
from simple_local_arg import retrieve_answers_with_llm_model  # Ensure this module is correctly implemented.

def initialize_session_state():
    """Initialize session state for storing chat messages."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    """Main Streamlit app."""
    st.title("Lynx Genie")
    
    # Initialize session state
    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare chat history with the last two messages
        chat_history = [f"{message['role']}: {message['content']}" for message in st.session_state.messages[-2:]]

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = retrieve_answers_with_llm_model(prompt, chat_history)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
