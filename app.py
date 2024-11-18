import streamlit as st
from simple_local_arg import retrieve_answers_with_llm_model
from asseteDetails import fetch_asset_details


def initialize_session_state():
    """Initialize session state for storing chat messages."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def extract_asset_id_from_prompt(prompt):
    """Extract the asset ID from the prompt after 'fetch asset details'."""
    if prompt.lower().startswith("fetch asset details"):
        return prompt[len("fetch asset details "):].strip()  # Extract text after the keyword
    return None

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

        # Extract asset ID if the prompt starts with "fetch asset details"
        asset_id = extract_asset_id_from_prompt(prompt)

        if asset_id:
            # Fetch asset details if an asset ID is provided
            with st.spinner("Fetching asset details..."):
                chat_history = [f"{message['role']}: {message['content']}" for message in st.session_state.messages[-2:]]
                asset_details = fetch_asset_details(asset_id)
            with st.chat_message("assistant"):
                st.markdown(asset_details)
            st.session_state.messages.append({"role": "assistant", "content": asset_details})
        else:
            # Handle general conversation flow
            chat_history = [f"{message['role']}: {message['content']}" for message in st.session_state.messages[-2:]]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = retrieve_answers_with_llm_model(prompt, chat_history)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
