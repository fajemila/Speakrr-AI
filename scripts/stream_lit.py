# streamlit_app.py
import streamlit as st
import requests
import pymongo
from audio_recorder_streamlit import audio_recorder
from streamlit import session_state
import base64

# Initialize connection to MongoDB database
@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/mydb")


mongo = init_connection()

# Define signup function
def signup():
    st.title("Sign up")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")
    confirm_password = st.text_input("Confirm your password", type="password")
    if st.button("Sign up"):
        if password == confirm_password:
            # Send username and password to flask api
            data = {"username": username, "password": password}
            response = requests.post("http://localhost:5000/signup", json=data)
            if response.status_code == 200:
                st.success("You have successfully signed up!")
            else:
                st.error(response.text)
        else:
            st.error("Passwords do not match")


# Define login function
def login():
    st.title("Log in")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")
    if st.button("Log in"):
        # Send username and password to flask api
        data = {"username": username, "password": password}
        response = requests.post("http://localhost:5000/login", json=data)
        if response.status_code == 200:
            st.success("You have successfully logged in!")
            # Set session state flag to True
            session_state.is_logged_in = True
            # Set session state username to username
            session_state.username = username
        else:
            st.error(response.text)


# Define welcome function
def welcome(username):
    st.title(f"Welcome to this site, {username}!")
    # Find the user document by username
    user = mongo["mydb"]["users"].find_one({"username": username})
    # Check if the user has an empty audios array
    if user["audios"]== []:
        st.title(
            f"Welcome to this site, {username}!. To get started we will be getting some voice samples from you. Please click the button below to start recording your voice."
        )
        # Define five different short texts
        texts = [
            "Hello world",
            "How are you?",
            "What is your name?",
            "Where are you from?",
            "What is your favorite color?",
        ]
        # Loop through the texts and record user voice for each one
        for i, text in enumerate(texts):
            # Display the text
            st.write(f"Text {i+1}: {text}")
            # Record user voice
            audio_bytes = audio_recorder(key=i)
            if audio_bytes:
                # Encode audio_bytes using base64 and decode it to a string
                audio_string = base64.b64encode(audio_bytes).decode()
                # Send user_id, text, and voice to flask api
                data = {"user_id": username, "text": text, "voice": audio_string}
                response = requests.post("http://localhost:5000/voice", json=data)
                if response.status_code == 200:
                    st.success(f"Your voice for text {i+1} has been recorded!")
                else:
                    st.error(response.text)

        # Create a button to go to the commands page
        session_state.command_page = "commands"
    else:
        # Create a button to go to the commands page
        session_state.command_page = "commands"


# Define commands function
def commands():
    st.title("Hello, What would you like to do today?")
    # Record user voice command and send it to flask api
    audio_bytes = audio_recorder(key="command")
    if audio_bytes:
        # Encode audio_bytes using base64 and decode it to a string
        audio_string = base64.b64encode(audio_bytes).decode()
        # Send user_id, text, and voice to flask api
        data = {"user_id": session_state.username, "voice_command": audio_string}
        response = requests.post("http://localhost:5000/get_commands", json=data)
        if response.status_code == 200:
            st.success(response.text)
        else:
            st.error(response.text)



# Define main function
def main():
    # Create a sidebar with signup and login options
    menu = ["Sign up", "Log in"]
    choice = st.sidebar.selectbox("Menu", menu)
    # Initialize session state flag and username
    if "is_logged_in" not in session_state:
        session_state.is_logged_in = False
    if "username" not in session_state:
        session_state.username = ""
    # Check session state flag and display content accordingly
    if session_state.is_logged_in:
        welcome(session_state.username)
        if session_state.command_page=="commands":
            commands()
    else:
        if choice == "Sign up":
            signup()
        elif choice == "Log in":
            login()
    



if __name__ == "__main__":
    main()
