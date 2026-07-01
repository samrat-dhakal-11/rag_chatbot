"""
auth.py
Very simple login/signup system for the Streamlit app.

Credentials are stored in users.json (in this same folder), passwords are
hashed with SHA-256 (never stored as plain text).

Default account created automatically on first run:
    username: admin
    password: admin123
(You should change/delete this after your first login.)
"""

import streamlit as st
import hashlib
import json
from pathlib import Path

USERS_FILE = Path(__file__).resolve().parent / "users.json"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # first-run default admin account
    default = {"admin": _hash("admin123")}
    with open(USERS_FILE, "w") as f:
        json.dump(default, f)
    return default


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def login_ui() -> bool:
    """
    Renders the login/signup UI.
    Returns True once the user is logged in, False otherwise.
    """
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if st.session_state.logged_in:
        return True

    st.title("🔐 Login — RAG PDF Chatbot")
    st.caption("Default account -> username: admin | password: admin123")

    tab1, tab2 = st.tabs(["Login", "Sign up"])
    users = _load_users()

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if username in users and users[username] == _hash(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create account", key="signup_btn"):
            if not new_user or not new_pass:
                st.error("Please fill both fields.")
            elif new_user in users:
                st.error("Username already exists.")
            else:
                users[new_user] = _hash(new_pass)
                _save_users(users)
                st.success("Account created! Please log in from the Login tab.")

    return False


def logout_button() -> None:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()