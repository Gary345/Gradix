import streamlit as st

from src.config.settings import APP_SUBTITLE, APP_TITLE


def render_header() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)