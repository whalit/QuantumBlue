"""Supports other functionalities.
"""


import streamlit as st



# Add a block of HTML code to the app
def apply_css():
    """
    Apply CSS styling to the app.
    """
    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)


def set_page(title='Chess', page_icon="♟️"):
    st.set_page_config(
        page_title=title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help':
                'https://github.com/fsmosca/streamlit-chess-board/issues',
            'Report a bug':
                "https://github.com/fsmosca/streamlit-chess-board/issues",
            'About': "# A Streamlit chess board"
        }
    )

    apply_css()
