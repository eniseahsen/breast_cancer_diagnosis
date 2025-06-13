import streamlit as st

def set_page_config():
    st.markdown('<h1 style="text-align:center;color:#e85a79;font-weight:bolder;font-size:40px;">Breast Health Diagnostic App</h1>', unsafe_allow_html=True)

def add_background():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.pinimg.com/736x/b2/a6/bb/b2a6bb4de0bb37d74cafff4232f8ad70.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

def style_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #ffe6e6;
            }
        </style>
    """, unsafe_allow_html=True)
