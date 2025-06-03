import streamlit as st
from modules import upload_page, mapping_page, reference_page

st.set_page_config(page_title="Variable Mapping Tool", layout="wide")
st.title("📊 ISS Specs Variables Mapping Tool")

# Navigation
page = st.sidebar.radio("Navigation", ["Upload Datasets", "Mapping & Export", "Variable & Label"])

# Route
if page == "Upload Datasets":
    upload_page.show()
elif page == "Mapping & Export":
    mapping_page.show()
elif page == "Variable & Label":
    reference_page.show()
