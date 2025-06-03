import streamlit as st
import pandas as pd

def show():
    if "dataset_fields" not in st.session_state or not st.session_state.dataset_fields:
        st.info("Please upload data to view reference variables.")
        return

    for study, vars in st.session_state.dataset_fields.items():
        st.markdown(f"### 📘 {study}")
        study_label_map = st.session_state.study_labels[study]
        ref_table = pd.DataFrame({
            "Label": list(study_label_map.keys()),
            "Variable": list(study_label_map.values())
        })
        st.dataframe(ref_table)
