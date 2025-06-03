import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def show():
    core_file = st.file_uploader("Upload Core Specs (must include 'variable' and 'label')", type=["csv"], key="core")
    study_files = st.file_uploader("Upload Study Specs (each must include 'variable' and 'label')", type=["csv"], accept_multiple_files=True, key="studies")
    prev_file = st.file_uploader("Optional: Upload previous result to resume", type=["csv"], key="prev")

    if core_file and study_files:
        core_df = pd.read_csv(core_file)
        if not set(["variable", "label"]).issubset(core_df.columns):
            st.error("Core Specs must contain 'variable' and 'label' columns.")
            st.stop()

        df = pd.DataFrame({
            "Main Variable": core_df["variable"],
            "Main Label": core_df["label"]
        })

        dataset_fields = {}
        study_labels = {}

        for i, file in enumerate(study_files):
            study_name = f"Study{i+1}"
            study_df = pd.read_csv(file)
            if not set(["variable", "label"]).issubset(study_df.columns):
                st.error(f"{study_name} must contain 'variable' and 'label' columns.")
                st.stop()
            dataset_fields[study_name] = study_df["variable"].dropna().tolist()
            study_labels[study_name] = study_df.set_index("label")["variable"].dropna().to_dict()

        def auto_match(main_labels, target_labels, label_to_var):
            vect = TfidfVectorizer().fit(main_labels + target_labels)
            main_vec = vect.transform(main_labels)
            target_vec = vect.transform(target_labels)
            sim = cosine_similarity(main_vec, target_vec)
            results = []
            for i, row in enumerate(sim):
                max_idx = np.argmax(row)
                score = row[max_idx]
                matched_label = target_labels[max_idx]
                matched_var = label_to_var.get(matched_label, "")
                if score > 0.6:
                    results.append(matched_var)
                else:
                    results.append("")
            return results

        for study in dataset_fields:
            df[study] = auto_match(df["Main Label"], list(study_labels[study].keys()), study_labels[study])

        if prev_file:
            prev_df = pd.read_csv(prev_file)
            for col in prev_df.columns:
                if col in df.columns and col not in ["Main Variable", "Main Label"]:
                    df[col] = prev_df[col]
            if "Derivation" in prev_df.columns:
                df["Derivation"] = prev_df["Derivation"]
                st.session_state.derivation_generated = True

        df["label_changed"] = False

        st.session_state.df = df
        st.session_state.dataset_fields = dataset_fields
        st.session_state.study_labels = study_labels
        st.session_state.derivation_generated = "Derivation" in df.columns

        st.success("✅ Data uploaded and matched. Go to 'Mapping & Export' page to continue.")
