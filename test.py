import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="Variable Mapping Tool", layout="wide")
st.title("📊 ISS Specs Variables Mapping Tool")

# Page switching
# page = st.selectbox("Navigation", ["Upload Datasets", "Mapping & Export", "Variable & Label"])
page = st.sidebar.radio("Navigation", ["Upload Datasets", "Mapping & Export", "Variable & Label"])

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_fields" not in st.session_state:
    st.session_state.dataset_fields = {}
if "study_labels" not in st.session_state:
    st.session_state.study_labels = {}
if "derivation_generated" not in st.session_state:
    st.session_state.derivation_generated = False

if page == "Upload Datasets":
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

        st.session_state.df = df
        st.session_state.dataset_fields = dataset_fields
        st.session_state.study_labels = study_labels
        st.success("✅ Data uploaded and matched. Go to 'Mapping & Export' page to continue.")

elif page == "Mapping & Export":
    if st.session_state.df is None:
        st.warning("Please upload data first.")
        st.stop()

    df = st.session_state.df.copy()
    dataset_fields = st.session_state.dataset_fields

    cell_style = JsCode("""
    function(params) {
        const main = params.data["Main Variable"];
        const val = params.value;
        if (params.colDef.field === "Main Variable" || params.colDef.field === "Main Label") {
            return {backgroundColor: "#dee2e6", fontWeight: "bold"};
        }
        if (params.colDef.field === "Derivation") {
            return {backgroundColor: "#cfe2ff"};
        }
        if (!val) {
            return {backgroundColor: "#f8d7da"};
        }
        if (val === main) {
            return {backgroundColor: "#d4edda"};
        }
        return {backgroundColor: "#fff3cd"};
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True, wrapText=True, autoHeight=True, resizable=True, cellStyle=cell_style)
    gb.configure_column("Main Variable", editable=False)
    gb.configure_column("Main Label", editable=False)
    if "Derivation" in df.columns:
        gb.configure_column("Derivation", editable=False)
    grid_options = gb.build()

    st.subheader("Variable Mapping Table (Editable)")
    grid_return = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        height=600
    )

    st.session_state.df = grid_return["data"]

    if not st.session_state.derivation_generated:
        if st.button("Generate Derivation Column"):
            new_df = st.session_state.df.copy()
            derivation = []
            for _, row in new_df.iterrows():
                conditions = []
                for study in dataset_fields:
                    val = row.get(study, "")
                    if val:
                        conditions.append(f"(studyid = '{study}' then {val})")
                if conditions:
                    derivation.append("if " + " or ".join(conditions))
                else:
                    derivation.append("")
            new_df["Derivation"] = derivation
            st.session_state.df = new_df
            st.session_state.derivation_generated = True
            st.success("✅ Derivation column generated.")
            st.experimental_rerun()

    st.download_button("📥 Export as CSV", data=st.session_state.df.to_csv(index=False).encode("utf-8"), file_name="matched_result.csv")

elif page == "Variable & Label":
    if st.session_state.dataset_fields:
        for study, vars in st.session_state.dataset_fields.items():
            st.markdown(f"### 📘 {study}")
            study_label_map = st.session_state.study_labels[study]
            ref_table = pd.DataFrame({
                "Label": list(study_label_map.keys()),
                "Variable": list(study_label_map.values())
            })
            st.dataframe(ref_table)
    else:
        st.info("Please upload data to view reference variables.")
