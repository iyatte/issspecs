import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import pandas as pd
from modules.utils import match_single_row, get_cell_style_js

def show():
    if "df" not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state.df.copy()
    dataset_fields = st.session_state.dataset_fields

    # 标记修改过的 label
    if "original_labels" not in st.session_state:
        st.session_state.original_labels = df["Main Label"].copy()

    df["label_changed"] = df["Main Label"] != st.session_state.original_labels

    # 仅对修改的行重新匹配
    for i, changed in enumerate(df["label_changed"]):
        if changed:
            for study in dataset_fields:
                matched_val = match_single_row(df.at[i, "Main Label"], st.session_state.study_labels[study])
                df.at[i, study] = matched_val

    # 表格配置
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True, wrapText=True, autoHeight=True, resizable=True, cellStyle=get_cell_style_js())
    gb.configure_column("Main Variable", editable=False)
    gb.configure_column("label_changed", hide=True)
    gb.configure_column("Main Label", editable=True)
    if "Derivation" in df.columns:
        gb.configure_column("Derivation", editable=False)
    grid_options = gb.build()

    st.subheader("📋 Variable Mapping Table (Editable)")
    grid_return = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        height=600
    )

    st.session_state.df = grid_return["data"]

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
