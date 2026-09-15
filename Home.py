import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulation Tool", layout="wide")
st.title("Simulation Tool - Step 1: Load Experience Loss & Exposure Data")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select a sheet to load:", xls.sheet_names)

        if sheet_name:
            # Preserve the workbook structure used by the existing tool.
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=2)
            columns = df_raw.columns.tolist()

            st.subheader("Select Input Columns")

            loss_col = st.selectbox(
                "Loss Amount Column",
                columns,
                index=0,
            )
            year_col = st.selectbox(
                "Year Column",
                columns,
                index=min(2, len(columns) - 1),
            )
            exposure_col = st.selectbox(
                "Projected Exposure Column",
                columns,
                index=min(3, len(columns) - 1),
            )
            count_col = st.selectbox(
                "Projected Claim Count Column",
                columns,
                index=min(4, len(columns) - 1),
            )

            if st.button("Load Selected Data", type="primary"):
                loss_data = df_raw[[loss_col]].rename(columns={loss_col: "Loss"}).copy()
                loss_data["Loss"] = pd.to_numeric(loss_data["Loss"], errors="coerce")
                loss_data = loss_data.dropna(subset=["Loss"])

                exposure_data = df_raw[[year_col, exposure_col, count_col]].copy()
                exposure_data.columns = [
                    "Year",
                    "Projected exposure",
                    "Projected Claim Count",
                ]

                for col in exposure_data.columns:
                    exposure_data[col] = pd.to_numeric(exposure_data[col], errors="coerce")

                exposure_data = exposure_data.dropna(how="all")

                st.session_state["loss_data"] = loss_data
                st.session_state["exposure_data"] = exposure_data

                st.success(f"Data loaded from sheet: {sheet_name}")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Loss Data")
                    st.dataframe(loss_data, width="stretch")
                with col2:
                    st.subheader("Exposure Data")
                    st.dataframe(exposure_data, width="stretch")

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
