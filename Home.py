import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simulation Tool", layout="wide")
st.title("Simulation Tool - Step 1: Load Experience Loss & Exposure Data")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Step 1: Get list of sheets
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Step 2: Let user select a sheet
        sheet_name = st.selectbox("Select a sheet to load:", sheet_names)

        if sheet_name:
            # Step 3: Read selected sheet
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=2)

            # Extract and store data in session state
            loss_col_name = df_raw.columns[0]
            loss_data = df_raw[[loss_col_name]].dropna()

            exposure_data = df_raw.iloc[:, 2:5].dropna(how='all')

            # Save to session state
            st.session_state["loss_data"] = loss_data
            st.session_state["exposure_data"] = exposure_data

            st.success(f"Data loaded from sheet: {sheet_name}")
            st.subheader(f"Loss Data ({loss_col_name})")
            st.dataframe(loss_data)

            st.subheader("Exposure Data (Columns C to E)")
            st.dataframe(exposure_data)

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
