import streamlit as st
import pandas as pd

st.title("Interactive Dashboard for data (_CR HDJ 13.xlsx)")

# Path to your default Excel file
default_file = "_CR HDJ 13.xlsx"

# File uploader widget for updating the Excel file
uploaded_file = st.file_uploader("Upload an Excel file to update data (optional)", type=["xlsx", "xls"])

# Load data either from uploaded file or the default file
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Loaded uploaded file successfully!")
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        df = None
else:
    try:
        df = pd.read_excel(default_file)
        st.info(f"Showing data from default file: {default_file}")
    except Exception as e:
        st.error(f"Error reading the default file: {e}")
        df = None

# Show the dataframe if loaded
if df is not None:
    st.write("### Data Preview:")
    st.dataframe(df)

    # Add a button for saving the data
    if st.button("Save Data"):
        try:
            df.to_csv("latest_data.csv", index=False)
            st.success("Data saved successfully to 'latest_data.csv'.")
        except Exception as e:
            st.error(f"Error saving the data: {e}")

else:
    st.info("No data available to display.")