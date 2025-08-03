# Core Python Libraries
from datetime import datetime
import io
import re
from io import BytesIO

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Statistical Analysis
from scipy.stats import (
    ttest_ind, ttest_rel, wilcoxon, shapiro,
    mannwhitneyu, chi2_contingency, fisher_exact
)
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LogisticRegression
from scipy.stats import f_oneway

# Machine Learning & Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, davies_bouldin_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Web App
import streamlit as st

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier

# ================================
# 🌐 Streamlit Page Setup
# ================================
st.set_page_config(
    page_title="DIAFOOT Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ================================
# 🎨 Custom CSS for Styling
# ================================
st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background: #e0f2fe; /* Light sky blue background */
        font-family: 'Roboto', sans-serif;
        color: #000000; /* Black text everywhere */
    }

    /* Main Title */
    .main-title {
        color: #000000; /* Black text */
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: #0a3d62; /* deep navy sky blue */
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
        color: #000000; /* Black text */
    }
    
    /* Buttons */
    .stButton>button {
        background: #e0f2fe; /* Light sky blue background */
        color: #000000; /* Black text */
        border-radius: 10px;
        padding: 12px 22px;
        font-weight: 600;
        border: none;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #0a3d62; /* deep navy sky blue */
        transform: scale(1.05);
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] label {
        color: #000000 !important;  /* Black text for label */
        font-weight: 600;
    }

    /* Style for st.info, st.warning, st.error */
    .stAlert {
        border-radius: 10px;
        font-weight: 600;
    }
    /* Dark blue background for info box */
    div[data-testid="stAlert"][kind="info"] {
        background-color: #1e3a8a !important;  /* deep blue */
        color: #ffffff !important;             /* white text */
    }
    /* Dark red background for error box */
    div[data-testid="stAlert"][kind="error"] {
        background-color: #991b1b !important;  /* deep red */
        color: #ffffff !important;             /* white text */
    }
    /* Deep orange background for warning box */
    div[data-testid="stAlert"][kind="warning"] {
        background-color: #d97706 !important;  /* deep orange */
        color: #ffffff !important;             /* white text */
    }
    
    /* Force all text inside st.info to black */
    div[data-testid="stInfo"] * {
        color: #000000 !important;  /* Black text for label */
        font-weight: 600;
    }

    /* Radio Box Container */
    .stRadio > div {
        background-color: #000000; /* Black background */
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .stRadio > div[role='radiogroup'] > label {
        color: #000000 !important; /* Black text */
        font-weight: 600;
    }

/* Table Styling */
    .stTable table {
        background-color: #ffffff; /* White background for table */
        color: #000000; /* Black text */
        border-radius: 10px;
        border-collapse: separate;
        border-spacing: 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .stTable th {
        background-color: #0a3d62; /* Deep navy sky blue for header */
        color: #ffffff; /* White text for header */
        font-weight: 600;
        padding: 12px;
    }
    .stTable td {
        padding: 12px;
        border-bottom: 1px solid #e0e0e0; /* Light gray border */
    }
    .stTable tr:last-child td {
        border-bottom: none; /* Remove border for last row */
    }
    .stTable tr:hover {
        background-color: #e0f2fe; /* Light sky blue on hover */
    }
    /* Selectbox Styling */
    div[data-testid="stSelectbox"] {
        background-color: #000000; /* Black background for selectbox container */
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stSelectbox"] label {
        color: #ffffff !important; /* White text for label */
        font-weight: 600;
    }
    div[data-testid="stSelectbox"] select {
        background-color: #000000; /* Black background for selectbox */
        color: #ffffff; /* White text for selectbox */
        border: 1px solid #e0e0e0; /* Light gray border */
        border-radius: 8px;
        padding: 8px;
        font-weight: 600;
        width: 100%;
    }
    div[data-testid="stSelectbox"] select option {
        background-color: #000000; /* Black background for dropdown options */
        color: #ffffff; /* White text for options */
    }
    div[data-testid="stSelectbox"] select option:hover {
        background-color: #0a3d62; /* Deep navy sky blue on hover for options */
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-title">📊 DIAFOOT Analysis Dashboard</div>', unsafe_allow_html=True)

# ================================
# 📤 File Upload
# ================================
uploaded_file = st.file_uploader("📁 Upload Excel file with 'DIAFOOT' sheet", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="DIAFOOT", header=None)

    # 🔍 Analysis Type Selection
    analysis_type = st.sidebar.radio(
        "🧪 Choose Analysis Type:",
        (
            "Basic Analysis",
            "Descriptive Analysis",
            "Normality Tests",
            "L/R Comparison by Anatomical Zone",
            "Comparison of Left and Right Foot Parameters",           
            "IWGDF Risk Grade Summary & Clustering",
            "Clustering (Important Parameters)",
            "Clustering (All Parameters)",
            "Correlation Between Key Parameters",
            "Bland-Altman Plots by Parameter and Side",
        ),
        index=0,
        help="Choose the type of analysis to perform on the uploaded data."
    )

    # Add more content and analysis sections here
    st.markdown("### Choose your analysis from the sidebar to start exploring the data.")

    # Target Rows for General Analysis
    target_rows = {
        6: "Date of Birth",
        16: "Grade IWGDF",
        17: "Height (m)", 18: "Weight (kg)", 19: "BMI",
        24: "AOMI",
        35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
        37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
        59: "Michigan Score (ok=13, risk=0)",
        72: "Michigan Score2 (ok=13, risk=0)",
        75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
        77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
        94: "Amplitude of dorsiflexion of right MTP1 R",
        95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
        97: "Amplitude talo-crurale L",
        108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R", 110: "Avg Pressure Max TM5 R",
        113: "Avg Pressure Max SESA L", 114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
        118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
        122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
        126: "US Thickness ED SESA R", 127: "US Thickness ED HALLUX R", 128: "US Thickness ED TM5 R",
        130: "US Thickness ED SESA L", 131: "US Thickness ED HALLUX L", 132: "US Thickness ED TM5 L",
        134: "US Thickness Hypodermis SESA R", 135: "US Thickness Hypodermis HALLUX R", 136: "US Thickness Hypodermis TM5 R",
        138: "US Thickness Hypodermis SESA L", 139: "US Thickness Hypodermis HALLUX L", 140: "US Thickness Hypodermis TM5 L",
        142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R", 144: "Total Tissue Thickness TM5 R",
        146: "Total Tissue Thickness SESA L", 147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
        150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
        154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
        158: "Temperature Hallux R", 159: "Temperature 5th Toe R", 160: "Temperature Plantar Arch R",
        161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R", 163: "Temperature Heel R",
        164: "Temperature Hallux L", 165: "Temperature 5th Toe L", 166: "Temperature Plantar Arch L",
        167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L", 169: "Temperature Heel L",
        170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
        172: "Average IR Temperature Foot R (Celsius)", 173: "Average IR Temperature Foot L (Celsius)",
        174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L",
        176: "Normalized Temperature R", 177: "Normalized Temperature L",
        212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L",
        214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L",
    }

    # ================================
    # 🧹 Data Preparation Function
    # ================================
    row_labels = df.iloc[:, 0]
    df_numeric = df.iloc[:, 1:]

    def data_reg(df_num, row_labels):
        target_rows = {
            6: "Date de naissance", 16:"Grade IWGDF", 19: "BMI",
            24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Épaisseur ED SESA R", 127: "US Épaisseur ED HALLUX R", 128: "US Épaisseur ED TM5 R",
            130: "US Épaisseur ED SESA L", 131: "US Épaisseur ED HALLUX L", 132: "US Épaisseur ED TM5 L",
            134: "US Épaisseur Hypoderme SESA R", 135: "US Épaisseur Hypoderme HALLUX R",
            136: "US Épaisseur Hypoderme TM5 R", 138: "US Épaisseur Hypoderme SESA L",
            139: "US Épaisseur Hypoderme HALLUX L", 140: "US Épaisseur Hypoderme TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",            
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R",
            160: "Temperature Plantar Arch R", 161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R",
            163: "Temperature Heel R", 164: "Temperature Hallux L", 165: "Temperature 5th Toe L",
            166: "Temperature Plantar Arch L", 167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L",
            169: "Temperature Heel L", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)",173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R",175: "Temperature Difference Hand-Foot L",176: "Normalized Temperature R",
            177: "Normalized Temperature L", 212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R",
            215: "SUDOSCAN Foot L",
        }
        return df_num, target_rows

    # Call the function with the correct variables
    df_numeric_reg, target_rows_reg = data_reg(df_numeric, row_labels)


    def prepare_data(df):
        # Check if risk grade label is in the expected row
        row_risk = df.iloc[16]
        if str(row_risk[0]).strip().lower() != "grade de risque iwgdf":
            st.error("Label 'Grade de risque IWGDF' not found in row 17.")
            st.stop()

        target_rows = {
            6: "Date of Birth", 16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R", 110: "Avg Pressure Max TM5 R",
            113: "Avg Pressure Max SESA L", 114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Thickness ED SESA R", 127: "US Thickness ED HALLUX R", 128: "US Thickness ED TM5 R",
            130: "US Thickness ED SESA L", 131: "US Thickness ED HALLUX L", 132: "US Thickness ED TM5 L",
            134: "US Thickness Hypodermis SESA R", 135: "US Thickness Hypodermis HALLUX R", 136: "US Thickness Hypodermis TM5 R",
            138: "US Thickness Hypodermis SESA L", 139: "US Thickness Hypodermis HALLUX L", 140: "US Thickness Hypodermis TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R", 144: "Total Tissue Thickness TM5 R",
            146: "Total Tissue Thickness SESA L", 147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R", 154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R", 160: "Temperature Plantar Arch R",
            161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R", 163: "Temperature Heel R",
            164: "Temperature Hallux L", 165: "Temperature 5th Toe L", 166: "Temperature Plantar Arch L",
            167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L", 169: "Temperature Heel L",
            170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)", 173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L",
            176: "Normalized Temperature R", 177: "Normalized Temperature L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L",
        }

        # Extract risk grades for patients from the risk row (drop NA, convert to int)
        risk_values = pd.to_numeric(row_risk[1:], errors='coerce').dropna().astype(int)

        # Select relevant data rows and columns matching risk values length
        selected_data = df.loc[target_rows.keys(), 1:1 + len(risk_values) - 1]
        selected_data.index = [target_rows[i] for i in selected_data.index]

        # Transpose and convert to numeric (each row = one patient)
        df_selected = selected_data.T.apply(pd.to_numeric, errors='coerce')

        # Add Grade and Group columns based on risk values
        df_selected["Grade"] = risk_values.loc[df_selected.index].values
        df_selected["Group"] = df_selected["Grade"].apply(lambda x: "A (Grades 0-1)" if x in [0, 1] else "B (Grades 2-3)")

        # Calculate Age from Date of Birth column if exists (format ddmmyy)
        if "Date of Birth" in df_selected.columns:
            today = pd.Timestamp.today()
            df_selected["Age"] = pd.to_datetime(df_selected["Date of Birth"], format="%d%m%y", errors='coerce').map(
                lambda d: (today - d).days / 365.25 if pd.notnull(d) else np.nan)

        # Extract numeric AOMI from string values (if exists)
        if "AOMI" in df_selected.columns:
            df_selected["AOMI"] = df_selected["AOMI"].astype(str).str.extract(r"(\d+(?:[.,]\d+)?)")[0].str.replace(",", ".").astype(float)

        return df_selected


    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        df_combined = prepare_data(df)

    # ================================
    # 📌 Basic Analysis
    # ================================
    if analysis_type == "Basic Analysis":
        st.header("📋 Basic Analysis")
        target_rows = {
            6: "Date of Birth", 16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R", 110: "Avg Pressure Max TM5 R",
            113: "Avg Pressure Max SESA L", 114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Thickness ED SESA R", 127: "US Thickness ED HALLUX R", 128: "US Thickness ED TM5 R",
            130: "US Thickness ED SESA L", 131: "US Thickness ED HALLUX L", 132: "US Thickness ED TM5 L",
            134: "US Thickness Hypodermis SESA R", 135: "US Thickness Hypodermis HALLUX R", 136: "US Thickness Hypodermis TM5 R",
            138: "US Thickness Hypodermis SESA L", 139: "US Thickness Hypodermis HALLUX L", 140: "US Thickness Hypodermis TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R", 144: "Total Tissue Thickness TM5 R",
            146: "Total Tissue Thickness SESA L", 147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R", 154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R", 160: "Temperature Plantar Arch R",
            161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R", 163: "Temperature Heel R",
            164: "Temperature Hallux L", 165: "Temperature 5th Toe L", 166: "Temperature Plantar Arch L",
            167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L", 169: "Temperature Heel L",
            170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)", 173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L",
            176: "Normalized Temperature R", 177: "Normalized Temperature L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L",
        }
        
        # -------- 🎂 Age Distribution --------
        
        st.subheader("🎂 Age Distribution")
        dob_row = df.iloc[6, 1:].dropna().astype(str).str.strip()

        # Calculate age from today
        dob_dates = pd.to_datetime(dob_row, dayfirst=True, errors='coerce')
        today = pd.Timestamp.today()
        ages = dob_dates.map(lambda d: (today - d).days / 365.25 if pd.notnull(d) else np.nan).dropna()

        if ages.empty:
            st.warning("⚠️ No valid date of birth data found.")
        else:
            average_age = ages.mean()
            std_age = ages.std()   
            st.write(f"**Average Age**: {average_age:.1f} years")
            st.write(f"**Standard Deviation**: {std_age:.1f}")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(ages, bins=10, color='skyblue', edgecolor='black')
            ax.axvline(average_age, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {average_age:.1f}')
            ax.set_title("Patient Age Distribution")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        # -------- 🚻 Gender Distribution --------

        st.subheader("🚻 Gender Distribution")
        gender_row = df[df[0].astype(str).str.strip().str.lower() == "genre h/f/i"]
        if not gender_row.empty:
            idx = gender_row.index[0]
            gender_vals = df.iloc[idx, 1:].astype(str).str.strip().str.upper()
            gender_map = {"H": "Male", "F": "Female", "I": "Unknown"}
            counts = gender_vals.value_counts()
            valid_counts = counts[counts.index.isin(gender_map.keys())]
            valid_counts.index = valid_counts.index.map(gender_map)
            total_patients = valid_counts.sum()
            percentages = (valid_counts / total_patients * 100).round(1)
            fig_gender, ax_gender = plt.subplots()
            ax_gender.pie(
                valid_counts,
                labels=[f"{label} ({count})" for label, count in zip(valid_counts.index, valid_counts)],
                autopct='%1.1f%%',
                startangle=140,
                colors=['lightblue', 'lightpink', 'gray']
            )
            ax_gender.axis('equal')
            st.pyplot(fig_gender)
            gender_percent_text = "\n".join(
                [f"- {label}: {count} patients ({percent}%)" for label, count, percent in zip(valid_counts.index, valid_counts, percentages)]
            )
            st.markdown(f"**Gender Breakdown:**\n{gender_percent_text}")

        else:
            st.warning("⚠️ Gender row ('Genre H/F/I') not found.")


        # -------- ⚖️ BMI Distribution --------
        
        st.subheader("⚖️ BMI Distribution")
        bmi_row = df[df[0].astype(str).str.strip().str.lower() == "bmi (poids[kg] / taille[m]2)"]
        if not bmi_row.empty:
            bmi_vals = pd.to_numeric(df.loc[bmi_row.index[0], 1:], errors='coerce').dropna()
            if not bmi_vals.empty:
                stats = bmi_vals.describe().round(2)
                st.write("**BMI Summary Statistics:**")
                st.dataframe(stats)

                # BMI Histogram
                fig_bmi, ax_bmi = plt.subplots(figsize=(6, 4))
                sns.histplot(bmi_vals, kde=True, ax=ax_bmi, color="mediumseagreen", edgecolor="black", bins=20)
                ax_bmi.set_title("BMI Distribution", fontsize=14)
                ax_bmi.set_xlabel("BMI", fontsize=12)
                ax_bmi.set_ylabel("Frequency", fontsize=12)
                ax_bmi.set_yticks([0, 1, 2])
                ax_bmi.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_bmi)
            else:
                st.warning("⚠️ No valid BMI values.")
        else:
            st.warning("⚠️ BMI label not found.")

        # -------- IWGDF Risk Grade Summary --------

        st.subheader("IWGDF Risk Grade Summary")
        grade_row = df[df[0].astype(str).str.strip().str.lower() == "grade de risque iwgdf"]
        st.markdown("""
        The **IWGDF (International Working Group on the Diabetic Foot)** risk grading system classifies diabetic patients based on their risk of developing foot ulcers. The grades are:

        - **Grade 0 Low risk (no neuropathy)**: No loss of protective sensation (LOPS), no peripheral artery disease (PAD).
        - **Grade 1 Moderate risk (neuropathy only)**: LOPS or PAD without deformity or history of ulcer.
        - **Grade 2 High risk (neuropathy + deformity or PAD)**: LOPS or PAD with foot deformity or prior history of ulcer.
        - **Grade 3 Very high risk (history of ulcer or amputation)**: LOPS or PAD with active ulcer, history of amputation, or both.

        This classification helps target prevention strategies and follow-up frequency.
        """)
        if not grade_row.empty:
            grades = pd.to_numeric(df.loc[grade_row.index[0], 1:], errors='coerce').dropna().astype(int)
            grade_counts = pd.Series([0, 1, 2, 3]).map(lambda g: (grades == g).sum())
            
            risk_labels = ["Low", "Moderate", "High", "Very High"]
            palette = sns.color_palette("Blues", 4)

            fig_grade, ax_grade = plt.subplots(figsize=(7, 4))
            bars = ax_grade.bar(range(4), grade_counts, color=palette, edgecolor='black', width=0.6)

            # Customize axes and background
            ax_grade.set_xticks(range(4))
            ax_grade.set_xticklabels(risk_labels, fontsize=11)
            ax_grade.set_ylabel("Number of Patients", fontsize=12)
            ax_grade.set_title("IWGDF Risk Grade Distribution", fontsize=13, weight='bold')

            # Remove top/right spines for a cleaner look
            ax_grade.spines['top'].set_visible(False)
            ax_grade.spines['right'].set_visible(False)
            ax_grade.grid(axis='y', linestyle='--', alpha=0.5)

            # Add value labels on top of bars
            for i, b in enumerate(bars):
                ax_grade.text(
                    b.get_x() + b.get_width()/2,
                    b.get_height() + 0.3,
                    str(int(grade_counts[i])),
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='semibold'
                )

            st.pyplot(fig_grade)

        else:
            st.warning("⚠️ IWGDF grade row not found.")
    
    
        # -------- Age of Diabetes by Type --------

        st.subheader("🩸 Age of Diabetes by Type")
        st.markdown("""
        This chart shows the **age at diagnosis** (i.e., "age of diabetes") across different **types of diabetes**:
        - 🟦 **Type 1**: Autoimmune, usually early onset
        - 🟨 **Type 2**: Metabolic, typically later onset
        - 🟥 **Atypical (AT)**: Other/rare forms
        """)
        label_age = "Age du diabète (années)"
        label_type = "Type de diabète 1/2/AT"

        # Retrieve rows
        age_row = df[df[0].astype(str).str.strip().str.lower() == label_age.lower()]
        type_row = df[df[0].astype(str).str.strip().str.lower() == label_type.lower()]
        if not age_row.empty and not type_row.empty:
            age_vals = pd.to_numeric(df.loc[age_row.index[0], 1:], errors='coerce')
            type_vals = df.loc[type_row.index[0], 1:].astype(str).str.strip().str.upper()

            # Filter valid types
            valid = (type_vals != "NON DIAB") & type_vals.isin(["1", "2", "AT"])
            df_diabetes = pd.DataFrame({
                "AgeOnset": age_vals[valid],
                "Type": type_vals[valid].map({"1": "Type 1", "2": "Type 2", "AT": "Atypical"})
            }).dropna()

            if not df_diabetes.empty:
                # Boxplot
                fig_age_type, ax_age_type = plt.subplots(figsize=(7, 4))
                palette = {"Type 1": "#4C72B0", "Type 2": "#EAAA00", "Atypical": "#D55E00"}

                sns.boxplot(data=df_diabetes, x="Type", y="AgeOnset", palette=palette, ax=ax_age_type)
                sns.stripplot(data=df_diabetes, x="Type", y="AgeOnset", color='black', size=4, jitter=True, alpha=0.5, ax=ax_age_type)

                ax_age_type.set_title("Distribution of Diabetes Age by Type", fontsize=13, weight='bold')
                ax_age_type.set_ylabel("Age at Diagnosis (years)")
                ax_age_type.set_xlabel("")
                ax_age_type.grid(axis='y', linestyle='--', alpha=0.4)
                st.pyplot(fig_age_type)

                # 📋 Summary table
                st.markdown("#### 📋 Summary Statistics by Type")
                summary = df_diabetes.groupby("Type")["AgeOnset"].agg(['count', 'mean', 'std', 'min', 'max']).round(1)
                st.dataframe(summary.style.format(precision=1))
            else:
                st.warning("⚠️ No valid diabetes type and age combinations found.")
        else:
            st.warning("⚠️ Labels for diabetes type or age not found in your dataset.")

    # ================================
    # 📌 Stat Summary Extractor
    # ================================
    elif analysis_type == "Descriptive Analysis":
        st.header("📊Descriptive Analysis")
        st.markdown("---")
        st.caption("📌 This section provides summary statistics (mean, median, std, min/max, quartiles) and normality tests (Shapiro-Wilk) for key biomechanical and clinical parameters collected from the DIAFOOT dataset.")
        st.markdown("---")
        
        normal_params = []
        non_normal_params = []
        summary_data = []
        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Épaisseur ED SESA R", 127: "US Épaisseur ED HALLUX R", 128: "US Épaisseur ED TM5 R",
            130: "US Épaisseur ED SESA L", 131: "US Épaisseur ED HALLUX L", 132: "US Épaisseur ED TM5 L",
            134: "US Épaisseur Hypoderme SESA R", 135: "US Épaisseur Hypoderme HALLUX R",
            136: "US Épaisseur Hypoderme TM5 R", 138: "US Épaisseur Hypoderme SESA L",
            139: "US Épaisseur Hypoderme HALLUX L", 140: "US Épaisseur Hypoderme TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",            
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R",
            160: "Temperature Plantar Arch R", 161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R",
            163: "Temperature Heel R", 164: "Temperature Hallux L", 165: "Temperature 5th Toe L",
            166: "Temperature Plantar Arch L", 167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L",
            169: "Temperature Heel L", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)",173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R",175: "Temperature Difference Hand-Foot L",176: "Normalized Temperature R",
            177: "Normalized Temperature L", 212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R",
            215: "SUDOSCAN Foot L",
        }
        
        ecart_df = pd.DataFrame()
        for idx, label in target_rows.items():
            raw_values = df.iloc[idx, 1:]
            values = pd.to_numeric(raw_values, errors='coerce').dropna()
            if values.empty:
                continue

            mean = values.mean()
            median = values.median()
            std = values.std()
            min_val = values.min()
            max_val = values.max()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            ecart = values - mean

            # Normality test
            w_stat, p_value = shapiro(values)
            summary_data.append({
                "Label": label,
                "Mean": round(mean, 2),
                "Median": round(median, 2),
                "StdDev": round(std, 2),
                "Min": round(min_val, 2),
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "Max": round(max_val, 2),
                "Shapiro-W": round(w_stat, 4),
                "p-value": round(p_value, 4),
                "Normal": p_value > 0.05
            })

            if p_value > 0.05:
                normal_params.append(label)
            else:
                non_normal_params.append(label)

            with st.expander(f"🔍 {label}"):
                st.write(f"**Mean:** {mean:.2f}")
                st.write(f"**Median:** {median:.2f}")
                st.write(f"**Std Dev:** {std:.2f}")
                st.write(f"**Min:** {min_val:.2f} | **Q1:** {q1:.2f} | **Q3:** {q3:.2f} | **Max:** {max_val:.2f}")
                st.write(f"**Shapiro-Wilk W:** {w_stat:.4f} | **p-value:** {p_value:.4f}")

                # Plot boxplot
                fig, ax = plt.subplots()
                sns.boxplot(x=values, ax=ax, color='lightblue', fliersize=5)
                ax.axvline(mean, color='red', linestyle='--', label=f'Mean ({mean:.2f})')
                ax.axvline(median, color='green', linestyle='-', label=f'Median ({median:.2f})')
                ax.legend()
                st.pyplot(fig)

                st.dataframe(ecart.rename("Deviation from Mean").reset_index(drop=True))

        # Create Excel file for download
        summary_df = pd.DataFrame(summary_data)
        with pd.ExcelWriter("stat_summary.xlsx", engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            ecart_df.to_excel(writer, sheet_name="Deviation", index=False)

        # Convert Excel to downloadable bytes
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            ecart_df.to_excel(writer, sheet_name="Deviation", index=False)
        output.seek(0)

        st.download_button(
            label="📥 Download Excel Report",
            data=output,
            file_name="DIAFOOT_Stat_Summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ================================
    # 📌 Normality Tests
    # ================================
    elif analysis_type == "Normality Tests":
        st.header("📊 Normality Tests for DIAFOOT Parameters")

        st.markdown("""
        **Normality Tests:**  
        - Shapiro-Wilk Test: suitable for small to medium samples  
        - Kolmogorov-Smirnov Test  
        \n
        **Interpretation:**  
        - If p-value > 0.05 → data follows a normal distribution  
        - If p-value ≤ 0.05 → data is non-normal, non-parametric tests are recommended  
        """)

        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Épaisseur ED SESA R", 127: "US Épaisseur ED HALLUX R", 128: "US Épaisseur ED TM5 R",
            130: "US Épaisseur ED SESA L", 131: "US Épaisseur ED HALLUX L", 132: "US Épaisseur ED TM5 L",
            134: "US Épaisseur Hypoderme SESA R", 135: "US Épaisseur Hypoderme HALLUX R",
            136: "US Épaisseur Hypoderme TM5 R", 138: "US Épaisseur Hypoderme SESA L",
            139: "US Épaisseur Hypoderme HALLUX L", 140: "US Épaisseur Hypoderme TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",            
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R",
            160: "Temperature Plantar Arch R", 161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R",
            163: "Temperature Heel R", 164: "Temperature Hallux L", 165: "Temperature 5th Toe L",
            166: "Temperature Plantar Arch L", 167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L",
            169: "Temperature Heel L", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)",173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R",175: "Temperature Difference Hand-Foot L",176: "Normalized Temperature R",
            177: "Normalized Temperature L", 212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R",
            215: "SUDOSCAN Foot L",
        }

        normal_params = []
        non_normal_params = []
        summary_data = []
        for idx, label in target_rows.items():
            raw_values = df.iloc[idx, 1:]
            values = pd.to_numeric(raw_values, errors='coerce').dropna()
            if values.empty:
                continue

            # Shapiro-Wilk Test
            w_stat, p_shapiro = shapiro(values)
            
            # Kolmogorov-Smirnov Test against normal distribution with sample mean and std
            from scipy.stats import kstest, norm
            ks_stat, p_ks = kstest(values, 'norm', args=(values.mean(), values.std()))
            is_normal = (p_shapiro > 0.05) and (p_ks > 0.05)
            summary_data.append({
                "Label": label,
                "Shapiro-W": round(w_stat, 4),
                "p-value Shapiro": round(p_shapiro, 4),
                "KS stat": round(ks_stat, 4),
                "p-value KS": round(p_ks, 4),
                "Normal": is_normal
            })

            if is_normal:
                normal_params.append(label)
            else:
                non_normal_params.append(label)

            with st.expander(f"🔍 {label}"):
                st.write(f"**Shapiro-Wilk:** W = {w_stat:.4f}, p = {p_shapiro:.4f}")
                st.write(f"**Kolmogorov-Smirnov:** stat = {ks_stat:.4f}, p = {p_ks:.4f}")

                # Plot histogram + KDE + mean & median lines
                fig, ax = plt.subplots()
                sns.histplot(values, kde=True, ax=ax, color='skyblue', bins=15)
                ax.axvline(values.mean(), color='red', linestyle='--', label=f'Mean ({values.mean():.2f})')
                ax.axvline(values.median(), color='green', linestyle='-', label=f'Median ({values.median():.2f})')
                ax.legend()
                st.pyplot(fig)

        # Display summary tables
        st.subheader("✅ Normally Distributed Parameters")
        if normal_params:
            st.table(pd.DataFrame(normal_params, columns=["Parameter"]))
        else:
            st.write("None")

        st.subheader("⚠️ Non-Normal Parameters")
        if non_normal_params:
            st.table(pd.DataFrame(non_normal_params, columns=["Parameter"]))
        else:
            st.write("None")

    # ================================
    # 📌 Hallux/SESA/TM5 – Left vs Right Comparison by Parameter Type
    # ================================
    elif analysis_type == "L/R Comparison by Anatomical Zone":
        st.subheader("📊 Hallux/SESA/TM5 – L/R Comparison by Parameter Type")
        st.markdown("---")
        st.caption("This section compares left vs. right values across three anatomical zones (Hallux, SESA, TM5) for multiple parameter types (e.g., ultrasound thickness, pressure, stiffness). Significant differences (p < 0.05) are marked with * (* for p < 0.05, ** for p < 0.01, *** for p < 0.001). Numerical results are shown below each plot.")
        st.markdown("---")
        
        # Epidermis + Dermis thickness values (manually selected rows)
        ep_derm_rows = {
            126: "Epiderm+Derm SESA D", 127: "Epiderm+Derm HALLUX D", 128: "Epiderm+Derm TM5 D",
            130: "Epiderm+Derm SESA G", 131: "Epiderm+Derm HALLUX G", 132: "Epiderm+Derm TM5 G"
        }
        # Hypodermis thickness values
        hypo_rows = {
            134: "Hypoderm SESA D", 135: "Hypoderm HALLUX D", 136: "Hypoderm TM5 D",
            138: "Hypoderm SESA G", 139: "Hypoderm HALLUX G", 140: "Hypoderm TM5 G"
        }
        
        # Barplot function with statistical significance, numerical results, and diagnostics
        def plot_bar_parameter(data_dict, title, ylabel):
            locations = ["HALLUX", "SESA", "TM5"]
            sides = ["D", "G"]
            mapped = {"D": "Right", "G": "Left"}
            plot_data = []
            p_values = {}
            table_data = []
            
            # Collect data, perform t-tests, and prepare table
            for loc in locations:
                right_vals = None
                left_vals = None
                for side in sides:
                    for key, label in data_dict.items():
                        if loc in label and side in label:
                            vals = df.loc[key, 1:].astype(float).dropna()
                            plot_data.append({
                                "Location": loc,
                                "Side": mapped[side],
                                "Mean": vals.mean(),
                                "STD": vals.std()
                            })
                            table_data.append({
                                "Zone": loc,
                                "Side": mapped[side],
                                "Mean": round(vals.mean(), 3) if not np.isnan(vals.mean()) else "NaN",
                                "STD": round(vals.std(), 3) if not np.isnan(vals.std()) else "NaN"
                            })
                            if side == "D":
                                right_vals = vals
                            else:
                                left_vals = vals
                if right_vals is not None and left_vals is not None:
                    if len(right_vals) == len(left_vals) and len(right_vals) > 0 and right_vals.std() > 0 and left_vals.std() > 0:
                        try:
                            t_stat, p_val = ttest_rel(right_vals, left_vals)
                            p_values[loc] = p_val if not np.isnan(p_val) else 1.0
                            table_data.append({
                                "Zone": loc,
                                "Side": "p-value",
                                "Mean": round(p_val, 4) if not np.isnan(p_val) else "NaN",
                                "STD": ""
                            })
                        except Exception as e:
                            st.warning(f"T-test failed for {loc}: {str(e)}")
                            p_values[loc] = 1.0
                            table_data.append({
                                "Zone": loc,
                                "Side": "p-value",
                                "Mean": "NaN",
                                "STD": "T-test failed"
                            })
                    else:
                        st.warning(f"T-test skipped for {loc}: " +
                                (f"Empty or mismatched data (Right: {len(right_vals)}, Left: {len(left_vals)})" if len(right_vals) != len(left_vals) or len(right_vals) == 0 else
                                    f"Zero variance in data for {loc}"))
                        p_values[loc] = 1.0
                        table_data.append({
                            "Zone": loc,
                            "Side": "p-value",
                            "Mean": "NaN",
                            "STD": "Invalid data"
                        })
                else:
                    p_values[loc] = 1.0
                    table_data.append({
                        "Zone": loc,
                        "Side": "p-value",
                        "Mean": "NaN",
                        "STD": "No data"
                    })
            
            # Create bar plot
            plot_df = pd.DataFrame(plot_data)
            if plot_df.empty:
                st.warning(f"No valid data for {title}")
                return
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x="Location", y="Mean", hue="Side", ax=ax,
                        palette="coolwarm", capsize=0.1)
            
            # Add significance annotations
            for i, loc in enumerate(locations):
                if loc in p_values and not np.isnan(p_values[loc]) and p_values[loc] < 0.05:
                    p_val = p_values[loc]
                    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    max_y = plot_df[plot_df["Location"] == loc]["Mean"].max()
                    if not np.isnan(max_y):
                        ax.text(i, max_y + 0.05 * max_y, stars, ha='center', va='bottom', fontsize=12)
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            st.pyplot(fig)
            
            # Display numerical results
            st.subheader(f"Results for {title}")
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df)
        
        # Plot thickness by skin layer
        plot_bar_parameter(ep_derm_rows, "Epidermis + Dermis Thickness – L/R", "Thickness (mm)")
        plot_bar_parameter(hypo_rows, "Hypodermis Thickness – L/R", "Thickness (mm)")
        
        # Define zones and side labels for more general use
        zones = ["HALLUX", "SESA", "TM5"]
        sides = {"R": "Right", "L": "Left"}
        
        # Group parameters by type
        def group_parameters_by_type(target_rows, types_keywords):
            grouped = {k: {} for k in types_keywords}
            for row_idx, label in target_rows.items():
                for zone in zones:
                    if zone in label:
                        for code, side in sides.items():
                            if f"{zone} {code}" in label or f"{code} {zone}" in label:
                                for k in types_keywords:
                                    if k.lower() in label.lower():
                                        grouped[k][row_idx] = f"{k} {zone} {side}"
            return grouped
        
        # Categories to compare L/R in barplots
        parameter_types = [
            "Avg Pressure", "Stiffness",
            "US Épaisseur ED", "US Épaisseur Hypoderme",
            "Total Tissue Thickness", "ROC"
        ]
        
        # Group selected target parameters into categories
        grouped_params = group_parameters_by_type(target_rows, parameter_types)
        
        # L/R bar plot per parameter type with statistical significance and numerical results
        def plot_combined_bar(data_dict, title, ylabel):
            plot_data = []
            p_values = {}
            table_data = []
            
            # Collect data, perform t-tests, and prepare table
            for zone in zones:
                right_vals = None
                left_vals = None
                for key, label in data_dict.items():
                    if zone in label:
                        for side_code, side_label in sides.items():
                            if side_label in label:
                                try:
                                    vals = df.loc[key, 1:].astype(float).dropna()
                                    plot_data.append({
                                        "Zone": zone,
                                        "Side": side_label,
                                        "Mean": vals.mean(),
                                        "STD": vals.std()
                                    })
                                    table_data.append({
                                        "Zone": zone,
                                        "Side": side_label,
                                        "Mean": round(vals.mean(), 3) if not np.isnan(vals.mean()) else "NaN",
                                        "STD": round(vals.std(), 3) if not np.isnan(vals.std()) else "NaN"
                                    })
                                    if side_label == "Right":
                                        right_vals = vals
                                    else:
                                        left_vals = vals
                                except:
                                    pass
                if right_vals is not None and left_vals is not None:
                    if len(right_vals) == len(left_vals) and len(right_vals) > 0 and right_vals.std() > 0 and left_vals.std() > 0:
                        try:
                            t_stat, p_val = ttest_rel(right_vals, left_vals)
                            p_values[zone] = p_val if not np.isnan(p_val) else 1.0
                            table_data.append({
                                "Zone": zone,
                                "Side": "p-value",
                                "Mean": round(p_val, 4) if not np.isnan(p_val) else "NaN",
                                "STD": ""
                            })
                        except Exception as e:
                            st.warning(f"T-test failed for {zone}: {str(e)}")
                            p_values[zone] = 1.0
                            table_data.append({
                                "Zone": zone,
                                "Side": "p-value",
                                "Mean": "NaN",
                                "STD": "T-test failed"
                            })
                    else:
                        st.warning(f"T-test skipped for {zone}: " +
                                (f"Empty or mismatched data (Right: {len(right_vals)}, Left: {len(left_vals)})" if len(right_vals) != len(left_vals) or len(right_vals) == 0 else
                                    f"Zero variance in data for {zone}"))
                        p_values[zone] = 1.0
                        table_data.append({
                            "Zone": zone,
                            "Side": "p-value",
                            "Mean": "NaN",
                            "STD": "Invalid data"
                        })
                else:
                    st.warning(f"No data available for {zone} (Right: {right_vals is not None}, Left: {left_vals is not None})")
                    p_values[zone] = 1.0
                    table_data.append({
                        "Zone": zone,
                        "Side": "p-value",
                        "Mean": "NaN",
                        "STD": "No data"
                    })
            
            plot_df = pd.DataFrame(plot_data)
            if plot_df.empty:
                st.warning(f"No valid data found for {title}")
                return
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x="Zone", y="Mean", hue="Side", ax=ax,
                        palette="coolwarm", capsize=0.1, errwidth=1)
            
            # Add significance annotations
            for i, zone in enumerate(zones):
                if zone in p_values and not np.isnan(p_values[zone]) and p_values[zone] < 0.05:
                    p_val = p_values[zone]
                    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    max_y = plot_df[plot_df["Zone"] == zone]["Mean"].max()
                    if not np.isnan(max_y):
                        ax.text(i, max_y + 0.05 * max_y, stars, ha='center', va='bottom', fontsize=12)
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            st.pyplot(fig)
            
            # Display numerical results
            st.subheader(f"Results for {title}")
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df)
        
        # Loop through each parameter group and plot it
        for ptype, row_dict in grouped_params.items():
            plot_combined_bar(row_dict, f"{ptype} – Left vs Right Comparison", "Mean Value")
        
    # ================================
    # 📌 Comparison of Left and Right Foot Parameters
    # ================================
    elif analysis_type == "Comparison of Left and Right Foot Parameters":
        st.header("Comparison of Left and Right Foot Parameters with Plots")
        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Épaisseur ED SESA R", 127: "US Épaisseur ED HALLUX R", 128: "US Épaisseur ED TM5 R",
            130: "US Épaisseur ED SESA L", 131: "US Épaisseur ED HALLUX L", 132: "US Épaisseur ED TM5 L",
            134: "US Épaisseur Hypoderme SESA R", 135: "US Épaisseur Hypoderme HALLUX R",
            136: "US Épaisseur Hypoderme TM5 R", 138: "US Épaisseur Hypoderme SESA L",
            139: "US Épaisseur Hypoderme HALLUX L", 140: "US Épaisseur Hypoderme TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",            
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R",
            160: "Temperature Plantar Arch R", 161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R",
            163: "Temperature Heel R", 164: "Temperature Hallux L", 165: "Temperature 5th Toe L",
            166: "Temperature Plantar Arch L", 167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L",
            169: "Temperature Heel L", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)",173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R",175: "Temperature Difference Hand-Foot L",176: "Normalized Temperature R",
            177: "Normalized Temperature L", 212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R",
            215: "SUDOSCAN Foot L",
        }

        # Generate paired parameters list
        paired_parameters = []
        for idx_r, label_r in target_rows.items():
            if label_r.endswith(" R"):
                label_l = label_r[:-2] + " L"
                idx_l = next((i for i, lbl in target_rows.items() if lbl == label_l), None)
                if idx_l is not None:
                    paired_parameters.append((label_r, label_l, idx_r, idx_l))

        st.subheader("🔬 Paired Tests + Boxplots")
        
        st.markdown("---")
        st.caption("🔍 This section compares left and right foot values for all matched biomechanical and clinical parameters.")
        st.caption("- It automatically detects parameter pairs (e.g., 'Stiffness TM5 R' vs. 'Stiffness TM5 L') based on zone and side.")
        st.caption("- Normality is tested using the Shapiro-Wilk test, followed by a paired t-test or Wilcoxon test depending on distribution.")
        st.caption("- Side-by-side boxplots visualize the distributions of left and right values for each zone and parameter.")
        st.caption("- All test results — including means, medians, p-values, and test type — can be exported in Excel format.")
        st.markdown("---")

        comparison_results = []
        for label_r, label_l, idx_r, idx_l in paired_parameters:
            values_r = pd.to_numeric(df.iloc[idx_r, 1:], errors='coerce').dropna()
            values_l = pd.to_numeric(df.iloc[idx_l, 1:], errors='coerce').dropna()
            common_len = min(len(values_r), len(values_l))
            values_r = values_r.iloc[:common_len]
            values_l = values_l.iloc[:common_len]

            # Check for minimum data length
            if common_len < 2:
                st.warning(f"Not enough data to test: {label_r} vs {label_l} (n={common_len})")
                continue

            # Normality Bland-Altman Pooled Plots
            p_r = shapiro(values_r)[1]
            p_l = shapiro(values_l)[1]
            if p_r > 0.05 and p_l > 0.05:
                stat, p_val = ttest_rel(values_r, values_l)
                test_name = "Paired t-test"
            else:
                stat, p_val = wilcoxon(values_r, values_l)
                test_name = "Wilcoxon signed-rank test"

            # Store results
            comparison_results.append({
                "Parameter Right": label_r,
                "Parameter Left": label_l,
                "Test Used": test_name,
                "Statistic": stat,
                "p-value": p_val,
                "Mean Right": values_r.mean(),
                "Mean Left": values_l.mean(),
                "Median Right": values_r.median(),
                "Median Left": values_l.median(),
                "N": common_len
            })
            st.markdown(f"### {label_r} vs {label_l}")
            st.write(f"**Test used:** {test_name}")
            st.write(f"**Statistic:** {stat:.4f}", f"**p-value:** {p_val:.4f}")
            data_plot = pd.DataFrame({
                'Right': values_r.reset_index(drop=True),
                'Left': values_l.reset_index(drop=True)
            })
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=data_plot, ax=ax, palette=["#90CAF9", "#F48FB1"])
            
            for i, col in enumerate(data_plot.columns):
                mean_val = data_plot[col].mean()
                median_val = data_plot[col].median()
                ax.hlines(mean_val, i - 0.25, i + 0.25, colors='red', label='Mean' if i == 0 else "", linewidth=2)
                ax.hlines(median_val, i - 0.25, i + 0.25, colors='blue', label='Median' if i == 0 else "", linewidth=2, linestyles='--')

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.set_title(f"Distribution: {label_r} vs {label_l}")
            st.pyplot(fig)
            st.markdown("---")

        # Create Excel download button for results
        comparison_df = pd.DataFrame(comparison_results)
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            comparison_df.to_excel(writer, index=False, sheet_name="Paired Comparison")

        st.download_button(
            label="📥 Download Comparison Results as Excel",
            data=excel_buffer.getvalue(),
            file_name="left_right_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    # ================================
    # 📌 IWGDF Risk Grade Summary & KMeans Clustering
    # ================================
    elif analysis_type == "IWGDF Risk Grade Summary & Clustering":     
        st.header("📌 IWGDF Risk Grade Summary & KMeans Clustering")  
        target_rows = {
            6: "Date of Birth", 16: "Grade IWGDF", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R", 110: "Avg Pressure Max TM5 R",
            113: "Avg Pressure Max SESA L", 114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Thickness ED SESA R", 127: "US Thickness ED HALLUX R", 128: "US Thickness ED TM5 R",
            130: "US Thickness ED SESA L", 131: "US Thickness ED HALLUX L", 132: "US Thickness ED TM5 L",
            134: "US Thickness Hypodermis SESA R", 135: "US Thickness Hypodermis HALLUX R", 136: "US Thickness Hypodermis TM5 R",
            138: "US Thickness Hypodermis SESA L", 139: "US Thickness Hypodermis HALLUX L", 140: "US Thickness Hypodermis TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R", 144: "Total Tissue Thickness TM5 R",
            146: "Total Tissue Thickness SESA L", 147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R", 154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R", 160: "Temperature Plantar Arch R",
            161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R", 163: "Temperature Heel R",
            164: "Temperature Hallux L", 165: "Temperature 5th Toe L", 166: "Temperature Plantar Arch L",
            167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L", 169: "Temperature Heel L",
            170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)", 173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L",
            176: "Normalized Temperature R", 177: "Normalized Temperature L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L",
        }

        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found in the Excel sheet.")
            st.stop()

        # Extract the actual values from the row found above
        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        patient_ids = risk_values.index

        selected_data = df.loc[target_rows.keys(), 1:1 + len(risk_values) - 1]
        selected_data.index = [target_rows[i] for i in selected_data.index]  # Convert index to labels
        features_df = selected_data.T.apply(pd.to_numeric, errors='coerce')  # Transpose and convert to numeric
        features_df = features_df.loc[patient_ids]  # Align rows to patient order

        # Handle missing data
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(features_df.values)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_imputed)

        #  Build results DataFrame with grade and cluster for each patient
        results = pd.DataFrame({
            "Patient": patient_ids,
            "IWGDF_Grade": risk_values.loc[patient_ids].values,
            "Cluster": clusters
        }, index=patient_ids)

        contingency = pd.crosstab(results["Cluster"], results["IWGDF_Grade"])

        # Plot clustering colored by group
        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(results["Patient"], results["IWGDF_Grade"], cmap="jet", edgecolor='k')
        ax.set_xlabel("Patient ID")
        ax.set_ylabel("IWGDF Grade")
        ax.set_title("Clustering Colored by Cluster")
        # Set x-axis ticks from 1 to 20
        ax.set_xticks(np.arange(1, 21, 1))
        # Set y-axis ticks at 0, 1, 2, 3
        ax.set_yticks(np.arange(0, 4, 1)) 

        st.pyplot(fig)

        st.markdown("ℹ️Each dot represents a patient")

        param_cols = [c for c in df_combined.columns if c not in ["Grade", "Group", "Cluster", "PCA1", "PCA2"]]

        grade_pairs = [(0, 1), (1, 2), (2, 3)]

        for g in [0, 1, 2, 3]:
            count = (df_combined["Grade"] == g).sum()
            st.write(f"Number of samples in Grade {g}: {count}")

        results = []
        min_samples = 2  

        for param in param_cols:
            for g1, g2 in grade_pairs:
                group1 = df_combined[df_combined["Grade"] == g1][param].dropna()
                group2 = df_combined[df_combined["Grade"] == g2][param].dropna()

                if len(group1) >= min_samples and len(group2) >= min_samples:
                    # Normality check
                    try:
                        _, p1 = shapiro(group1)
                        _, p2 = shapiro(group2)
                        normal = (p1 > 0.05) and (p2 > 0.05)
                    except Exception as e:
                        normal = False

                    if normal:
                        stat, pval = ttest_ind(group1, group2)
                        test_name = "t-test"
                    else:
                        stat, pval = mannwhitneyu(group1, group2)
                        test_name = "Mann-Whitney U"

                    results.append({
                        "Parameter": param,
                        "Grade 1": g1,
                        "Grade 2": g2,
                        "Test": test_name,
                        "p-value": pval,
                        "Significant": pval < 0.05,
                        "Mean Grade 1": group1.mean(),
                        "Mean Grade 2": group2.mean()
                    })

        if len(results) == 0:
            st.warning(f"No statistical tests were performed because no groups had at least {min_samples} samples each.")
        else:
            df_results = pd.DataFrame(results)


        st.header("Statistical Tests Between Combined Grades")

        param_cols = [c for c in df_combined.columns if c not in ["Grade", "Group", "Cluster", "PCA1", "PCA2"]]

        df_combined['Grade_combined'] = df_combined['Grade'].apply(lambda x: 'A (0-1)' if x in [0, 1] else ('B (2-3)' if x in [2, 3] else np.nan))

        for group_label in ['A (0-1)', 'B (2-3)']:
            count = (df_combined["Grade_combined"] == group_label).sum()
            st.write(f"Number of samples in Group {group_label}: {count}")

        results = []
        min_samples = 2  

        for param in param_cols:
            groupA = df_combined[df_combined["Grade_combined"] == 'A (0-1)'][param].dropna()
            groupB = df_combined[df_combined["Grade_combined"] == 'B (2-3)'][param].dropna()

            if len(groupA) >= min_samples and len(groupB) >= min_samples:
                # Normality check
                try:
                    _, p1 = shapiro(groupA)
                    _, p2 = shapiro(groupB)
                    normal = (p1 > 0.05) and (p2 > 0.05)
                except Exception:
                    normal = False

                if normal:
                    stat, pval = ttest_ind(groupA, groupB)
                    test_name = "t-test"
                else:
                    stat, pval = mannwhitneyu(groupA, groupB)
                    test_name = "Mann-Whitney U"

                results.append({
                    "Parameter": param,
                    "Group 1": "A (0-1)",
                    "Group 2": "B (2-3)",
                    "Test": test_name,
                    "p-value": pval,
                    "Significant": pval < 0.05,
                    "Mean Group 1": groupA.mean(),
                    "Mean Group 2": groupB.mean()
                })

        if len(results) == 0:
            st.warning(f"No statistical tests were performed because no groups had at least {min_samples} samples each.")
        else:
            df_results = pd.DataFrame(results)
            st.write("Statistical test results sorted by p-value:")
            st.dataframe(df_results.sort_values("p-value"))
            
        def compare_groups_safely(name, group1_idx, group2_idx, features_df, min_samples=2, alpha=0.1, effect_threshold=0.05):
            st.markdown(f"#### {name}")
            st.write(f"Group sizes: {len(group1_idx)} vs {len(group2_idx)}")

                    
            # 📉 Not enough patients → Show only group means
            if len(group1_idx) < min_samples or len(group2_idx) < min_samples:
                st.warning("⚠️ Not enough patients to run significance tests, but means are still shown.")
                rows = []
                for feat in features_df.columns:
                    m1 = features_df.loc[group1_idx, feat].mean()
                    m2 = features_df.loc[group2_idx, feat].mean()
                    delta = m2 - m1
                    rows.append((feat, m1, m2, delta))
                df = pd.DataFrame(rows, columns=["Feature", "Mean Group1", "Mean Group2", "Δ Mean"])
                df = df.sort_values("\u0394 Mean", key=abs, ascending=False)
                st.dataframe(df)
                return df

            # ✅ Enough data → Run Mann-Whitney U test + effect size
            else:
                sig_feats = []
                for feat in features_df.columns:
                    vals1 = features_df.loc[group1_idx, feat].dropna()
                    vals2 = features_df.loc[group2_idx, feat].dropna()
                    if len(vals1) < min_samples or len(vals2) < min_samples:
                        continue
                    stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
                    m1 = vals1.mean()
                    m2 = vals2.mean()
                    diff = m2 - m1
                    full_range = features_df[feat].max() - features_df[feat].min()
                    effect = abs(diff) / full_range if full_range else 0
                    if p < alpha and effect > effect_threshold:
                        sig_feats.append((feat, m1, m2, diff, p, effect))

                if not sig_feats:
                    st.info(f"No significant features found (p < {alpha} and effect > {effect_threshold})")
                    return pd.DataFrame()
                else:
                    df = pd.DataFrame(sig_feats, columns=["Feature", "Mean G1", "Mean G2", "Δ Mean", "p-value", "Effect Size"])
                    df = df.sort_values("Effect Size", ascending=False)
                    st.dataframe(df)

                    return df

    # ================================
    # Clustering with More Important Parameters
    # ================================
    elif analysis_type == "Clustering (Important Parameters)":
        st.header("Clustering with More Important Parameters")
        important_params = {
            16: "Grade IWGDF",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score", 72: "Michigan Score2",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Thickness ED SESA R", 127: "US Thickness ED HALLUX R", 128: "US Thickness ED TM5 R",
            130: "US Thickness ED SESA L", 131: "US Thickness ED HALLUX L", 132: "US Thickness ED TM5 L",
            134: "US Thickness Hypodermis SESA R", 135: "US Thickness Hypodermis HALLUX R",
            136: "US Thickness Hypodermis TM5 R", 138: "US Thickness Hypodermis SESA L",
            139: "US Thickness Hypodermis HALLUX L", 140: "US Thickness Hypodermis TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude of talo-crural joint R",
            97: "Amplitude of talo-crural joint L",
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R",
            160: "Temperature Plantar Arch R", 161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R",
            163: "Temperature Heel R", 164: "Temperature Hallux L", 165: "Temperature 5th Toe L",
            166: "Temperature Plantar Arch L", 167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L",
            169: "Temperature Heel L", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)", 173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L",
            176: "Normalized Temperature R", 177: "Normalized Temperature L",
        }

        # Extract IWGDF grades and patient file numbers
        risk_row = df.iloc[16]
        risk_values = pd.to_numeric(risk_row[1:], errors='coerce').dropna().astype(int)
        patient_ids = risk_values.index.tolist()

        # Select number of clusters
        k = st.slider("Number of clusters", 2, 6, 4)

        # Define clustering algorithms
        clustering_algos = {
            "Agglomerative (ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "Agglomerative (average)": AgglomerativeClustering(n_clusters=k, linkage="average"),
            "KMeans": KMeans(n_clusters=k, random_state=42),
            "Gaussian Mixture Model": GaussianMixture(n_components=k, random_state=42),
        }

        # Color map for clusters
        color_map = {0: "green", 1: "blue", 2: "orange", 3: "red"}

        # Marker map for IWGDF grades
        marker_map = {0: 'o', 1: 's', 2: '^', 3: '*'}

        def clean_column_name(name):
            # Remove invalid characters and replace spaces with underscores
            name = re.sub(r'[^\w\s]', '', name).strip().replace(' ', '_')
            # Ensure name starts with a letter
            if not name[0].isalpha():
                name = 'p_' + name
            return name

        def prepare_data(include_grade=True):
            params = important_params.copy()
            if not include_grade:
                params.pop(16, None)
            
            selected_features = df.loc[list(params.keys())]
            # Clean index names
            cleaned_index = {k: clean_column_name(v) for k, v in params.items()}
            selected_features.index = [cleaned_index[k] for k in selected_features.index]
            selected_features = selected_features.loc[:, patient_ids]
            features_df = selected_features.T.apply(pd.to_numeric, errors='coerce')
            features_df['Patient_File_Number'] = features_df.index
            true_labels = risk_values.reindex(features_df.index)
            
            # Clean data
            valid_features = features_df.isnull().sum() < features_df.shape[0] * 0.5
            full_df_cleaned = features_df.loc[:, valid_features]
            full_df_cleaned = full_df_cleaned.loc[:, full_df_cleaned.count() >= 20]
            full_df_cleaned = full_df_cleaned.apply(lambda col: col.fillna(col.median()) if col.name != 'Patient_File_Number' else col, axis=0)
            
            if full_df_cleaned.shape[1] <= 1 or full_df_cleaned.shape[0] < 2:
                return None, None, None, None
            
            # Outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(full_df_cleaned.drop(columns=['Patient_File_Number']))
            non_outliers = outlier_labels == 1
            full_df_cleaned = full_df_cleaned[non_outliers]
            true_labels = true_labels[non_outliers]
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(full_df_cleaned.drop(columns=['Patient_File_Number']))
            
            return X_scaled, full_df_cleaned, true_labels, scaler

        # Prepare data for both cases
        X_scaled_with, full_df_cleaned_with, true_labels_with, scaler_with = prepare_data(include_grade=True)
        X_scaled_without, full_df_cleaned_without, true_labels_without, scaler_without = prepare_data(include_grade=False)

        if X_scaled_with is None or X_scaled_without is None:
            st.warning("Not enough valid data for clustering with important parameters after outlier removal.")
            st.stop()

        # Align indices
        common_indices = full_df_cleaned_with.index.intersection(full_df_cleaned_without.index)
        if len(common_indices) < 2:
            st.warning("Not enough overlapping patients for clustering after outlier removal.")
            st.stop()

        X_scaled_with = X_scaled_with[full_df_cleaned_with.index.isin(common_indices)]
        X_scaled_without = X_scaled_without[full_df_cleaned_without.index.isin(common_indices)]
        true_labels = true_labels_with.loc[common_indices]
        full_df_cleaned_with = full_df_cleaned_with.loc[common_indices]
        full_df_cleaned_without = full_df_cleaned_without.loc[common_indices]

        # Perform PCA for visualization
        pca_with = PCA(n_components=2)
        coords_with = pca_with.fit_transform(X_scaled_with)
        pca_without = PCA(n_components=2)
        coords_without = pca_without.fit_transform(X_scaled_without)

        # Streamlit header and parameter display
        st.markdown("""
        The important parameters are:
        - IWGDF Grade
        - MESI
        - Michigan Score
        - Pressure
        - Stiffness
        - Thickness
        - Amplitude
        - Temperature
        """)
        
        st.markdown("---")
        st.caption("Agglomerative (ward): Hierarchically merges clusters by minimizing variance within clusters.")
        st.caption("Agglomerative (average): Hierarchically merges clusters using average distance between points.")
        st.caption("KMeans: Partitions data into k clusters by minimizing distance to cluster centroids.")
        st.caption("Gaussian Mixture Model: Clusters data using probabilistic Gaussian distributions.")
        st.markdown("---")
        
        metrics_with = {"Algorithm": [], "Silhouette": [], "Calinski-Harabasz": [], "Davies-Bouldin": [], "ARI": [], "NMI": []}
        metrics_without = {"Algorithm": [], "Silhouette": [], "Calinski-Harabasz": [], "Davies-Bouldin": [], "ARI": [], "NMI": []}
        feature_importance_dict_with = {}
        feature_importance_dict_without = {}
        confusion_matrices = {}
        cluster_stats_dict = {}

        # Create subplots (2 columns: with grade, without grade)
        n_algos = len(clustering_algos)
        fig, axes = plt.subplots(n_algos, 2, figsize=(12, 5 * n_algos))
        axes = axes.reshape(n_algos, 2) if n_algos > 1 else np.array([axes])

        # Function to calculate within-group inertia
        def calculate_inertia(X, labels):
            inertia = 0
            for cluster in np.unique(labels):
                cluster_points = X[labels == cluster]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            return inertia

        # Function to run clustering and plot
        def run_clustering(X_scaled, full_df_cleaned, true_labels, coords, title_prefix, axes, col_idx, feature_importance_dict):
            for idx, (name, algo) in enumerate(clustering_algos.items()):
                if name.startswith("Agglomerative (ward)") and X_scaled.shape[1] < 2:
                    st.warning(f"Ward linkage requires at least 2 features for {title_prefix}. Skipping.")
                    continue
                
                # Perform clustering
                labels = algo.fit_predict(X_scaled)
                silhouette = silhouette_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)
                ari = adjusted_rand_score(true_labels, labels)
                nmi = normalized_mutual_info_score(true_labels, labels)
                
                metrics = metrics_with if "With Grade" in title_prefix else metrics_without
                metrics["Algorithm"].append(name)
                metrics["Silhouette"].append(silhouette)
                metrics["Calinski-Harabasz"].append(ch)
                metrics["Davies-Bouldin"].append(db)
                metrics["ARI"].append(ari)
                metrics["NMI"].append(nmi)
                
                # Calculate confusion matrix (all IWGDF grades)
                conf_matrix = confusion_matrix(true_labels, labels)
                confusion_matrices[f"{title_prefix}_{name}"] = conf_matrix
                
                # Calculate feature importance
                rf = RandomForestClassifier(random_state=42)
                rf.fit(X_scaled, labels)
                feature_importance_dict[name] = pd.Series(rf.feature_importances_, index=full_df_cleaned.drop(columns=['Patient_File_Number']).columns)
                
                # Calculate mean and median per cluster
                cluster_stats = full_df_cleaned.copy()
                cluster_stats['Cluster'] = labels
                mean_stats = cluster_stats.groupby('Cluster').mean()
                median_stats = cluster_stats.groupby('Cluster').median()
                
                # Plot with cluster colors, grade markers, and patient file numbers
                ax = axes[idx, col_idx]
                for grade in sorted(true_labels.unique()):
                    mask = true_labels == grade
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=[color_map.get(lbl, "black") for lbl in labels[mask]], 
                        marker=marker_map[grade], s=80, label=f"Grade {grade}"
                    )
                    # Add patient file numbers as annotations
                    for i, patient_id in enumerate(full_df_cleaned['Patient_File_Number'][mask]):
                        ax.annotate(
                            patient_id, 
                            (coords[mask, 0][i] + 0.02, coords[mask, 1][i] + 0.02),  # Slight offset
                            fontsize=10, 
                            fontweight='bold', 
                            alpha=0.9,
                        )
                
                ax.set_title(f"{name} ({title_prefix})\nSilhouette={silhouette:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}", fontsize=14, pad=20)
                ax.set_xlabel("PCA1", fontsize=10)
                ax.set_ylabel("PCA2", fontsize=10)
                
                # Create legends
                grade_handles = [plt.Line2D([0], [0], marker=marker_map[grade], color='w', label=f"Grade {grade}", 
                                            markerfacecolor='gray', markersize=10) 
                                for grade in sorted(true_labels.unique())]
                cluster_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Cluster {i}", 
                                            markerfacecolor=color_map.get(i, "black"), markersize=10) 
                                for i in range(k)]
                ax.legend(handles=grade_handles + cluster_handles, title="IWGDF Grades & Clusters", loc="upper right", fontsize=8)
                
                # Store mean and median stats
                cluster_stats_dict[f"{title_prefix}_{name}_mean"] = mean_stats
                cluster_stats_dict[f"{title_prefix}_{name}_median"] = median_stats

        # Run clustering for both cases
        st.subheader("Clustering Comparison: With and Without IWGDF Grade")
        run_clustering(X_scaled_with, full_df_cleaned_with, true_labels, coords_with, "With Grade", axes, 0, feature_importance_dict_with)
        run_clustering(X_scaled_without, full_df_cleaned_without, true_labels, coords_without, "Without Grade", axes, 1, feature_importance_dict_without)

        # Explicitly draw the figure to avoid internal state changes
        fig.canvas.draw()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Close figure to free memory

        # Display metrics
        st.write("### Clustering Metrics (With IWGDF Grade)")
        metrics_df_with = pd.DataFrame(metrics_with).sort_values(by="Silhouette", ascending=False)
        st.dataframe(metrics_df_with.style.format("{:.3f}", subset=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI"]))

        st.write("### Clustering Metrics (Without IWGDF Grade)")
        metrics_df_without = pd.DataFrame(metrics_without).sort_values(by="Silhouette", ascending=False)
        st.dataframe(metrics_df_without.style.format("{:.3f}", subset=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI"]))

        # Display confusion matrices in one figure
        st.write("### Confusion Matrices (All IWGDF Grades vs Clusters)")
        n_matrices = len(confusion_matrices)
        n_rows = (n_matrices + 1) // 2  # Ceiling division to ensure enough rows
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
        axes = axes.ravel() if n_matrices > 1 else [axes]
        
        for idx, (key, conf_matrix) in enumerate(confusion_matrices.items()):
            if idx < len(axes):  # Ensure we don't access out-of-bounds axes
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(key)
                axes[idx].set_xlabel('Predicted Clusters')
                axes[idx].set_ylabel('IWGDF Grades')
        
        # Hide any unused subplots
        for idx in range(len(confusion_matrices), len(axes)):
            axes[idx].set_visible(False)
        
        fig.canvas.draw()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Close figure to free memory

        # Outlier Analysis
        st.write("### Outlier Analysis")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_scaled_without)
        outliers = full_df_cleaned_without[outlier_labels == -1]
        st.write(f"Number of outliers detected: {len(outliers)}")
        st.write("Outlier Patient File Numbers:")
        st.dataframe(outliers[['Patient_File_Number']])

        # IWGDF Group Homogeneity Analysis
        st.write("### IWGDF Group Homogeneity Analysis")
        
        # Calculate within-group inertia
        inertia = calculate_inertia(X_scaled_without, true_labels)
        st.write(f"Within-group inertia for IWGDF groups: {inertia:.3f}")
        st.write("**Interpretation**: Within-group inertia measures the compactness of IWGDF groups. The value suggests moderate to high variability within grades, indicating that points within each IWGDF grade are relatively spread out from their centroid.")

        # Check sample sizes per IWGDF grade
        grade_counts = true_labels.value_counts().sort_index()
        st.write("#### Sample Sizes per IWGDF Grade")
        st.write(grade_counts)
        st.write("**Note**: MANOVA requires sufficient samples per group (typically > number of features). If any grade has very few samples, results may be unreliable.")

        # MANOVA with IWGDF grade as factor
        st.write("#### MANOVA with IWGDF Grade as Factor")
        manova_data = full_df_cleaned_without.drop(columns=['Patient_File_Number']).copy()
        manova_data['IWGDF'] = true_labels
        formula = ' + '.join(full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns) + ' ~ IWGDF'
        try:
            manova = MANOVA.from_formula(formula, data=manova_data)
            manova_result = manova.mv_test()
            st.write("MANOVA Results:")
            st.write(manova_result)
            st.write("**Interpretation**: A p-value of 0 (or <0.001) across all test statistics (e.g., Pillai’s Trace, Wilks’ Lambda) indicates highly significant differences between IWGDF groups, suggesting they are well-separated in the multivariate parameter space. However, verify data quality (e.g., sufficient samples per group, no multicollinearity) to ensure reliability.")
        except Exception as e:
            st.warning(f"MANOVA failed: {str(e)}. Check data for sufficient samples, valid features, or multicollinearity.")

        # Silhouette score for IWGDF labels as imposed clusters
        silhouette_iwgdf = silhouette_score(X_scaled_without, true_labels)
        st.write(f"Silhouette Score for IWGDF labels as imposed clusters: {silhouette_iwgdf:.3f}")
        st.write("**Interpretation**: A low Silhouette Score (close to 0) indicates high overlap between IWGDF groups, suggesting poor separation. A higher score (>0.5) suggests well-defined groups.")

        # Parameter Importance Analysis
        st.write("### Parameter Importance Analysis")
        
        # LASSO Regression for automatic variable selection
        st.write("#### Logistic Regression with LASSO Regularization")
        try:
            lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            lasso.fit(X_scaled_without, true_labels)
            lasso_importance = pd.Series(np.abs(lasso.coef_[0]), index=full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns)
            lasso_importance = lasso_importance.sort_values(ascending=False).head(5)
            st.write("LASSO regression allows automatic selection of the most predictive variables. Non-zero coefficients highlight key parameters for IWGDF grade prediction.")
            lasso_df = pd.DataFrame({'Parameter': lasso_importance.index, 'Coefficient': lasso_importance.values})
            st.dataframe(lasso_df)
        except Exception as e:
            st.warning(f"LASSO regression failed: {str(e)}. Check data for valid features or sufficient samples.")

        # Random Forest Classifier for feature importance
        st.write("#### Random Forest Classifier Feature Importance")
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_scaled_without, true_labels)
        rf_importance = pd.Series(rf_classifier.feature_importances_, index=full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns)
        rf_importance = rf_importance.sort_values(ascending=False).head(5)
        st.write("Random Forest Classifier extracts the relative importance of variables in distinguishing IWGDF grades.")
        rf_df = pd.DataFrame({'Parameter': rf_importance.index, 'Importance': rf_importance.values})
        st.dataframe(rf_df)

        # Univariate ANOVAs and Eta Squared
        st.write("#### Univariate ANOVA with Eta Squared")
        eta_squared = {}
        for col in full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns:
            groups = [full_df_cleaned_without[col][true_labels == grade].dropna() for grade in sorted(true_labels.unique())]
            if all(len(g) > 0 for g in groups):
                f_stat, p_value = f_oneway(*groups)
                ss_total = np.sum((full_df_cleaned_without[col] - full_df_cleaned_without[col].mean())**2)
                ss_between = sum(len(g) * (g.mean() - full_df_cleaned_without[col].mean())**2 for g in groups)
                eta_squared[col] = ss_between / ss_total if ss_total > 0 else 0
        eta_squared_df = pd.DataFrame.from_dict(eta_squared, orient='index', columns=['Eta Squared']).sort_values(by='Eta Squared', ascending=False).head(5)
        st.write("Univariate ANOVAs calculate R² (eta squared) for each parameter, establishing a clear hierarchy of the most influential variables for IWGDF group differences.")
        st.dataframe(eta_squared_df)

        # Display feature importance from clustering
        st.write("### Feature Importance (With IWGDF Grade)")
        for algo_name, importance in feature_importance_dict_with.items():
            st.write(f"#### {algo_name}")
            top_features = importance.sort_values(ascending=False).head(5)
            top_features_df = pd.DataFrame({'Parameter': top_features.index, 'Importance': top_features.values})
            st.dataframe(top_features_df)

        st.write("### Feature Importance (Without IWGDF Grade)")
        for algo_name, importance in feature_importance_dict_without.items():
            st.write(f"#### {algo_name}")
            top_features = importance.sort_values(ascending=False).head(5)
            top_features_df = pd.DataFrame({'Parameter': top_features.index, 'Importance': top_features.values})
            st.dataframe(top_features_df)

        # Display mean and median statistics
        for algo_name in clustering_algos.keys():
            if algo_name.startswith("Agglomerative (ward)") and (X_scaled_with.shape[1] < 2 or X_scaled_without.shape[1] < 2):
                continue
            st.write(f"#### Statistics for {algo_name}")
            
            st.write("**With IWGDF Grade**")
            st.write("Mean Values per Cluster")
            mean_key = f"With Grade_{algo_name}_mean"
            if mean_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[mean_key].style.format("{:.2f}"))
            
            st.write("Median Values per Cluster")
            median_key = f"With Grade_{algo_name}_median"
            if median_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[median_key].style.format("{:.2f}"))
            
            st.write("**Without IWGDF Grade**")
            st.write("Mean Values per Cluster")
            mean_key = f"Without Grade_{algo_name}_mean"
            if mean_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[mean_key].style.format("{:.2f}"))
            
            st.write("Median Values per Cluster")
            median_key = f"Without Grade_{algo_name}_median"
            if mean_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[median_key].style.format("{:.2f}"))

        # Interpretation
        st.markdown("""
        ### Interpretation:
        - **Outlier Detection**:
            - Outliers are detected using Isolation Forest with a contamination rate of 0.1 (10% of data points assumed as outliers).
            - Outlier patient file numbers are listed to identify potentially anomalous cases for further clinical review.
        - **Clustering Visualization**:
            - **Left Column (With Grade)**: Clusters include IWGDF grade as a feature. Points are colored by cluster labels (0: green, 1: blue, 2: orange, 3: red) with symbols for IWGDF grades (0: circle, 1: square, 2: triangle, 3: star). Each point is annotated with the patient file number.
            - **Right Column (Without Grade)**: Clusters exclude IWGDF grade, relying on biomechanical, vascular, and thermal parameters. Same color/symbol scheme for comparison, with patient file numbers annotated.
        - **Clustering Metrics**:
            - **Silhouette Score**: Values >0.5 indicate well-separated clusters; values near 0 or negative suggest overlap.
            - **Calinski-Harabasz**: Higher values indicate better-defined clusters with high between-cluster variance.
            - **Davies-Bouldin**: Lower values suggest compact, well-separated clusters; higher values indicate overlap.
            - **ARI**: Values near 1 show strong agreement with IWGDF grades; low values suggest alternative groupings.
            - **NMI**: Values near 1 indicate high shared information with IWGDF grades; low values suggest distinct clusters.
        - **Confusion Matrices**: Show alignment between IWGDF grades (0, 1, 2, 3) and cluster labels. High diagonal values indicate good correspondence; off-diagonal values suggest misalignments, potentially indicating limitations in IWGDF grading.
        - **IWGDF Group Homogeneity**:
            - **Within-group Inertia**: Measures the compactness of IWGDF groups by summing the squared distances of points to their grade centroids. Moderate to high values suggest variability within grades.
            - **MANOVA (IWGDF Grade as Factor)**: A p-value of 0 (or <0.001) indicates significant differences between IWGDF groups, suggesting they are well-separated in the multivariate parameter space. Verify data quality to ensure reliability.
            - **Silhouette Score for IWGDF Labels**: A low score indicates high overlap between groups, suggesting poor separation. A higher score (>0.5) suggests well-defined groups.
        - **Parameter Importance**:
            - **Logistic Regression with LASSO Regularization**: Non-zero coefficients highlight key discriminators for IWGDF grades.
            - **Random Forest Classifier**: Higher importance scores indicate greater contribution to classification.
            - **Univariate ANOVAs with R² (Eta Squared)**: Higher eta squared values indicate parameters explaining more variance between grades.
        - **Key Insights**:
            - Outlier detection helps identify patients with unusual parameter profiles, which may warrant further investigation.
            - Patient file numbers on plots enable tracking of individual cases within clusters.
            - Low ARI/NMI in the without-grade case suggests natural groupings may differ from IWGDF grades, potentially offering alternative stratifications.
            - Parameters with high LASSO coefficients, Random Forest importance, or eta squared values guide clinical focus.
        """)

    # ================================
    # Clustering with All Parameters
    # ================================
    elif analysis_type == "Clustering (All Parameters)":
        st.header("Clustering with All Parameters")
        important_params = {
            6: "Date of Birth",
            16: "Grade IWGDF",
            17: "Height (m)", 18: "Weight (kg)", 19: "BMI",
            24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)",
            72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R", 110: "Avg Pressure Max TM5 R",
            113: "Avg Pressure Max SESA L", 114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Thickness ED SESA R", 127: "US Thickness ED HALLUX R", 128: "US Thickness ED TM5 R",
            130: "US Thickness ED SESA L", 131: "US Thickness ED HALLUX L", 132: "US Thickness ED TM5 L",
            134: "US Thickness Hypodermis SESA R", 135: "US Thickness Hypodermis HALLUX R", 136: "US Thickness Hypodermis TM5 R",
            138: "US Thickness Hypodermis SESA L", 139: "US Thickness Hypodermis HALLUX L", 140: "US Thickness Hypodermis TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R", 144: "Total Tissue Thickness TM5 R",
            146: "Total Tissue Thickness SESA L", 147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
            154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R", 160: "Temperature Plantar Arch R",
            161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R", 163: "Temperature Heel R",
            164: "Temperature Hallux L", 165: "Temperature 5th Toe L", 166: "Temperature Plantar Arch L",
            167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L", 169: "Temperature Heel L",
            170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)", 173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L",
            176: "Normalized Temperature R", 177: "Normalized Temperature L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L",
            214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L",
        }

        # Debug: Print analysis type
        st.write(f"Current analysis type: {analysis_type}")

        # Extract IWGDF grades and patient file numbers
        risk_row = df.iloc[16]
        risk_values = pd.to_numeric(risk_row[1:], errors='coerce').dropna().astype(int)
        patient_ids = risk_values.index.tolist()

        # Select number of clusters
        k = st.slider("Number of clusters", 2, 6, 4)

        # Define clustering algorithms
        clustering_algos = {
            "Agglomerative (ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "Agglomerative (average)": AgglomerativeClustering(n_clusters=k, linkage="average"),
            "KMeans": KMeans(n_clusters=k, random_state=42),
            "Gaussian Mixture Model": GaussianMixture(n_components=k, random_state=42),
        }

        # Color map for clusters
        color_map = {0: "green", 1: "blue", 2: "orange", 3: "red"}

        # Marker map for IWGDF grades
        marker_map = {0: 'o', 1: 's', 2: '^', 3: '*'}

        def clean_column_name(name):
            name = re.sub(r'[^\w\s]', '', name).strip().replace(' ', '_')
            if not name[0].isalpha():
                name = 'p_' + name
            return name

        def prepare_data(include_grade=True):
            params = important_params.copy()
            if not include_grade:
                params.pop(16, None)
            
            selected_features = df.loc[list(params.keys())]
            cleaned_index = {k: clean_column_name(v) for k, v in params.items()}
            selected_features.index = [cleaned_index[k] for k in selected_features.index]
            selected_features = selected_features.loc[:, patient_ids]
            
            # Identify categorical columns
            categorical_cols = [cleaned_index[k] for k in [24, 75, 76, 77, 78] if k in params]
            numeric_cols = [col for col in selected_features.index if col not in categorical_cols]
            
            # Convert numeric columns
            features_df = selected_features.loc[numeric_cols].T.apply(pd.to_numeric, errors='coerce')
            features_df['Patient_File_Number'] = features_df.index
            
            # Encode categorical columns
            if categorical_cols:
                cat_df = selected_features.loc[categorical_cols].T
                cat_df_encoded = pd.get_dummies(cat_df, dummy_na=True)
                features_df = pd.concat([features_df, cat_df_encoded], axis=1)
            
            full_df = features_df.copy()
            true_labels = risk_values.reindex(full_df.index)
            
            # Clean data
            valid_features = full_df.isnull().sum() < full_df.shape[0] * 0.5
            full_df_cleaned = full_df.loc[:, valid_features]
            
            full_df_cleaned = full_df_cleaned.loc[:, full_df_cleaned.count() >= 10]
            full_df_cleaned = full_df_cleaned.apply(lambda col: col.fillna(col.median()) if col.name != 'Patient_File_Number' else col, axis=0)
            
            if full_df_cleaned.shape[1] <= 1 or full_df_cleaned.shape[0] < 2:
                return None, None, None, None
            
            # Outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(full_df_cleaned.drop(columns=['Patient_File_Number']))
            non_outliers = outlier_labels == 1
            full_df_cleaned = full_df_cleaned[non_outliers]
            true_labels = true_labels[non_outliers]
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(full_df_cleaned.drop(columns=['Patient_File_Number']))
            
            return X_scaled, full_df_cleaned, true_labels, scaler

        # Prepare data for both cases
        X_scaled_with, full_df_cleaned_with, true_labels_with, scaler_with = prepare_data(include_grade=True)
        X_scaled_without, full_df_cleaned_without, true_labels_without, scaler_without = prepare_data(include_grade=False)

        if X_scaled_with is None or X_scaled_without is None:
            st.warning("Not enough valid data for clustering with all parameters after outlier removal.")
            st.stop()

        # Align indices
        common_indices = full_df_cleaned_with.index.intersection(full_df_cleaned_without.index)
        if len(common_indices) < 2:
            st.warning("Not enough overlapping patients for clustering after outlier removal.")
            st.stop()

        X_scaled_with = X_scaled_with[full_df_cleaned_with.index.isin(common_indices)]
        X_scaled_without = X_scaled_without[full_df_cleaned_without.index.isin(common_indices)]
        true_labels = true_labels_with.loc[common_indices]
        full_df_cleaned_with = full_df_cleaned_with.loc[common_indices]
        full_df_cleaned_without = full_df_cleaned_without.loc[common_indices]

        # Perform PCA for visualization
        pca_with = PCA(n_components=2)
        coords_with = pca_with.fit_transform(X_scaled_with)
        st.write(f"Explained variance ratio (With Grade): {pca_with.explained_variance_ratio_}")
        pca_without = PCA(n_components=2)
        coords_without = pca_without.fit_transform(X_scaled_without)
        st.write(f"Explained variance ratio (Without Grade): {pca_without.explained_variance_ratio_}")

        # Streamlit header and parameter display
        st.markdown("""
        The parameters used for clustering are:
        - IWGDF Grade
        - Anthropometric (Height, Weight, BMI)
        - AOMI
        - MESI (Ankle Pressure, Big Toe Systolic Pressure Index)
        - Michigan Score
        - Charcot (Acute and Chronic)
        - Amplitude (Dorsiflexion, Talo-crural)
        - Pressure (Max SESA, HALLUX, TM5)
        - Stiffness (SESA, HALLUX, TM5)
        - Ultrasound Thickness (Epidermis, Hypodermis, Total Tissue)
        - ROC (SESA, HALLUX, TM5)
        - Temperature (Foot regions, Hand-Foot Differences, Normalized)
        - SUDOSCAN (Hand and Foot)
        """)
        
        st.markdown("---")
        st.caption("Agglomerative (ward): Hierarchically merges clusters by minimizing variance within clusters.")
        st.caption("Agglomerative (average): Hierarchically merges clusters using average distance between points.")
        st.caption("KMeans: Partitions data into k clusters by minimizing distance to cluster centroids.")
        st.caption("Gaussian Mixture Model: Clusters data using probabilistic Gaussian distributions.")
        st.markdown("---")
        
        # Corrected dictionary initialization
        metrics_with = {"Algorithm": [], "Silhouette": [], "Calinski-Harabasz": [], "Davies-Bouldin": [], "ARI": [], "NMI": []}
        metrics_without = {"Algorithm": [], "Silhouette": [], "Calinski-Harabasz": [], "Davies-Bouldin": [], "ARI": [], "NMI": []}
        feature_importance_dict_with = {}
        feature_importance_dict_without = {}
        confusion_matrices = {}
        cluster_stats_dict = {}

        # Create subplots (2 columns: with grade, without grade)
        n_algos = len(clustering_algos)
        fig, axes = plt.subplots(n_algos, 2, figsize=(12, 5 * n_algos))
        axes = axes.reshape(n_algos, 2) if n_algos > 1 else np.array([axes])

        # Function to calculate within-group inertia
        def calculate_inertia(X, labels):
            inertia = 0
            for cluster in np.unique(labels):
                cluster_points = X[labels == cluster]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            return inertia

        # Function to run clustering and plot
        def run_clustering(X_scaled, full_df_cleaned, true_labels, coords, title_prefix, axes, col_idx, feature_importance_dict):
            for idx, (name, algo) in enumerate(clustering_algos.items()):
                if name.startswith("Agglomerative (ward)") and X_scaled.shape[1] < 2:
                    st.warning(f"Ward linkage requires at least 2 features for {title_prefix}. Skipping.")
                    continue
                
                # Perform clustering
                labels = algo.fit_predict(X_scaled)
                silhouette = silhouette_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)
                ari = adjusted_rand_score(true_labels, labels)
                nmi = normalized_mutual_info_score(true_labels, labels)
                
                metrics = metrics_with if "With Grade" in title_prefix else metrics_without
                metrics["Algorithm"].append(name)
                metrics["Silhouette"].append(silhouette)
                metrics["Calinski-Harabasz"].append(ch)
                metrics["Davies-Bouldin"].append(db)
                metrics["ARI"].append(ari)
                metrics["NMI"].append(nmi)
                
                # Calculate confusion matrix
                conf_matrix = confusion_matrix(true_labels, labels)
                confusion_matrices[f"{title_prefix}_{name}"] = conf_matrix
                
                # Calculate feature importance
                rf = RandomForestClassifier(random_state=42)
                rf.fit(X_scaled, labels)
                feature_importance_dict[name] = pd.Series(rf.feature_importances_, index=full_df_cleaned.drop(columns=['Patient_File_Number']).columns)
                
                # Calculate mean and median per cluster
                cluster_stats = full_df_cleaned.copy()
                cluster_stats['Cluster'] = labels
                mean_stats = cluster_stats.groupby('Cluster').mean()
                median_stats = cluster_stats.groupby('Cluster').median()
                
                # Plot with cluster colors, grade markers, and patient file numbers
                ax = axes[idx, col_idx]
                for grade in sorted(true_labels.unique()):
                    mask = true_labels == grade
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=[color_map.get(lbl, "black") for lbl in labels[mask]], 
                        marker=marker_map[grade], s=80, label=f"Grade {grade}"
                    )
                    # Add patient file numbers as annotations with improved legibility
                    for i, patient_id in enumerate(full_df_cleaned['Patient_File_Number'][mask]):
                        ax.annotate(
                            patient_id, 
                            (coords[mask, 0][i] + 0.02, coords[mask, 1][i] + 0.02),  # Slight offset
                            fontsize=10, 
                            fontweight='bold', 
                            alpha=0.9,
                        )

                ax.set_title(f"{name} ({title_prefix})\nSilhouette={silhouette:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}", fontsize=14, pad=20)
                ax.set_xlabel("PCA1", fontsize=10)
                ax.set_ylabel("PCA2", fontsize=10)
                
                # Create legends with restored marker shapes
                grade_handles = [plt.Line2D([0], [0], marker=marker_map[grade], color='w', label=f"Grade {grade}", 
                                            markerfacecolor='gray', markersize=10) 
                                for grade in sorted(true_labels.unique())]
                cluster_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Cluster {i}", 
                                            markerfacecolor=color_map.get(i, "black"), markersize=10) 
                                for i in range(k)]
                ax.legend(handles=grade_handles + cluster_handles, title="IWGDF Grades & Clusters", loc="upper left", fontsize=8)
                
                cluster_handles = [
                    plt.Line2D(
                        [0], [0], 
                        marker='o', 
                        color='w', 
                        label=f"Cluster {i}", 
                        markerfacecolor=color_map.get(i, "black"), 
                        markeredgecolor='black',
                        markersize=10
                    ) 
                    for i in range(k)
                ]
                ax.legend(handles=grade_handles + cluster_handles, title="IWGDF Grades & Clusters", loc="upper right", fontsize=8)
                
                # Store mean and median stats
                cluster_stats_dict[f"{title_prefix}_{name}_mean"] = mean_stats
                cluster_stats_dict[f"{title_prefix}_{name}_median"] = median_stats

        # Run clustering for both cases
        st.subheader("Clustering Comparison: With and Without IWGDF Grade")
        run_clustering(X_scaled_with, full_df_cleaned_with, true_labels, coords_with, "With Grade", axes, 0, feature_importance_dict_with)
        run_clustering(X_scaled_without, full_df_cleaned_without, true_labels, coords_without, "Without Grade", axes, 1, feature_importance_dict_without)

        # Explicitly draw the figure
        fig.canvas.draw()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Display metrics
        st.write("### Clustering Metrics (With IWGDF Grade)")
        metrics_df_with = pd.DataFrame(metrics_with).sort_values(by="Silhouette", ascending=False)
        st.dataframe(metrics_df_with.style.format("{:.3f}", subset=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI"]))

        st.write("### Clustering Metrics (Without IWGDF Grade)")
        metrics_df_without = pd.DataFrame(metrics_without).sort_values(by="Silhouette", ascending=False)
        st.dataframe(metrics_df_without.style.format("{:.3f}", subset=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI"]))

        # Display confusion matrices
        st.write("### Confusion Matrices (All IWGDF Grades vs Clusters)")
        n_matrices = len(confusion_matrices)
        n_rows = (n_matrices + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
        axes = axes.ravel() if n_matrices > 1 else [axes]
        
        for idx, (key, conf_matrix) in enumerate(confusion_matrices.items()):
            if idx < len(axes):
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(key)
                axes[idx].set_xlabel('Predicted Clusters')
                axes[idx].set_ylabel('IWGDF Grades')
        
        for idx in range(len(confusion_matrices), len(axes)):
            axes[idx].set_visible(False)
        
        fig.canvas.draw()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Outlier Analysis
        st.write("### Outlier Analysis")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_scaled_without)
        outliers = full_df_cleaned_without[outlier_labels == -1]
        st.write(f"Number of outliers detected: {len(outliers)}")
        st.write("Outlier Patient File Numbers:")
        st.dataframe(outliers[['Patient_File_Number']])

        # IWGDF Group Homogeneity Analysis
        st.write("### IWGDF Group Homogeneity Analysis")

        inertia = calculate_inertia(X_scaled_without, true_labels)
        st.write(f"Within-group inertia for IWGDF groups: {inertia:.3f}")
        st.write("**Interpretation**: Within-group inertia measures the compactness of IWGDF groups. Lower values indicate tighter clusters.")

        grade_counts = true_labels.value_counts().sort_index()
        st.write("#### Sample Sizes per IWGDF Grade")
        st.write(grade_counts)
        st.write("**Note**: MANOVA requires sufficient samples per group (typically > number of features).")

        # MANOVA with IWGDF Grade as Factor
        st.write("#### MANOVA with IWGDF Grade as Factor")
        manova_data = full_df_cleaned_without.drop(columns=['Patient_File_Number']).copy()
        manova_data['IWGDF'] = true_labels

        # Clean column names for MANOVA formula
        def clean_manova_column_name(name):
            name = re.sub(r'[^\w\s]', '', name).strip().replace(' ', '_')
            if not name[0].isalpha():
                name = 'p_' + name
            return name

        # Apply cleaning to column names
        manova_data.columns = [clean_manova_column_name(col) if col != 'IWGDF' else col for col in manova_data.columns]

        # Check for sufficient data
        n_features = manova_data.shape[1] - 1  # Exclude 'IWGDF'
        n_samples_per_group = manova_data.groupby('IWGDF').size()
        min_samples = n_samples_per_group.min()

        if min_samples <= n_features:
            st.warning(f"MANOVA may be unreliable: smallest group size ({min_samples}) is not greater than number of features ({n_features}).")
        elif min_samples < 2:
            st.warning("MANOVA failed: At least one IWGDF group has fewer than 2 samples.")
        else:
            try:
                # Construct formula with cleaned column names
                formula = ' + '.join([col for col in manova_data.columns if col != 'IWGDF']) + ' ~ IWGDF'
                st.write(f"MANOVA formula: {formula}")
                manova = MANOVA.from_formula(formula, data=manova_data)
                manova_result = manova.mv_test()
                st.write("MANOVA Results:")
                st.write(manova_result)
                st.write("**Interpretation**: A p-value < 0.05 across test statistics (e.g., Pillai’s Trace, Wilks’ Lambda) indicates significant differences between IWGDF groups.")
            except Exception as e:
                st.warning(f"MANOVA failed: {str(e)}. Possible causes: invalid column names, insufficient samples, multicollinearity, or non-numeric data.")
                st.write("Column names used:", list(manova_data.columns))
                st.write("Data types:", manova_data.dtypes)
                st.write("Sample sizes per group:", n_samples_per_group)

        silhouette_iwgdf = silhouette_score(X_scaled_without, true_labels)
        st.write(f"Silhouette Score for IWGDF labels: {silhouette_iwgdf:.3f}")
        st.write("**Interpretation**: Higher scores (>0.5) indicate well-separated groups.")

        # Parameter Importance Analysis
        st.write("### Parameter Importance Analysis")
        
        st.write("#### Logistic Regression with LASSO Regularization")
        try:
            lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            lasso.fit(X_scaled_without, true_labels)
            lasso_importance = pd.Series(np.abs(lasso.coef_[0]), index=full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns)
            lasso_importance = lasso_importance.sort_values(ascending=False).head(5)
            st.dataframe(pd.DataFrame({'Parameter': lasso_importance.index, 'Coefficient': lasso_importance.values}))
        except Exception as e:
            st.warning(f"LASSO regression failed: {str(e)}.")

        st.write("#### Random Forest Classifier Feature Importance")
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_scaled_without, true_labels)
        rf_importance = pd.Series(rf_classifier.feature_importances_, index=full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns)
        rf_importance = rf_importance.sort_values(ascending=False).head(5)
        st.dataframe(pd.DataFrame({'Parameter': rf_importance.index, 'Importance': rf_importance.values}))

        st.write("#### Univariate ANOVA with Eta Squared")
        eta_squared = {}
        for col in full_df_cleaned_without.drop(columns=['Patient_File_Number']).columns:
            groups = [full_df_cleaned_without[col][true_labels == grade].dropna() for grade in sorted(true_labels.unique())]
            if all(len(g) > 0 for g in groups):
                f_stat, p_value = f_oneway(*groups)
                ss_total = np.sum((full_df_cleaned_without[col] - full_df_cleaned_without[col].mean())**2)
                ss_between = sum(len(g) * (g.mean() - full_df_cleaned_without[col].mean())**2 for g in groups)
                eta_squared[col] = ss_between / ss_total if ss_total > 0 else 0
        eta_squared_df = pd.DataFrame.from_dict(eta_squared, orient='index', columns=['Eta Squared']).sort_values(by='Eta Squared', ascending=False).head(5)
        st.dataframe(eta_squared_df)

        # Display feature importance from clustering
        st.write("### Feature Importance (With IWGDF Grade)")
        for algo_name, importance in feature_importance_dict_with.items():
            st.write(f"#### {algo_name}")
            top_features = importance.sort_values(ascending=False).head(5)
            st.dataframe(pd.DataFrame({'Parameter': top_features.index, 'Importance': top_features.values}))

        st.write("### Feature Importance (Without IWGDF Grade)")
        for algo_name, importance in feature_importance_dict_without.items():
            st.write(f"#### {algo_name}")
            top_features = importance.sort_values(ascending=False).head(5)
            st.dataframe(pd.DataFrame({'Parameter': top_features.index, 'Importance': top_features.values}))

        # Display mean and median statistics
        for algo_name in clustering_algos.keys():
            if algo_name.startswith("Agglomerative (ward)") and (X_scaled_with.shape[1] < 2 or X_scaled_without.shape[1] < 2):
                continue
            st.write(f"#### Statistics for {algo_name}")
            
            st.write("**With IWGDF Grade**")
            st.write("Mean Values per Cluster")
            mean_key = f"With Grade_{algo_name}_mean"
            if mean_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[mean_key].style.format("{:.2f}"))
            
            st.write("Median Values per Cluster")
            median_key = f"With Grade_{algo_name}_median"
            if median_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[median_key].style.format("{:.2f}"))
            
            st.write("**Without IWGDF Grade**")
            st.write("Mean Values per Cluster")
            mean_key = f"Without Grade_{algo_name}_mean"
            if mean_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[mean_key].style.format("{:.2f}"))
            
            st.write("Median Values per Cluster")
            median_key = f"Without Grade_{algo_name}_median"
            if mean_key in cluster_stats_dict:
                st.dataframe(cluster_stats_dict[median_key].style.format("{:.2f}"))

        # Interpretation
        st.markdown("""
        ### Interpretation:
        - **Outlier Detection**:
            - Outliers are detected using Isolation Forest with a contamination rate of 0.1 (10% of data points assumed as outliers).
            - Outlier patient file numbers are listed to identify potentially anomalous cases for further clinical review.
        - **Clustering Visualization**:
            - **Left Column (With Grade)**: Clusters include IWGDF grade as a feature. Points are colored by cluster labels (0: green, 1: blue, 2: orange, 3: red) with symbols for IWGDF grades (0: circle, 1: square, 2: triangle, 3: star). Each point is annotated with the patient file number, styled with a larger, bold font and a semi-transparent white background for improved legibility.
            - **Right Column (Without Grade)**: Clusters exclude IWGDF grade, relying on all other parameters. Same color/symbol scheme for comparison, with patient file numbers annotated similarly.
            - **Legend**: The legend restores the original marker shapes, with IWGDF grades shown as circles, squares, triangles, and stars, and clusters shown as circles with their respective colors.
        - **Clustering Metrics**:
            - **Silhouette Score**: Values >0.5 indicate well-separated clusters; values near 0 or negative suggest overlap.
            - **Calinski-Harabasz**: Higher values indicate better-defined clusters with high between-cluster variance.
            - **Davies-Bouldin**: Lower values suggest compact, well-separated clusters; higher values indicate overlap.
            - **ARI**: Values near 1 show strong agreement with IWGDF grades; low values suggest alternative groupings.
            - **NMI**: Values near 1 indicate high shared information with IWGDF grades; low values suggest distinct clusters.
        - **Confusion Matrices**: Show alignment between IWGDF grades (0, 1, 2, 3) and cluster labels. High diagonal values indicate good correspondence; off-diagonal values suggest misalignments.
        - **IWGDF Group Homogeneity**:
            - **Within-group Inertia**: Measures the compactness of IWGDF groups. Lower values indicate tighter clusters.
            - **MANOVA**: A p-value < 0.05 indicates significant differences between IWGDF groups.
            - **Silhouette Score for IWGDF Labels**: Higher scores (>0.5) indicate well-separated groups.
        - **Parameter Importance**:
            - **Logistic Regression with LASSO Regularization**: Non-zero coefficients highlight key discriminators for IWGDF grades.
            - **Random Forest Classifier**: Higher importance scores indicate greater contribution to classification.
            - **Univariate ANOVAs with Eta Squared**: Higher eta squared values indicate parameters explaining more variance between grades.
        - **Key Insights**:
            - Outlier detection helps identify patients with unusual parameter profiles.
            - Patient file numbers on plots, with enhanced legibility, enable tracking of individual cases within clusters.
            - Low ARI/NMI in the without-grade case suggests natural groupings may differ from IWGDF grades.
            - Parameters with high LASSO coefficients, Random Forest importance, or eta squared values guide clinical focus.
        """)
    # ================================
    # 📌 Correlation Between Key Parameters
    # ================================
    elif analysis_type == "Correlation Between Key Parameters":
        st.subheader("🔗 Correlation Between Selected DIAFOOT Parameters")
        target_rows = {
            6: "Date de naissance", 16:"Grade IWGDF", 19: "BMI",
            24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Amplitude of dorsiflexion of right MTP1 R",
            95: "Amplitude of dorsiflexion of right MTP1 L", 96: "Amplitude talo-crurale R",
            97: "Amplitude talo-crurale L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Épaisseur ED SESA R", 127: "US Épaisseur ED HALLUX R", 128: "US Épaisseur ED TM5 R",
            130: "US Épaisseur ED SESA L", 131: "US Épaisseur ED HALLUX L", 132: "US Épaisseur ED TM5 L",
            134: "US Épaisseur Hypoderme SESA R", 135: "US Épaisseur Hypoderme HALLUX R",
            136: "US Épaisseur Hypoderme TM5 R", 138: "US Épaisseur Hypoderme SESA L",
            139: "US Épaisseur Hypoderme HALLUX L", 140: "US Épaisseur Hypoderme TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",            
            158: "Temperature Hallux R", 159: "Temperature 5th Toe R",
            160: "Temperature Plantar Arch R", 161: "Temperature Lateral Sole R", 162: "Temperature Forefoot R",
            163: "Temperature Heel R", 164: "Temperature Hallux L", 165: "Temperature 5th Toe L",
            166: "Temperature Plantar Arch L", 167: "Temperature Lateral Sole L", 168: "Temperature Forefoot L",
            169: "Temperature Heel L", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean L",
            172: "Average IR Temperature Foot R (Celsius)",173: "Average IR Temperature Foot L (Celsius)",
            174: "Temperature Difference Hand-Foot R",175: "Temperature Difference Hand-Foot L",176: "Normalized Temperature R",
            177: "Normalized Temperature L", 212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R",
            215: "SUDOSCAN Foot L",
        }
        # Clean the full DataFrame (drop rows with too many NaNs)
        df_corr = df_combined.dropna(thresh=5)

        # Select only numeric columns
        df_corr_numeric = df_corr.select_dtypes(include=[np.number])

        if df_corr_numeric.shape[0] >= 2:
            # Compute correlation matrix
            corr_matrix = df_corr_numeric.corr()

            # Show highly correlated pairs
            st.markdown("### 🔍 Highly Correlated Pairs")
            threshold = st.slider("Threshold", 0.5, 1.0, 0.8, 0.05)
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        high_corr.append({
                            "Feature 1": corr_matrix.columns[i],
                            "Feature 2": corr_matrix.columns[j],
                            "Correlation": round(corr_val, 3)
                        })

            if high_corr:
                high_corr_df = pd.DataFrame(high_corr)

                # Show in Streamlit
                st.dataframe(high_corr_df)

                towrite = BytesIO()
                high_corr_df = pd.DataFrame(high_corr)  # Ensure this is a DataFrame

                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    high_corr_df.to_excel(writer, index=False, sheet_name='High Correlations')

                # Important: reset after closing the writer
                towrite.seek(0)

                st.download_button(
                    label="📥 Download Highly Correlated Pairs (Excel)",
                    data=towrite,
                    file_name="highly_correlated_pairs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # 📈 Correlation Visualizations
                st.markdown("---")
                st.subheader("📈 Correlation Visualizations")

                fig4, ax4 = plt.subplots() 
                x_feature = st.selectbox("Choose X-axis variable:", df_corr_numeric.columns)
                y_feature = st.selectbox("Choose Y-axis variable:", df_corr_numeric.columns, index=1)
                x = df_corr_numeric[x_feature]
                y = df_corr_numeric[y_feature]
                
                
                # Calculate bubble size and clean NaN/inf
                size = np.abs(x * y)
                size = np.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0)  # replace NaN and inf with 0

                # Filter points where x or y is NaN or inf
                mask = np.isfinite(x) & np.isfinite(y)

                ax4.scatter(x[mask], y[mask], s=size[mask], alpha=0.5, c=size[mask], cmap='coolwarm')

                if x_feature != y_feature:
                    # Scatter Plot
                    st.write("**Scatter Plot**")
                    fig1, ax1 = plt.subplots()
                    ax1.scatter(df_corr_numeric[x_feature], df_corr_numeric[y_feature], alpha=0.6)
                    ax1.set_xlabel(x_feature)
                    ax1.set_ylabel(y_feature)
                    st.pyplot(fig1)
                    st.markdown(f"ℹ️ **Scatter Plot Explanation:** Each point represents one patient. The X and Y axes are the selected parameters. The plot shows their relationship and spread.")

                    # Line plot with regression
                    st.write("**Line Plot with Regression Line**")
                    fig2 = sns.lmplot(data=df_corr_numeric, x=x_feature, y=y_feature, aspect=1.5)
                    st.pyplot(fig2)
                    st.markdown(f"ℹ️ **Regression Plot Explanation:** Shows linear trend and confidence interval between {x_feature} and {y_feature}. Helps assess correlation strength and direction.")


                    x = df_corr_numeric[x_feature]
                    y = df_corr_numeric[y_feature]

                    size = np.abs(x * y)
                    size = np.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0)

                    mask = np.isfinite(x) & np.isfinite(y)

                    # Pair Plot (scatter matrix)
                    st.write("**Pair Plot (Scatterplot Matrix)**")
                    selected_vars = st.multiselect("Select variables for pairplot (max 6):", df_corr_numeric.columns.tolist(), default=df_corr_numeric.columns[:4].tolist())
                    if len(selected_vars) >= 2:
                        fig5 = sns.pairplot(df_corr_numeric[selected_vars])
                        st.pyplot(fig5)
                        st.markdown(f"ℹ️ **Pair Plot Explanation:** Matrix of scatterplots showing relationships between pairs of selected variables.")

                    # Correlogram (Filtered for |correlation| > 0.7)
                    st.write("**Correlogram (|correlation| > 0.7)**")
                    filtered_corr = corr_matrix.copy()
                    filtered_corr[abs(filtered_corr) < 0.7] = np.nan  # hide weak correlations

                    fig6, ax6 = plt.subplots(figsize=(12, 10))
                    sns.heatmap(filtered_corr, annot=False, cmap="vlag", ax=ax6, mask=filtered_corr.isnull())
                    st.pyplot(fig6)
                    st.markdown(f"ℹ️ **Correlogram Explanation:** Heatmap showing only strong correlations (absolute value > 0.7). Helps identify highly related parameter pairs.")

    # ================================
    # Bland-Altman Plots by Parameter and Side
    # ================================
    elif analysis_type == "Bland-Altman Plots by Parameter and Side":
        st.subheader("📉 Bland-Altman Plots by Parameter and Side")

        durometre_rows = {
            118: "Durometre SESA D", 119: "Durometre HALLUX D", 120: "Durometre TM5 D",
            122: "Durometre SESA G", 123: "Durometre HALLUX G", 124: "Durometre TM5 G"
        }

        us_epaisseur_ed_rows = {
            126: "US Épaisseur ED SESA D", 127: "US Épaisseur ED HALLUX D", 128: "US Épaisseur ED TM5 D",
            130: "US Épaisseur ED SESA G", 131: "US Épaisseur ED HALLUX G", 132: "US Épaisseur ED TM5 G"
        }

        us_epaisseur_hypo_rows = {
            134: "US Épaisseur Hypoderme SESA D", 135: "US Épaisseur Hypoderme HALLUX D", 136: "US Épaisseur Hypoderme TM5 D",
            138: "US Épaisseur Hypoderme SESA G", 139: "US Épaisseur Hypoderme HALLUX G", 140: "US Épaisseur Hypoderme TM5 G"
        }

        parameter_groups = {
            "Durometre [Hardness]": durometre_rows,
            "US Épaisseur ED [Epiderm + Derm] (mm)": us_epaisseur_ed_rows,
            "US Épaisseur Hypoderme (mm)": us_epaisseur_hypo_rows
        }

        def plot_bland_altman(data1, data2, title="", unit=""):
            mean = (data1 + data2) / 2
            diff = data1 - data2
            md = np.mean(diff)
            sd = np.std(diff)  # SDr

            st.markdown(f"**➕ SDr (SD of Differences) for {title}:** {sd:.3f} {unit}")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(mean, diff, alpha=0.5)
            ax.axhline(md, color='blue', linestyle='--', label='Mean')
            ax.axhline(md + 1.96 * sd, color='red', linestyle='--', label='+1.96 SD')
            ax.axhline(md - 1.96 * sd, color='red', linestyle='--', label='-1.96 SD')
            ax.set_title(f"Bland-Altman: {title}")
            ax.set_xlabel(f"Mean {unit}")
            ax.set_ylabel("Difference")
            ax.legend()
            ax.grid(True)
            return fig

        df_numeric = df.T.apply(pd.to_numeric, errors='coerce')

        zones = ["SESA", "HALLUX", "TM5"]

        for group_name, rows in parameter_groups.items():
            st.markdown(f"### 📁 {group_name}")

            selected = df.loc[list(rows.keys()), 1:]
            selected.index = [rows[i] for i in rows.keys()]
            selected = selected.T.apply(pd.to_numeric, errors='coerce')
            all_diffs_pooled = []


            for zone in zones:
                col_d = [col for col in selected.columns if zone in col and " D" in col]
                col_g = [col for col in selected.columns if zone in col and " G" in col]

                if col_d and col_g:
                    col_d = col_d[0]
                    col_g = col_g[0]

                    data_d = selected[col_d].dropna()
                    data_g = selected[col_g].dropna()

                    min_len = min(len(data_d), len(data_g))
                    data_d = data_d[:min_len]
                    data_g = data_g[:min_len]

                    diffs = data_d - data_g
                    all_diffs_pooled.extend(diffs.values)

                    fig = plot_bland_altman(data_d, data_g,
                        title=f"{zone} - Right (D) vs Left (G)",
                        unit=group_name.split("(")[-1].replace(")", "") if "(" in group_name else ""
                    )
                    st.pyplot(fig)

                    sdr_zone = np.std(diffs)
                    st.markdown(f"**SDr pooled for {zone}:** {sdr_zone:.3f} {group_name.split('(')[-1].replace(')', '') if '(' in group_name else ''}")

            if all_diffs_pooled:
                all_diffs_pooled = np.array(all_diffs_pooled)
                overall_sdr = np.std(all_diffs_pooled)
                st.markdown(f"### 🔷 Overall pooled SDr for {group_name}: {overall_sdr:.3f} {group_name.split('(')[-1].replace(')', '') if '(' in group_name else ''}")

                all_right = []
                all_left = []
                for zone in zones:
                    col_d = [col for col in selected.columns if zone in col and " D" in col]
                    col_g = [col for col in selected.columns if zone in col and " G" in col]
                    if col_d and col_g:
                        col_d = col_d[0]
                        col_g = col_g[0]
                        d_vals = selected[col_d].dropna()
                        g_vals = selected[col_g].dropna()
                        min_len = min(len(d_vals), len(g_vals))
                        all_right.extend(d_vals[:min_len])
                        all_left.extend(g_vals[:min_len])

                all_right = np.array(all_right)
                all_left = np.array(all_left)

                if len(all_right) > 0 and len(all_left) > 0:
                    fig_pooled = plot_bland_altman(
                        all_right,
                        all_left,
                        title=f"Pooled Left vs Right - {group_name}",
                        unit=group_name.split("(")[-1].replace(")", "") if "(" in group_name else ""
                    )
                    st.pyplot(fig_pooled)
