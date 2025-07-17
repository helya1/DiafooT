import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from datetime import datetime
import io
from io import BytesIO
from scipy.stats import mannwhitneyu
from matplotlib import cm
cm.get_cmap("coolwarm")
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from sklearn.cluster import AgglomerativeClustering


# ================================
# 🌐 Streamlit Page Setup
# ================================
st.set_page_config(page_title="DIAFOOT Analysis Dashboard", layout="wide")
st.title("📊 DIAFOOT Analysis Dashboard")

# ================================
# 📤 File Upload
# ================================
uploaded_file = st.file_uploader("Upload Excel file with 'DIAFOOT' sheet", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="DIAFOOT", header=None)

    analysis_type = st.sidebar.radio(
        "🧪 Choose Analysis Type:",
        ("Basic Analysis", "Descriptive Analysis", "Normality Tests",
        "Hallux/SESA/TM5 – L/R Comparison by Parameter Type", "Comparison of Left and Right Foot Parameters",
        "Clustering ignoring IWGDF Grade", "Clustering with IWGDF Grade Groups", "GMM Clustering","IWGDF Risk Grade Summary & Clustering", "Grade  & Clustering", "Correlation Between Key Parameters",
        "Intra-Group Comparison","Bland-Altman Plots by Parameter and Side",
        "Bland-Altman Pooled Plots for all parameters",
        "Multivariate Group Comparison (MANOVA)", "Multiple Linear Regression", "Exploratory PCA")
    )

    target_rows = {
        6: "Date of Birth", 16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
        35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
        59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
        75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
        77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
        94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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
    
    target_rows_reg = {
        16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI",
        35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
        75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
        77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
        94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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
    # ================================
    # 🧹 Data Preparation Function
    # ================================
    row_labels = df.iloc[:, 0]
    df_numeric = df.iloc[:, 1:]

    def data_reg(df_num, row_labels):
        target_rows = {
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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
            174: "Temperature Difference Hand-Foot R", 175: "Temperature Difference Hand-Foot L"
        }
        # Possibly add any processing you want here, for now just return inputs as is
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
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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
        st.header("📋 Basic Descriptive Analysis")

        # ==========================
        # 🎂 Age Distribution
        # ==========================
        dob_row = df.iloc[6, 1:].dropna().astype(str).str.strip()

        # Try to parse dates with flexible day-first format
        dob_dates = pd.to_datetime(dob_row, dayfirst=True, errors='coerce')

        # Calculate age from today
        today = pd.Timestamp.today()
        ages = dob_dates.map(lambda d: (today - d).days / 365.25 if pd.notnull(d) else np.nan).dropna()

        if ages.empty:
            st.warning("⚠️ No valid date of birth data found.")
        else:
            # Stats
            average_age = ages.mean()
            std_age = ages.std()

            st.subheader("🎂 Age Distribution")
            st.write(f"**Average Age**: {average_age:.1f} years")
            st.write(f"**Standard Deviation**: {std_age:.1f}")

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(ages, bins=10, color='skyblue', edgecolor='black')
            ax.axvline(average_age, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {average_age:.1f}')
            ax.set_title("Patient Age Distribution")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)


        # GENDER DISTRIBUTION
        st.subheader("🚻 Gender Distribution")
        gender_row = df[df[0].astype(str).str.strip().str.lower() == "genre h/f/i"]
        if not gender_row.empty:
            idx = gender_row.index[0]
            gender_vals = df.iloc[idx, 1:].astype(str).str.strip().str.upper()
            gender_map = {"H": "Male", "F": "Female", "I": "Unknown"}
            counts = gender_vals.value_counts()
            valid_counts = counts[counts.index.isin(gender_map.keys())]
            valid_counts.index = valid_counts.index.map(gender_map)
            fig_gender, ax_gender = plt.subplots()
            ax_gender.pie(valid_counts, labels=valid_counts.index, autopct='%1.1f%%', 
                        startangle=140, colors=['lightblue', 'lightpink', 'gray'])
            ax_gender.axis('equal')
            st.pyplot(fig_gender)
        else:
            st.warning("⚠️ Gender row ('Genre H/F/I') not found.")

        # BMI ANALYSIS
        st.subheader("⚖️ BMI Distribution")
        bmi_row = df[df[0].astype(str).str.strip().str.lower() == "bmi (poids[kg] / taille[m]2)"]
        if not bmi_row.empty:
            bmi_vals = pd.to_numeric(df.loc[bmi_row.index[0], 1:], errors='coerce').dropna()
            if not bmi_vals.empty:
                stats = bmi_vals.describe().round(2)
                st.write("**BMI Summary Statistics:**")
                st.dataframe(stats)
                fig_bmi, ax_bmi = plt.subplots()
                sns.histplot(bmi_vals, kde=True, ax=ax_bmi, color="lightgreen")
                ax_bmi.set_title("BMI Distribution")
                st.pyplot(fig_bmi)
            else:
                st.warning("⚠️ No valid BMI values.")
        else:
            st.warning("⚠️ BMI label not found.")

        # IWGDF GRADE ANALYSIS
        st.subheader("🦶 IWGDF Risk Grade Summary")
        grade_row = df[df[0].astype(str).str.strip().str.lower() == "grade de risque iwgdf"]
        if not grade_row.empty:
            grades = pd.to_numeric(df.loc[grade_row.index[0], 1:], errors='coerce').dropna().astype(int)
            grade_counts = pd.Series([0, 1, 2, 3]).map(lambda g: (grades == g).sum())
            risk_labels = ["Low", "Moderate", "High", "Very High"]
            fig_grade, ax_grade = plt.subplots()
            bars = ax_grade.bar(range(4), grade_counts, color="cornflowerblue")
            ax_grade.set_xticks(range(4))
            ax_grade.set_xticklabels(risk_labels)
            ax_grade.set_ylabel("Number of Patients")
            ax_grade.set_title("IWGDF Risk Grade Frequency")
            for i, b in enumerate(bars):
                ax_grade.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                            str(int(grade_counts[i])), ha='center', fontsize=10)
            st.pyplot(fig_grade)
        else:
            st.warning("⚠️ IWGDF grade row not found.")

        # DIABETES AGE BY TYPE
        st.subheader("🩸 Age of Diabetes by Type")
        label_age = "Age du diabète (années)"
        label_type = "Type de diabète 1/2/AT"
        age_row = df[df[0].astype(str).str.strip().str.lower() == label_age.lower()]
        type_row = df[df[0].astype(str).str.strip().str.lower() == label_type.lower()]
        if not age_row.empty and not type_row.empty:
            idx_age = age_row.index[0]
            idx_type = type_row.index[0]
            age_vals = pd.to_numeric(df.loc[idx_age, 1:], errors='coerce')
            type_vals = df.loc[idx_type, 1:].astype(str).str.strip().str.upper()
            valid = (type_vals != "NON DIAB") & type_vals.isin(["1", "2", "AT"])
            df_diabetes = pd.DataFrame({
                "AgeOnset": age_vals[valid],
                "Type": type_vals[valid].map({"1": "Type 1", "2": "Type 2", "AT": "Atypical"})
            }).dropna()
            if not df_diabetes.empty:
                summary = df_diabetes.groupby("Type")["AgeOnset"].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
                st.dataframe(summary)
            else:
                st.warning("⚠️ No valid diabetes type and age combinations found.")
        else:
            st.warning("⚠️ Labels for diabetes type or age not found.")

        # HbA1c by Insulin Use
        st.subheader("🧪 HbA1c by Insulin Use")
        hba1c_label = "Hba1c"
        insulin_label = "Insuline (Y/N)"
        hba1c_row = df[df[0].astype(str).str.strip().str.lower() == hba1c_label.lower()]
        insulin_row = df[df[0].astype(str).str.strip().str.lower() == insulin_label.lower()]
        if not hba1c_row.empty and not insulin_row.empty:
            hba1c_vals = pd.to_numeric(df.loc[hba1c_row.index[0], 1:], errors='coerce')
            insulin_vals = df.loc[insulin_row.index[0], 1:].astype(str).str.strip().str.upper()
            df_hba1c = pd.DataFrame({"Hba1c": hba1c_vals, "Insulin": insulin_vals})
            df_hba1c = df_hba1c[df_hba1c["Insulin"].isin(["Y", "N"])].dropna()
            if not df_hba1c.empty:
                df_hba1c["Insulin"] = df_hba1c["Insulin"].map({"Y": "Yes", "N": "No"})
                stats_hba1c = df_hba1c.groupby("Insulin")["Hba1c"].agg(["count", "mean", "std", "min", "median", "max"]).round(2)
                st.dataframe(stats_hba1c)
                fig_hba1c, ax_hba1c = plt.subplots()
                sns.boxplot(data=df_hba1c, x="Insulin", y="Hba1c", palette="Set2", ax=ax_hba1c)
                ax_hba1c.set_title("Hba1c Distribution by Insulin Use")
                st.pyplot(fig_hba1c)
            else:
                st.warning("⚠️ No valid Hba1c and insulin values found.")
        else:
            st.warning("⚠️ Hba1c or Insulin label not found.")

    # ================================
    # 📌 Stat Summary Extractor
    # ================================
    elif analysis_type == "GMM Clustering":
        st.subheader("🔍 GMM Clustering Based on Features Correlated with Grade")

        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import adjusted_rand_score
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # 🔹 Step 1: Locate risk grades row
        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found.")
            st.stop()

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        expected_length = len(risk_values)

        # 🔹 Step 2: Define features of interest (row indices)
        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L",
            96: "ROM Ankle R", 97: "ROM Ankle L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            126: "US Épaisseur ED SESA D", 127: "US Épaisseur ED HALLUX D", 128: "US Épaisseur ED TM5 D",
            130: "US Épaisseur ED SESA G", 131: "US Épaisseur ED HALLUX G", 132: "US Épaisseur ED TM5 G",
            134: "US Épaisseur Hypoderme SESA D", 135: "US Épaisseur Hypoderme HALLUX D",
            136: "US Épaisseur Hypoderme TM5 D", 138: "US Épaisseur Hypoderme SESA G",
            139: "US Épaisseur Hypoderme HALLUX G", 140: "US Épaisseur Hypoderme TM5 G",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
            154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L",
            19: "BMI", 24: "AOMI", 59: "Michigan Score", 72: "Charcot Sanders", 75: "ATCD Charcot Aigue D",
            76: "ATCD Charcot Aigue G", 158: "Temperature Hallux D", 159: "Temperature 5th Toe D",
            160: "Temperature Plantar Arch D", 161: "Temperature Lateral Sole D", 162: "Temperature Forefoot D",
            163: "Temperature Heel D", 164: "Temperature Hallux G", 165: "Temperature 5th Toe G",
            166: "Temperature Plantar Arch G", 167: "Temperature Lateral Sole G", 168: "Temperature Forefoot G",
            169: "Temperature Heel G", 170: "Temperature Hand Mean D", 171: "Temperature Hand Mean G",
        }

        # 🔹 Step 3: Extract features + Impute missing
        feature_dict = {}
        invalid_rows = []

        for idx, name in target_rows.items():
            try:
                values = pd.to_numeric(df.iloc[idx, 1:], errors='coerce').values
                values = values[:expected_length]  # truncate to expected_length (20)
                if np.count_nonzero(~np.isnan(values)) < expected_length:
                    # Impute missing values with mean
                    mean_val = np.nanmean(values)
                    if np.isnan(mean_val):
                        invalid_rows.append((idx, name, "All values are NaN"))
                        continue
                    imputed = np.where(np.isnan(values), mean_val, values)
                    values = imputed
                if len(values) != expected_length:
                    invalid_rows.append((idx, name, f"Length after imputation: {len(values)}"))
                    continue
                feature_dict[name] = values
            except Exception as e:
                invalid_rows.append((idx, name, str(e)))

        if not feature_dict:
            st.error("❌ No valid features extracted. Check the data format.")
            st.write("🛠️ Debug Info - Invalid/Mismatched rows:")
            st.write(invalid_rows)
            st.stop()

        # Ensure all feature arrays are same length
        lengths = [len(v) for v in feature_dict.values()]
        if len(set(lengths)) != 1:
            st.error("❌ Features have unequal lengths. Cannot construct DataFrame.")
            st.write("🛠️ Feature lengths:", lengths)
            st.stop()

        df_features = pd.DataFrame(feature_dict)
        df_features["Grade"] = risk_values.values[:expected_length]
        df_features["Group"] = df_features["Grade"].apply(lambda x: "A (Grades 0-1)" if x in [0, 1] else "B (Grades 2-3)")

        # 🔹 Step 4: Correlation with Grade
        corr_with_grade = df_features.drop(columns=["Group"]).corr()["Grade"].drop("Grade")
        top_features = corr_with_grade.abs().sort_values(ascending=False).head(6).index.tolist()

        if not top_features:
            st.error("❌ No top features correlated with grade.")
            st.stop()

        st.write("📌 Using top features:", ", ".join(top_features))

        # 🔹 Step 5: GMM Clustering
        X = df_features[top_features]
        y_true = df_features["Grade"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gmm = GaussianMixture(n_components=4, random_state=42)
        clusters = gmm.fit_predict(X_scaled)
        df_features["GMM_Cluster"] = clusters

        # 🔹 Step 6: Contingency Table & Heatmap
        cluster_vs_grade = pd.crosstab(df_features["Grade"], df_features["GMM_Cluster"])
        st.write("### 🔁 Cluster vs True Grade (Contingency Table)")
        st.dataframe(cluster_vs_grade)

        fig_heatmap, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cluster_vs_grade, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("GMM Clustering vs Grade")
        ax.set_xlabel("GMM Cluster")
        ax.set_ylabel("True Grade")
        st.pyplot(fig_heatmap)



        st.subheader("Interpretation of Contingency Table & Heatmap:")
        st.markdown("""
        - This heatmap visually represents the `pd.crosstab` contingency table, showing the counts of data points where a specific 'True Grade' corresponds to a specific 'GMM Cluster'.
        - **Ideal Scenario:** In a perfect clustering, each 'True Grade' row would have a single high value in one 'GMM Cluster' column and zeros elsewhere, indicating a one-to-one mapping.
        - **Observations from this Heatmap:**
            - **True Grade 0 (Row 0):** Largely maps to **GMM Cluster 1** (high count), with very few (or none) in other clusters. This indicates a good association.
            - **True Grade 1 (Row 1):** Appears to be predominantly assigned to **GMM Cluster 2**.
            - **True Grade 2 (Row 2):** Also predominantly assigned to **GMM Cluster 2**. This suggests that GMM Cluster 2 might be merging True Grade 1 and True Grade 2, indicating that these two grades are not well-separated by the GMM, or they share similar characteristics in the feature space.
            - **True Grade 3 (Row 3):** Is distributed across multiple GMM Clusters (e.g., GMM Cluster 0, GMM Cluster 2, GMM Cluster 3). This indicates that True Grade 3 is not cleanly separated into a single GMM cluster, suggesting a more complex or dispersed representation in the feature space for this grade.
        - **In summary:** The heatmap provides a clear visual breakdown of how well the GMM clusters align with the actual grade labels. We can identify which true grades are well-captured by specific clusters and which are mixed or spread out.
        """)
        # 🔹 Step 7: Adjusted Rand Index (ARI)
        ari = adjusted_rand_score(y_true, clusters)
        st.metric("🧮 Adjusted Rand Index (ARI)", f"{ari:.3f}")


        st.subheader("Interpretation of Adjusted Rand Index (ARI):")
        st.markdown(f"""
        - The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings. In this case, it compares the GMM clustering with the 'True Grade' labels.
        - **Range:** The ARI score ranges from -1 to 1.
            - An ARI of **1.0** indicates perfect agreement between the clustering and the true labels (i.e., the clusters exactly match the true grades).
            - An ARI of **0.0** indicates that the clustering is random, as good as chance.
            - A negative ARI indicates that the clustering is worse than random.
        - **Current ARI ({ari:.3f}):** Based on this value, we can quantitatively assess the quality of our GMM clustering.
            - If the value is close to 1, the GMM has done a good job of identifying the inherent grade structure.
            - If it's closer to 0, the clustering is not significantly better than random assignment.
            - Our current ARI indicates a [**insert your specific interpretation based on the value, e.g., 'moderate to good agreement', 'weak agreement', etc.**]. For example, if it's 0.6, you might say "a moderate to good agreement between the GMM clusters and the true grades."
        """)

        # 🔹 Step 8: PCA Visualization
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["GMM_Cluster"] = clusters
        pca_df["True Grade"] = y_true.values

        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="GMM_Cluster", style="True Grade", palette="deep", ax=ax_pca)
        ax_pca.set_title("PCA View: GMM Clusters vs True Grades")
        st.pyplot(fig_pca)


        st.subheader("Interpretation of PCA Visualization:")
        st.markdown("""
        - This scatter plot shows the data points projected onto the first two Principal Components (PC1 and PC2), which capture the most variance in the dataset.
        - **Color (`hue='GMM_Cluster'`)**: Each data point is colored according to the GMM cluster it was assigned to. This shows the spatial grouping identified by the GMM.
        - **Style (`style='True Grade'`)**: Each data point also has a unique marker style based on its actual 'True Grade'. This allows for a direct visual comparison between the GMM's clusters and the true underlying grades.
        - **Observations from the plot:**
            - **Spatial Separation of GMM Clusters:** We can observe how well the GMM algorithm has separated the data points into distinct regions in the 2D PCA space. Ideally, points of the same color (GMM cluster) would be tightly grouped.
            - **Alignment with True Grades:** By looking at both color and style, we can see:
                - **Well-separated Grades:** If a GMM cluster (a specific color) predominantly contains points of a single true grade style, it indicates good separation. For example, if **GMM Cluster 1 (orange circles)** primarily aligns with **True Grade 0 (black dots)**, it's a good match.
                - **Mixed Grades:** If a GMM cluster contains points with various true grade styles, it suggests that the GMM is mixing those grades. For example, if **GMM Cluster 2 (green squares)** shows both **True Grade 1 (black x)** and **True Grade 2 (black square)**, it confirms the merging observed in the heatmap.
                - **Spread Grades:** If points of a single true grade style are spread across multiple GMM clusters, it suggests that the GMM couldn't form a single, cohesive cluster for that grade. This is evident if **True Grade 3 (plus signs)** are found within multiple GMM clusters (e.g., blue, red).
        - **In summary:** The PCA plot provides a valuable visual complement to the quantitative metrics and contingency table. It helps us understand *why* the ARI score is what it is, by showing the geometric relationships between the clustered data and their true labels.
        """)
        
    # ================================
    # 📌 Stat Summary Extractor
    # ================================
    elif analysis_type == "Descriptive Analysis":
        st.header("📊Descriptive Analysis")
        normal_params = []
        non_normal_params = []
        summary_data = []
        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R",
            97: "ROM Ankle L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
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

                # Plot boxplot with mean line
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
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R",
            97: "ROM Ankle L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
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
    # 📌 Hallux/SESA/TM5 – L/R Comparison by Parameter Type
    # ================================
    elif analysis_type == "Hallux/SESA/TM5 – L/R Comparison by Parameter Type":
        st.subheader("📊 Comparison of Left vs Right by Anatomical Zone and Parameter Type")

        ep_derm_rows = {
            126: "Epiderm+Derm SESA D", 127: "Epiderm+Derm HALLUX D", 128: "Epiderm+Derm TM5 D",
            130: "Epiderm+Derm SESA G", 131: "Epiderm+Derm HALLUX G", 132: "Epiderm+Derm TM5 G"
        }

        hypo_rows = {
            134: "Hypoderm SESA D", 135: "Hypoderm HALLUX D", 136: "Hypoderm TM5 D",
            138: "Hypoderm SESA G", 139: "Hypoderm HALLUX G", 140: "Hypoderm TM5 G"
        }

        def plot_bar_parameter(data_dict, title, ylabel):
            locations = ["HALLUX", "SESA", "TM5"]
            sides = ["D", "G"]
            mapped = {"D": "Right", "G": "Left"}

            plot_data = []
            for loc in locations:
                for side in sides:
                    for key, label in data_dict.items():
                        if loc in label and side in label:
                            vals = df.loc[key, 1:].astype(float)
                            mean = vals.mean()
                            std = vals.std()
                            plot_data.append({
                                "Location": loc,
                                "Side": mapped[side],
                                "Mean": mean,
                                "STD": std
                            })

            plot_df = pd.DataFrame(plot_data)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x="Location", y="Mean", hue="Side", ax=ax, palette="coolwarm", capsize=0.1)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            st.pyplot(fig)

        plot_bar_parameter(ep_derm_rows, "Combined Epidermis + Dermis Thickness", "Thickness (mm)")
        plot_bar_parameter(hypo_rows, "Hypodermis Thickness", "Thickness (mm)")

        # Define your anatomical zones
        zones = ["HALLUX", "SESA", "TM5"]
        sides = {"R": "Right", "L": "Left"}

        # Function to group parameters by type (based on label keyword)
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

        # Define the categories you want to visualize
        parameter_types = ["Avg Pressure", "Stiffness", "US Épaisseur ED", "US Épaisseur Hypoderme", "Total Tissue Thickness", "ROC"]

        # Group row indices by type
        grouped_params = group_parameters_by_type(target_rows, parameter_types)

        # Generic plotter
        def plot_combined_bar(data_dict, title, ylabel):
            plot_data = []
            for key, label in data_dict.items():
                for zone in zones:
                    for side_code, side_label in sides.items():
                        if zone in label and side_label in label:
                            try:
                                vals = df.loc[key, 1:].astype(float)
                                plot_data.append({
                                    "Zone": zone,
                                    "Side": side_label,
                                    "Mean": vals.mean(),
                                    "STD": vals.std()
                                })
                            except:
                                pass

            plot_df = pd.DataFrame(plot_data)

            if plot_df.empty:
                st.warning(f"No valid data found for {title}")
                return

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x="Zone", y="Mean", hue="Side", ax=ax,
                        palette="coolwarm", capsize=0.1, errwidth=1)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            st.pyplot(fig)

        # Plot each type
        for ptype, row_dict in grouped_params.items():
            plot_combined_bar(row_dict, f"{ptype} – Left vs Right Comparison", "Mean Value")
              
    # ================================
    # 📌 Comparison of Left and Right Foot Parameters
    # ================================
    elif analysis_type == "Comparison of Left and Right Foot Parameters":

        st.header("🦶 Comparison of Left and Right Foot Parameters with Plots")

        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score(ok=13, risque=0)", 72: "Michigan Score2(ok=13, risque=0)", 
            75: "Medical history of acute Charcot R",76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R",
            97: "ROM Ankle L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
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

        # Collect all comparison results
        comparison_results = []

        for label_r, label_l, idx_r, idx_l in paired_parameters:
            values_r = pd.to_numeric(df.iloc[idx_r, 1:], errors='coerce').dropna()
            values_l = pd.to_numeric(df.iloc[idx_l, 1:], errors='coerce').dropna()

            common_len = min(len(values_r), len(values_l))
            values_r = values_r.iloc[:common_len]
            values_l = values_l.iloc[:common_len]

            # 🔒 Check for minimum data length
            if common_len < 3:
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

            # Plotting as before
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
        target_rows = {
            6: "Date of Birth", 16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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

        # 🔍 Step 2: Locate the row that contains IWGDF risk grades
        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found in the Excel sheet.")
            st.stop()

        # 🔢 Extract the actual values from the row found above
        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        patient_ids = risk_values.index

        # 🧠 Step 3: Extract selected features for each patient
        selected_data = df.loc[target_rows.keys(), 1:1 + len(risk_values) - 1]
        selected_data.index = [target_rows[i] for i in selected_data.index]  # Convert index to labels
        features_df = selected_data.T.apply(pd.to_numeric, errors='coerce')  # Transpose and convert to numeric
        features_df = features_df.loc[patient_ids]  # Align rows to patient order

        # ⚠️ Handle missing data
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(features_df.values)

        # 🤖 Step 4: Apply KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_imputed)

        # 🧾 Build results DataFrame with grade and cluster for each patient
        results = pd.DataFrame({
            "Patient": patient_ids,
            "IWGDF_Grade": risk_values.loc[patient_ids].values,
            "Cluster": clusters
        }, index=patient_ids)

        # 🧪 Step 5: Evaluate how well clusters match IWGDF grades
        st.markdown("### 👮️ Clustering vs. IWGDF Grade")
        contingency = pd.crosstab(results["Cluster"], results["IWGDF_Grade"])
        st.dataframe(contingency)
        ari = adjusted_rand_score(results["IWGDF_Grade"], results["Cluster"])
        st.markdown(f"**Adjusted Rand Index:** {ari:.3f}")

        # 📢 Interpretation of clustering validity
        if ari < 0.2:
            st.info("🧠 The clustering does not align well with the IWGDF grading — it groups patients differently.")
        elif ari < 0.5:
            st.info("🧠 Moderate alignment between clusters and IWGDF grades — possible overlap or partial structure.")
        else:
            st.success("🧠 Clustering agrees well with IWGDF grades — clustering reflects the known risk structure.")

        # 🎨 Step 6: Plot clustering colored by group
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

        st.markdown("ℹ️ **Chart Interpretation**")
        st.info("""
        - Each dot represents a patient.
        """)

        # ============================
        # 📊 Function to compare groups
        # ============================
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


        # 📊 Step 7: Compare risk grade transitions
        st.markdown("### 🔍 Parameter Transitions Between Risk Grades")
        g0 = results[results["IWGDF_Grade"] == 0].index
        g12 = results[results["IWGDF_Grade"].isin([1, 2])].index
        df_0_12 = compare_groups_safely("Grade 0 → Grade 1 or 2", g0, g12, features_df)

        g2 = results[results["IWGDF_Grade"] == 2].index
        g3 = results[results["IWGDF_Grade"] == 3].index
        df_2_3 = compare_groups_safely("Grade 2 → Grade 3", g2, g3, features_df)

        # 📊 Step 8: Compare clusters directly
        st.markdown("### 🔍 Parameter Differences Between Clusters")

        cluster0_idx = results[results["IWGDF_Grade"].isin([0, 1])].index
        cluster1_idx = results[results["IWGDF_Grade"].isin([2, 3])].index

        df_cluster_comparison = compare_groups_safely(
            "Cluster 0 (Grades 0–1) vs Cluster 1 (Grades 2–3)",
            cluster0_idx,
            cluster1_idx,
            features_df
        )


        # 💾 Step 9: Export all analysis results to Excel
        output_file = "IWGDF_transition_results.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            results.to_excel(writer, sheet_name="Clustering", index=False)
            contingency.to_excel(writer, sheet_name="Contingency")
            df_0_12.to_excel(writer, sheet_name="0_to_1or2", index=False)
            df_2_3.to_excel(writer, sheet_name="2_to_3", index=False)

        # 💡 Provide download button
        with open(output_file, "rb") as f:
            st.download_button("📅 Download Transition Results", f, file_name=output_file)



    # ================================
    # Clustering ignoring IWGDF Grade
    # ================================
    elif analysis_type == "Clustering ignoring IWGDF Grade":
        st.header("Unsupervised Clustering ignoring IWGDF Grade")

        # Transpose: features as rows, patients as columns
        data = df_combined.drop(columns=["Grade", "Group"], errors="ignore").T

        # Drop features (rows) with >50% missing values (i.e., not enough patient data)
        threshold = data.shape[1] * 0.5
        data = data.loc[data.isnull().sum(axis=1) < threshold]

        # Impute missing values with row-wise median
        data = data.apply(lambda row: row.fillna(row.median()), axis=1)

        # Keep only features (rows) with at least 20 valid patient values
        data = data.loc[data.count(axis=1) >= 20]

        # Transpose back for clustering (patients = rows, features = columns)
        features = data.T

        st.write(f"Remaining features after cleaning ({features.shape[1]} features):")
        st.write(list(features.columns))

        if features.shape[1] == 0 or features.shape[0] < 2:
            st.error("Not enough valid data for clustering after cleaning.")
        else:
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            # Agglomerative Clustering
            k = st.slider("Select number of clusters", 2, 6, 3)
            linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])

            # 'ward' requires Euclidean distances
            if linkage_method == "ward" and X_scaled.shape[1] < 2:
                st.warning("Ward linkage requires at least 2 features. Please adjust filtering.")
            else:
                clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                cluster_labels = clustering.fit_predict(X_scaled)

                df_combined["Cluster_without_Grade"] = cluster_labels

                st.write("Cluster assignments (first 10 patients):")
                st.dataframe(df_combined[["Cluster_without_Grade", "Grade"]].head(10))

                # PCA for 2D visualization
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
                df_combined["PCA1"] = components[:, 0]
                df_combined["PCA2"] = components[:, 1]

                fig, ax = plt.subplots()
                sns.scatterplot(data=df_combined, x="PCA1", y="PCA2", hue="Cluster_without_Grade", palette="Set2", ax=ax)
                ax.set_title(f"PCA plot colored by Agglomerative Clustering ({linkage_method})")
                st.pyplot(fig)

    # ================================
    # Clustering with IWGDF Grade Groups
    # ================================
    elif analysis_type == "Clustering with IWGDF Grade Groups":
        st.header("Clustering with IWGDF Grade Groups (Agglomerative Clustering)")

        risk_row = df.iloc[16]
        risk_label = risk_row[0]
        if str(risk_label).strip().lower() != "grade de risque iwgdf":
            st.error("Row 16 does not contain 'Grade de risque IWGDF' label.")
            st.stop()
            
        risk_values = pd.to_numeric(risk_row[1:], errors='coerce').dropna().astype(int)
        patient_ids = risk_values.index.tolist()

        selected_features = df.drop(index=["Grade de risque IWGDF"], errors='ignore')
        selected_features = selected_features.loc[:, patient_ids]

        features_df = selected_features.T.apply(pd.to_numeric, errors='coerce')  # ترنسپوز: ردیف‌ها بیماران

        features_df["Grade de risque IWGDF"] = risk_values.reindex(features_df.index)

        df_A = features_df[features_df["Grade de risque IWGDF"].isin([0, 1])]
        df_B = features_df[features_df["Grade de risque IWGDF"].isin([2, 3])]

        def run_clustering(df_group, group_name):
            st.subheader(f"Group {group_name} (n={df_group.shape[0]})")

            if df_group.shape[0] < 2:
                st.warning("Not enough patients in this group.")
                return None 

            data = df_group.drop(columns=["Grade", "Group", "Grade de risque IWGDF"], errors="ignore").T
            
            # Drop features (rows) with >50% missing values (i.e., not enough patient data)
            threshold = data.shape[1] * 0.5
            
            data = data.loc[data.isnull().sum(axis=1) < threshold]
            
            # Impute missing values with row-wise median
            data = data.apply(lambda row: row.fillna(row.median()), axis=1)
            
            # Keep only features (rows) with at least 20 valid patient values
            data = data.loc[data.count(axis=1) >= 20]

            features = data.T

            st.write(f"Remaining features for clustering ({features.shape[1]} features):")
            st.write(list(features.columns))

            if features.shape[1] == 0:
                st.error("No usable features after cleaning.")
                return None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            k = st.slider(f"Number of clusters for Group {group_name}", 2, 6, 3, key=f"k_{group_name}")
            linkage_method = st.selectbox(f"Linkage method for Group {group_name}",
                                        ["ward", "complete", "average", "single"], key=f"linkage_{group_name}")

            if linkage_method == "ward" and X_scaled.shape[1] < 2:
                st.warning("Ward linkage requires at least 2 features.")
                return None

            clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
            cluster_labels = clustering.fit_predict(X_scaled)

            df_group = df_group.copy()
            df_group["Cluster"] = cluster_labels

            st.write(df_group[["Cluster", "Grade de risque IWGDF"]].head(10))

            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            df_group["PCA1"] = components[:, 0]
            df_group["PCA2"] = components[:, 1]

            fig, ax = plt.subplots()
            sns.scatterplot(data=df_group, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", ax=ax)
            ax.set_title(f"PCA projection – Group {group_name}")
            st.pyplot(fig)

            return df_group

        df_A_result = run_clustering(df_A, "A (Grade 0–1)")
        df_B_result = run_clustering(df_B, "B (Grade 2–3)")

        # Combine results for comparison
        df_combined["Cluster_with_Grade"] = np.nan

        if df_A_result is not None:
            df_combined.loc[df_A_result.index, "Cluster_with_Grade"] = df_A_result["Cluster"]
        if df_B_result is not None:
            df_combined.loc[df_B_result.index, "Cluster_with_Grade"] = df_B_result["Cluster"]

        if "Cluster_without_Grade" in df_combined.columns and df_combined["Cluster_with_Grade"].notna().sum() > 1:
            common = df_combined.dropna(subset=["Cluster_without_Grade", "Cluster_with_Grade"])
            if len(common) > 1:
                ari = adjusted_rand_score(common["Cluster_without_Grade"], common["Cluster_with_Grade"])
                st.write(f"### Adjusted Rand Index (ARI) between clusterings:")
                st.write(f"ARI = {ari:.3f}")

                contingency = pd.crosstab(common["Cluster_without_Grade"], common["Cluster_with_Grade"])
                st.write("### Contingency Table (Cluster_without_Grade vs Cluster_with_Grade):")
                st.dataframe(contingency)
            else:
                st.warning("Not enough patients with both cluster labels to compare.")
        else:
            st.warning("Cluster labels missing for comparison.")

    # ================================
    # Grade  & Clustering
    # ================================
    
    elif analysis_type == "Grade  & Clustering":
        st.subheader("🔎 Exploratory Clustering (without IWGDF grade)")

        target_rows = {
            6: "Date of Birth", 16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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

        exclude_keys = [6, 16]
        rows_to_use = [k for k in target_rows.keys() if k not in exclude_keys]

        try:
            st.title("IWGDF Risk Grade Clustering & Analysis")

            # Select feature rows and rename indices
            df_selected = df.loc[rows_to_use, :].copy()
            df_selected.index = [target_rows[i] for i in rows_to_use]

            # Transpose so patients = rows, features = columns
            data = df_selected.T
            data = data.apply(pd.to_numeric, errors='coerce')

            # Extract Grade IWGDF row (index 16) if present for comparison
            grade_series = None
            if 16 in df.index:
                grade_series = pd.to_numeric(df.loc[16, :], errors='coerce')
                grade_series.name = "Grade IWGDF"

            # Filter out features with all NaNs
            valid_columns = data.columns[data.notna().any()]
            data_valid = data[valid_columns]

            st.write(f"Number of variables used: {len(valid_columns)}")

            # Impute missing values with mean
            imputer = SimpleImputer(strategy='mean')
            data_imputed_array = imputer.fit_transform(data_valid)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data_imputed_array)

            if X_scaled.shape[0] < 10:
                st.warning("Not enough patients for clustering (minimum 10 required).")
                st.stop()

            # Find best number of clusters by silhouette score
            silhouette_scores = {}
            for k in range(2, 7):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                silhouette_scores[k] = silhouette_score(X_scaled, labels)

            best_k = max(silhouette_scores, key=silhouette_scores.get)
            st.success(f"Optimal number of clusters (Silhouette score): **{best_k}**")

            # Final clustering
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            # Create DataFrame with imputed data + cluster labels
            data_imputed = pd.DataFrame(data_imputed_array, columns=valid_columns, index=data_valid.index)
            data_imputed["Cluster"] = clusters

            # Add Grade IWGDF if available (aligned by patient index)
            if grade_series is not None:
                grade_aligned = grade_series.reindex(data_imputed.index)
                data_imputed["Grade IWGDF"] = grade_aligned

            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            data_imputed["PC1"] = X_pca[:, 0]
            data_imputed["PC2"] = X_pca[:, 1]

            # Show cluster mean feature values
            st.write("### 📊 Mean Feature Values by Cluster")
            cluster_summary = data_imputed.groupby("Cluster").mean(numeric_only=True)
            st.dataframe(cluster_summary.style.format("{:.2f}"))

            # Show differences and significance tests between consecutive clusters
            st.write("### 🔍 Differences & Statistical Tests Between Consecutive Clusters")
            sorted_clusters = sorted(data_imputed["Cluster"].unique())

            for i in range(len(sorted_clusters) - 1):
                c1, c2 = sorted_clusters[i], sorted_clusters[i + 1]
                group1 = data_imputed[data_imputed["Cluster"] == c1]
                group2 = data_imputed[data_imputed["Cluster"] == c2]

                delta = cluster_summary.loc[c2] - cluster_summary.loc[c1]
                delta.name = f"{c1} → {c2}"

                p_values = []
                for var in cluster_summary.columns:
                    vals1 = group1[var].dropna()
                    vals2 = group2[var].dropna()
                    if len(vals1) < 3 or len(vals2) < 3 or np.all(vals1 == vals1.iloc[0]) or np.all(vals2 == vals2.iloc[0]):
                        p = np.nan
                    else:
                        try:
                            _, p = ttest_ind(vals1, vals2, equal_var=False)
                            if np.isnan(p) or np.isinf(p):
                                raise ValueError
                        except:
                            try:
                                _, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
                            except:
                                p = np.nan
                    p_values.append(p)

                stats_df = pd.DataFrame({
                    "Mean Difference": delta,
                    "p-value": p_values
                }, index=cluster_summary.columns)

                stats_df = stats_df.sort_values(by="Mean Difference", key=lambda x: abs(x), ascending=False)

                st.write(f"#### Cluster {c1} → {c2}")
                st.dataframe(stats_df.style.format({"Mean Difference": "{:+.2f}", "p-value": "{:.2e}"}))

                st.write("**Interpretation of top 5 differentiating parameters:**")
                for param, row in stats_df.head(5).iterrows():
                    diff_val = row["Mean Difference"]
                    p_val = row["p-value"]
                    signif = "significant" if (not np.isnan(p_val) and p_val < 0.05) else "not significant"
                    direction = "increased" if diff_val > 0 else "decreased"
                    st.markdown(f"- **{param}** : {direction} by {diff_val:+.2f} (p = {p_val:.2e}, {signif})")

            # PCA scatter plot of clusters
            st.write("### 🧭 PCA Projection of Clusters")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data_imputed, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=70, ax=ax)
            ax.set_title("PCA Projection of Patients by Cluster")
            st.pyplot(fig)

            # Cluster vs IWGDF grade comparison
            if grade_series is not None:
                st.write("### 🧮 Cluster vs IWGDF grade comparison")

                known = data_imputed.copy()
                known["Grade IWGDF"] = pd.to_numeric(known["Grade IWGDF"], errors='coerce')
                known = known.dropna(subset=["Grade IWGDF"])

                if known.empty:
                    st.warning("No patients with known numeric IWGDF grades for comparison.")
                else:
                    # Categorize grades finely
                    def categorize_grade_fine(x):
                        return f"Grade {int(x)}"

                    known["IWGDF_Group"] = known["Grade IWGDF"].apply(categorize_grade_fine)

                    contingency = pd.crosstab(known["Cluster"], known["IWGDF_Group"])
                    st.dataframe(contingency)

                    # Force run tests even if shape < 2x2
                    try:
                        chi2, p_chi, dof, expected = chi2_contingency(contingency, correction=False)
                        st.markdown(f"**Chi-square test:** χ² = {chi2:.2f}, p = {p_chi:.4f}")

                        n = contingency.values.sum()
                        phi2 = chi2 / n if n > 0 else 0
                        r, k = contingency.shape
                        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1)) if r > 1 and k > 1 else np.nan
                        st.markdown(f"**Cramér’s V:** {cramers_v:.3f}" if not np.isnan(cramers_v) else "**Cramér’s V:** Not defined (single row or column)")

                        if contingency.shape == (2, 2):
                            oddsratio, p_fisher = fisher_exact(contingency)
                            st.markdown(f"**Fisher's Exact test p-value:** {p_fisher:.4f}")
                        else:
                            st.info("Fisher's Exact test only for 2x2 tables.")
                    except Exception as e:
                        st.warning(f"⚠️ Statistical tests could not be performed reliably: {e}")

                    st.write("### Interpretation:")
                    st.markdown(
                        """
                        - **Note:** Statistical test results may be unreliable with very small or unbalanced groups.
                        - Proceed with caution interpreting p-values and effect sizes.
                        """
                    )


            # Export results button
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                data_imputed.to_excel(writer, sheet_name="Clustered Data")
                if grade_series is not None and not known.empty:
                    contingency.to_excel(writer, sheet_name="Contingency Table")
            buffer.seek(0)
            st.download_button("📥 Download full results (Excel)", buffer.getvalue(), "iwgdf_clustering_analysis.xlsx")

        except Exception as e:
            st.error(f"⚠️ Error occurred: {e}")
            

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
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R",
            97: "ROM Ankle L", 108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
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




    # =====================================================
    # Intra-Group Comparison
    # =====================================================
    elif analysis_type == "Intra-Group Comparison":
        st.header("📈 Intra-Group Comparison")

        selected_group = st.radio("Choose a group for intra-group comparison:", ["A (Grades 0-1)", "B (Grades 2-3)"])
        g1, g2 = (0, 1) if selected_group == "A (Grades 0-1)" else (2, 3)

        sub1 = df_combined[df_combined["Grade"] == g1]
        sub2 = df_combined[df_combined["Grade"] == g2]

        st.markdown(f"- **Group**: {selected_group} — comparing Grade {g1} vs Grade {g2}")
        st.markdown(f"- Grade {g1}: {len(sub1)} patients | Grade {g2}: {len(sub2)} patients")

        for col in df_combined.columns[:-2]:
            values1 = pd.to_numeric(sub1[col], errors='coerce').dropna()
            values2 = pd.to_numeric(sub2[col], errors='coerce').dropna()

            st.write(f"### 🔹 {col}")
            st.write(f"Grade {g1} - count: {len(values1)}, Grade {g2} - count: {len(values2)}")

            # Shapiro normality check
            if len(values1) >= 3:
                p_shapiro1 = shapiro(values1)[1]
                st.write(f"Shapiro Grade {g1}: p = {p_shapiro1:.4f}")
            else:
                st.write(f"Shapiro Grade {g1}: ❌ Not enough data (min 3)")

            if len(values2) >= 3:
                p_shapiro2 = shapiro(values2)[1]
                st.write(f"Shapiro Grade {g2}: p = {p_shapiro2:.4f}")
            else:
                st.write(f"Shapiro Grade {g2}: ❌ Not enough data (min 3)")

            # Always do Mann-Whitney
            stat, pval = mannwhitneyu(values1, values2, alternative='two-sided')
            st.write(f"**Mann–Whitney U test**: p = `{pval:.4f}`")
            st.divider()


            group_a = df_combined[df_combined["Group"] == "A (Grades 0-1)"]
            group_b = df_combined[df_combined["Group"] == "B (Grades 2-3)"]

            for col in df_combined.columns[:-2]:
                values_a = pd.to_numeric(group_a[col], errors='coerce').dropna()
                values_b = pd.to_numeric(group_b[col], errors='coerce').dropna()

                if len(values_a) < 3 or len(values_b) < 3:
                    st.write(f"**{col}** – ❗ Not enough data (min 3 per group).")
                    continue

                is_norm_a = shapiro(values_a)[1] > 0.05
                is_norm_b = shapiro(values_b)[1] > 0.05

                if is_norm_a and is_norm_b:
                    stat, pval = ttest_ind(values_a, values_b, equal_var=False)
                    test_type = "Independent t-test"
                else:
                    stat, pval = mannwhitneyu(values_a, values_b, alternative='two-sided')
                    test_type = "Mann–Whitney U"

                st.markdown(f"**{col}** — {test_type}, p = `{pval:.4f}`")
                

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
            sd = np.std(diff)

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

        for group_name, rows in parameter_groups.items():
            st.markdown(f"### 📁 {group_name}")
            selected = df.loc[list(rows.keys()), 1:]
            selected.index = [rows[i] for i in rows.keys()]
            selected = selected.T.apply(pd.to_numeric, errors='coerce')

            zones = ["SESA", "HALLUX", "TM5"]
            for zone in zones:
                col_d = [col for col in selected.columns if zone in col and " D" in col]
                col_g = [col for col in selected.columns if zone in col and " G" in col]
                if col_d and col_g:
                    col_d = col_d[0]
                    col_g = col_g[0]
                    data_d = selected[col_d].dropna()
                    data_g = selected[col_g].dropna()
                    min_len = min(len(data_d), len(data_g))
                    fig = plot_bland_altman(
                        data_d[:min_len],
                        data_g[:min_len],
                        title=f"{zone} - Right (D) vs Left (G)",
                        unit=group_name.split("(")[-1].replace(")", "") if "(" in group_name else ""
                    )
                    st.pyplot(fig)

    # ================================
    # Bland-Altman Pooled Plots for all parameters
    # ================================
    elif analysis_type == "Bland-Altman Pooled Plots for all parameters":
        st.subheader("📊 Bland-Altman Pooled Plots for Parameters with HALLUX, SESA, TM5")

        from collections import defaultdict

        param_groups = defaultdict(lambda: {"SESA": {}, "HALLUX": {}, "TM5": {}})

        for key, label in target_rows.items():
            for zone in ["SESA", "HALLUX", "TM5"]:
                if zone in label:
                    side = "R" if "R" in label else "L" if "L" in label else None
                    if side:
                        param_base = label.replace(zone, "").replace(" R", "").replace(" L", "").strip()
                        param_groups[param_base][zone][side] = key
        complete_params = {
            param: zones for param, zones in param_groups.items()
            if all(len(zones[z]) == 2 for z in ["SESA", "HALLUX", "TM5"])
        }

        def plot_bland_altman(data1, data2, title="", unit=""):
            mean = (data1 + data2) / 2
            diff = data1 - data2
            md = np.mean(diff)
            sd = np.std(diff)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(mean, diff, alpha=0.6)
            ax.axhline(md, color='blue', linestyle='--', label='Mean')
            ax.axhline(md + 1.96 * sd, color='red', linestyle='--', label='+1.96 SD')
            ax.axhline(md - 1.96 * sd, color='red', linestyle='--', label='-1.96 SD')
            ax.set_title(f"Bland-Altman: {title}")
            ax.set_xlabel(f"Mean {unit}")
            ax.set_ylabel("Difference")
            ax.legend()
            ax.grid(True)
            return fig, md, sd

        table_data = []

        for param_name, zones in complete_params.items():
            st.markdown(f"### 🔍 {param_name}")
            pooled_r, pooled_l = [], []

            for zone in ["SESA", "HALLUX", "TM5"]:
                r_idx = zones[zone]["R"]
                l_idx = zones[zone]["L"]

                right_vals = df.loc[r_idx, 1:].astype(float)
                left_vals = df.loc[l_idx, 1:].astype(float)

                paired = [(r, l) for r, l in zip(right_vals, left_vals) if not (np.isnan(r) or np.isnan(l))]
                if len(paired) > 0:
                    r_clean, l_clean = zip(*paired)
                    pooled_r.extend(r_clean)
                    pooled_l.extend(l_clean)

            pooled_r, pooled_l = np.array(pooled_r), np.array(pooled_l)
            if len(pooled_r) == 0 or len(pooled_l) == 0:
                st.warning(f"No valid data for {param_name}")
                continue

            fig, md, sd = plot_bland_altman(pooled_r, pooled_l, title=f"{param_name} (Pooled Zones)", unit="")
            st.pyplot(fig)

            pooled_mean = np.mean((pooled_r + pooled_l) / 2)
            table_data.append((param_name, round(pooled_mean, 2), round(md, 2), round(1.96 * sd, 2)))

        df_table = pd.DataFrame(table_data, columns=["Parameter", "Mean", "Mean Difference", "±1.96 SD"])
        st.markdown("### 📋 Statistical Summary Table")
        st.dataframe(df_table)

   
   
   
    elif analysis_type == "Multivariate Group  (MANOVA)":
        st.header("📊 MANOVA: Global Comparison Across Grades")

        df_combined = df_combined.rename(columns=lambda x: str(x).replace(" ", "_").replace("-", "_"))

        # Select features for MANOVA
        features = st.multiselect("Select features for MANOVA:", df_combined.select_dtypes(include=[np.number]).columns.tolist(), default=df_combined.select_dtypes(include=[np.number]).columns[:5].tolist())
        group_col = "Group"

        if len(features) >= 2:
            from statsmodels.multivariate.manova import MANOVA

            # Prepare data
            df_manova = df_combined[[group_col] + features].dropna()
            df_manova[group_col] = df_manova[group_col].astype(str)

            # MANOVA
            formula = f"{' + '.join(features)} ~ {group_col}"
            maov = MANOVA.from_formula(formula, data=df_manova)
            st.text(maov.mv_test())
        else:
            st.warning("Please select at least 2 features for MANOVA.")
            
            
            
            

    # ================================
    # 📊 Multivariate & Predictive Analysis
    # ================================
    elif analysis_type == "Multivariate Group Comparison (MANOVA)":
        st.header("📊 MANOVA: Comparing Parameters Between IWGDF Risk Groups")

        target_rows = {
            6: "Date of Birth", 16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI", 24: "AOMI",
            35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
            59: "Michigan Score (ok=13, risk=0)", 72: "Michigan Score2 (ok=13, risk=0)",
            75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
            77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
            94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        patient_ids = pd.Series(range(1, len(risk_values) + 1), index=risk_values.index)

        selected_indices = list(target_rows.keys())
        selected_labels = [target_rows[i] for i in selected_indices]
        data = df.loc[selected_indices, risk_values.index + 1]
        data.index = selected_labels
        df_selected = data.T.apply(pd.to_numeric, errors='coerce')

        # Add Grade and Group columns
        df_selected["Grade"] = risk_values.values
        df_selected["Group"] = df_selected["Grade"].apply(lambda x: "A (0-1)" if x in [0, 1] else "B (2-3)")

        # Impute missing values with column means (excluding Group and Grade)
        cols_to_impute = df_selected.columns.difference(["Grade", "Group"])
        df_selected[cols_to_impute] = df_selected[cols_to_impute].apply(lambda col: col.fillna(col.mean()), axis=0)

        # Remove columns with no variance (constant or empty)
        df_selected = df_selected.loc[:, df_selected.nunique(dropna=True) > 1]

        if "Group" not in df_selected.columns or df_selected["Group"].nunique() < 2:
            st.warning("⚠️ Not enough valid groups for MANOVA.")
        else:
            # Clean column names for formula compatibility
            def safe_colname(col):
                return (
                    col.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("-", "_")
                    .replace("=", "")
                    .replace(",", "")
                    .replace(".", "")
                    .replace("°", "")
                    .replace("/", "_")
                )

            df_renamed = df_selected.rename(columns={col: safe_colname(col) for col in df_selected.columns})
            dependent_vars = [col for col in df_renamed.columns if col not in ["Group", "Grade"]]

            formula = f"{' + '.join(dependent_vars)} ~ Group"

            try:
                from statsmodels.multivariate.manova import MANOVA
                manova = MANOVA.from_formula(formula, data=df_renamed)
                st.subheader("📍 MANOVA Results (Group A vs B)")
                res = manova.mv_test()
                st.text(res)

                # -------- Interpretation --------
                st.markdown("### 📋 Interpretation:")
                wilks_p = None
                for test_name, test_res in res.results.items():
                    if "Wilks' lambda" in test_res:
                        wilks_p = test_res["Wilks' lambda"]["Pr > F"]
                        break
                if wilks_p is not None:
                    if wilks_p < 0.05:
                        st.success(f"The MANOVA indicates a significant difference between risk groups (p = {wilks_p:.4f}).")
                    else:
                        st.info(f"No significant difference between risk groups was found (p = {wilks_p:.4f}).")
                else:
                    st.warning("Could not find Wilks' lambda p-value for interpretation.")

                # -------- Boxplots --------
                import matplotlib.pyplot as plt
                import seaborn as sns

                st.subheader("📊 Boxplots of Parameters by Group")
                for var in dependent_vars:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x="Group", y=var, data=df_renamed, ax=ax, palette="Set2")
                    ax.set_title(f"Distribution of {var} by Group")
                    ax.set_xlabel("Risk Group")
                    ax.set_ylabel(var)
                    st.pyplot(fig)

                # -------- Violin plots --------
                st.subheader("🎻 Violin Plots of Parameters by Group")
                for var in dependent_vars:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.violinplot(x="Group", y=var, data=df_renamed, ax=ax, palette="Set3")
                    ax.set_title(f"Violin Plot of {var} by Group")
                    ax.set_xlabel("Risk Group")
                    ax.set_ylabel(var)
                    st.pyplot(fig)

                # Let user select variables for pairplot and heatmap
                selected_vars = st.multiselect(
                    "Select parameters for pairplot and correlation heatmap:",
                    options=dependent_vars,
                    default=dependent_vars[:6]  # default to first 6 variables
                )

                if selected_vars:
                    st.subheader("🔎 Pairplot of Selected Variables")
                    pairplot_fig = sns.pairplot(df_renamed, vars=selected_vars, hue="Group", palette="Set1")
                    st.pyplot(pairplot_fig)

                    st.subheader("📈 Correlation Heatmap of Selected Variables")
                    corr = df_renamed[selected_vars].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Please select at least one parameter to show the plots.")

                # -------- Pairplot --------
                vars_for_pairplot = dependent_vars[:]  # select first 6 vars (adjust as needed)
                st.subheader("🔎 Pairplot of Selected Variables")
                pairplot_fig = sns.pairplot(df_renamed, vars=vars_for_pairplot, hue="Group", palette="Set1")
                st.pyplot(pairplot_fig)

                # -------- Correlation Heatmap with threshold --------
                st.subheader("📈 Correlation Heatmap of Selected Variables (|corr| > 0.7)")

                corr = df_renamed[vars_for_pairplot].corr()

                # Mask correlations with abs value <= 0.7
                mask = corr.abs() <= 0.7
                corr_filtered = corr.mask(mask)

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_filtered, annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1,
                            center=0, linewidths=0.5, linecolor='gray', square=True, cbar_kws={"shrink": .8})

                st.pyplot(fig)

            except Exception as e:
                st.error("❌ Error in MANOVA analysis:")
                st.exception(e)

    # ================================
    # 📊 Multivariate & Predictive Analysis
    # ================================
    elif analysis_type == "Multiple Linear Regression":
        st.header("🔢 Multiple Linear Regression")

        df_T = df_numeric_reg.T

        target_options = {i: name for i, name in target_rows_reg.items()}
        target_index = st.selectbox("Select Target Variable", options=list(target_options.keys()), format_func=lambda x: target_options[x])

        y = df_T[target_index]
        predictor_indices = [i for i in target_rows_reg if i != target_index]
        X = df_T[predictor_indices]

        data = pd.concat([X, y], axis=1)

        data[predictor_indices] = data[predictor_indices].apply(pd.to_numeric, errors='coerce')
        data[target_index] = pd.to_numeric(data[target_index], errors='coerce')
        data_clean = data.dropna()

        # After cleaning:
        X_clean = data_clean[predictor_indices]
        y_clean = data_clean[target_index]

        # Check if we have enough data to proceed
        if X_clean.shape[0] == 0 or y_clean.shape[0] == 0:
            st.warning("No valid data available after cleaning. Please select different variables or check your data.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            model = LinearRegression()
            model.fit(X_scaled, y_clean)

            y_pred = model.predict(X_scaled)
            r2 = model.score(X_scaled, y_clean)

            st.subheader(f"R² Score: {r2:.3f}")
            st.write("**Intercept:**", model.intercept_)

            coef_df = pd.DataFrame({
                "Feature": [target_rows_reg[i] for i in predictor_indices],
                "Coefficient": model.coef_
            }).sort_values(by="Coefficient", key=abs, ascending=False)

            st.dataframe(coef_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="coolwarm")
            ax.axvline(0, color='gray', linestyle='--')
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(y_clean, y_pred, alpha=0.7)
            ax2.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'r--')
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.set_title("Actual vs Predicted")
            st.pyplot(fig2)
            
            
                
    elif analysis_type == "Exploratory PCA":
        st.header("🔍 Exploratory PCA: Visualizing Separation by IWGDF Grade")

        row_labels = df.iloc[:, 0]
        df_numeric = df.iloc[:, 1:]
        df_T = df_numeric.T

        try:
            # Get IWGDF grades from row 17 (index 16)
            risk_row = df.iloc[16, 1:]
            risk_grades = pd.to_numeric(risk_row, errors='coerce')

            # Define variables for PCA
            pca_rows = {
                16: "Grade IWGDF", 17: "Height (m)", 18: "Weight (kg)", 19: "BMI",
                35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L", 37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L",
                75: "Medical history of acute Charcot R", 76: "Medical history of acute Charcot L",
                77: "Chronic Charcot (R Sanders)", 78: "Chronic Charcot (L Sanders)",
                94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L", 96: "ROM Ankle R", 97: "ROM Ankle L",
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

            df_features = df_T[list(pca_rows.keys())].copy()
            df_features.columns = [pca_rows[i] for i in df_features.columns]
            df_features = df_features.apply(pd.to_numeric, errors='coerce')

            # Handle missing values
            max_missing = st.slider("Max allowed % of missing values per feature", 0, 100, 30)
            threshold = max_missing / 100
            df_features = df_features.loc[:, df_features.isna().mean() < threshold]
            df_clean = df_features.dropna()

            # Align risk grades (and drop those with NaN)
            risk_clean = risk_grades.loc[df_clean.index].dropna()
            df_clean = df_clean.loc[risk_clean.index]

            # Final check
            if df_clean.shape[0] == 0:
                st.error("❌ No valid data left after cleaning (too many NaNs). Check your file or lower the missing value threshold.")
                st.stop()

            # PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            explained = pca.explained_variance_ratio_ * 100

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=risk_clean.astype(int), cmap="coolwarm", edgecolor="k", alpha=0.8)
            ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
            ax.set_title("PCA: Patients Colored by IWGDF Grade")
            legend1 = ax.legend(*scatter.legend_elements(), title="Risk Grade")
            ax.add_artist(legend1)
            st.pyplot(fig)

            # Show metrics
            st.write(f"**Explained Variance:** PC1 = {explained[0]:.1f}%, PC2 = {explained[1]:.1f}%")
            st.write("📌 Features used:", df_clean.shape[1])
            st.write("📌 Patients after cleaning:", df_clean.shape[0])
            st.dataframe(df_clean)

        except Exception as e:
            st.error(f"⚠️ PCA failed: {e}")




# ================================
# 📎 File Not Uploaded Message
# ================================
else:
    st.info("Please upload an Excel file containing the 'DIAFOOT' sheet.")
