import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


# --- Assume df and risk_values are loaded here ---
df = pd.read_excel("_CR HDJ 13.xlsx", sheet_name="DIAFOOT", header=None)

# --- Retrieve IWGDF Risk Grades ---
label_risk = "Grade de risque IWGDF"
row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
if row_risk.empty:
    raise ValueError(f"Label '{label_risk}' not found.")
idx_risk = row_risk.index[0]
risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
patient_numbers = pd.Series(range(1, len(risk_values) + 1), index=risk_values.index)

# For demonstration, you need to prepare these variables beforehand.

# ==== DATA PREPARATION FUNCTION ====

@st.cache_data
def prepare_data(df, risk_values):
    # Extract ED thickness
    ed_data = df.iloc[126:134, 1:1+len(risk_values)]
    ed_data.index = [
        "ED SESA R (mm)",
        "ED HALLUX R (mm)",
        "ED TM5 R (mm)",
        "ED Other R (mm)",
        "ED SESA L (mm)",
        "ED HALLUX L (mm)",
        "ED TM5 L (mm)",
        "ED Other L (mm)"
    ]
    df_ed = ed_data.T.apply(pd.to_numeric, errors='coerce')

    # Extract hypodermis ultrasound
    hypoderm_data = df.iloc[135:143, 1:1+len(risk_values)]
    hypoderm_data.index = [
        "US Hypoderme SESA R (mm)",
        "US Hypoderme HALLUX R (mm)",
        "US Hypoderme TM5 R (mm)",
        "US Hypoderme Other R (mm)",
        "US Hypoderme SESA L (mm)",
        "US Hypoderme HALLUX L (mm)",
        "US Hypoderme TM5 L (mm)",
        "US Hypoderme Other L (mm)"
    ]
    df_hypo = hypoderm_data.T.apply(pd.to_numeric, errors='coerce')

    # Combine
    df_combined = pd.concat([df_ed, df_hypo], axis=1)
    df_combined = df_combined.dropna()

    # Add Grade and Group
    df_combined['Grade'] = risk_values.loc[df_combined.index].values
    df_combined['Group'] = df_combined['Grade'].apply(
        lambda x: 'A (Grades 0-1 😊👍)' if x in [0, 1] else 'B (Grades 2-3 ⚠️)'
    )
    return df_combined

# === MAIN APP ===

st.title("📊 Dashboard ED Thickness & Hypodermis Ultrasound Analysis")

uploaded_file = st.file_uploader("Upload Excel file with DIAFOOT sheet", type=["xlsx"])

if uploaded_file:
    # Load the full Excel file
    df = pd.read_excel(uploaded_file, sheet_name="DIAFOOT", header=None)

    # Get risk_values similarly to your method
    label_risk = "Grade de risque IWGDF"
    row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
    if row_risk.empty:
        st.error(f"Label '{label_risk}' not found in the file.")
    else:
        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        patient_indices = risk_values.index

        df_combined = prepare_data(df, risk_values)

        # Filter by Group
        group_filter = st.multiselect(
            "Select Risk Group(s):",
            options=df_combined['Group'].unique(),
            default=df_combined['Group'].unique()
        )
        df_filtered = df_combined[df_combined['Group'].isin(group_filter)]

        st.write(f"### Selected Data ({len(df_filtered)} patients):")
        st.dataframe(df_filtered)

        # --- T-test ---
        st.subheader("📊 T-test Results Between Groups")
        group_a = df_combined[df_combined['Group'].str.startswith('A')]
        group_b = df_combined[df_combined['Group'].str.startswith('B')]

        ttest_results = []
        for col in df_combined.columns[:-2]:
            t_stat, p_val = ttest_ind(group_a[col], group_b[col], equal_var=False, nan_policy='omit')
            ttest_results.append({"Variable": col, "p-value": p_val})
        ttest_df = pd.DataFrame(ttest_results).sort_values("p-value")

        st.table(ttest_df.style.format({"p-value": "{:.4f}"}))

        # --- Correlation matrix ---
        st.subheader("📈 Correlation Matrix")
        corr_matrix = df_filtered.drop(columns=['Group', 'Grade']).corr()
        fig_corr, ax_corr = plt.subplots(figsize=(14,10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # --- Linear Regression ---
        st.subheader("🔁 Linear Regression to Predict Grade")

        if len(df_filtered) < 2:
            st.warning("Not enough data to perform regression.")
        else:
            X = df_filtered.drop(columns=['Grade', 'Group'])
            y = df_filtered['Grade']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            coeff_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_
            }).sort_values(by="Coefficient", key=abs, ascending=False)

            st.dataframe(coeff_df.style.format({"Coefficient": "{:.4f}"}))

            fig_reg, ax_reg = plt.subplots(figsize=(12,6))
            sns.barplot(x="Coefficient", y="Feature", data=coeff_df, palette="cubehelix", ax=ax_reg)
            ax_reg.set_title("Feature Importance in Grade Prediction")
            st.pyplot(fig_reg)

else:
    st.info("Please upload an Excel file containing the DIAFOOT sheet.")

