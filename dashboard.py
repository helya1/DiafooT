import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="DIAFOOT Analysis Dashboard", layout="wide")
st.title("📊 DIAFOOT Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Excel file with 'DIAFOOT' sheet", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="DIAFOOT", header=None)

    analysis_type = st.sidebar.radio(
        "🧪 Choose Analysis Type:",
        ("📋 Stat Summary Extractor", "🧬 ED Thickness & Hypodermis Ultrasound Analysis")
    )

    if analysis_type == "📋 Stat Summary Extractor":
        target_rows = {
            17: "Taille (m)",
            18: "Poids (kg)",
            35: "MESI PRESSION GO D",
            36: "MESI PRESSION GO G",
            37: "IPS GO D",
            38: "IPS GO G",
            94: "AMPLI MTP1 D",
            95: "AMPLI MTP1 G",
            96: "AMPLI TALO CRURALE D",
            97: "AMPLI TALO CRURALE G",
            108: "Pression MOYENNE DES MAX SESA D",
            109: "Pression MOYENNE DES MAX HALLUX D",
            110: "Pression MOYENNE DES MAX  TM5 D",
            113: "Pression MOYENNE DES MAX  SESA G",
            114: "Pression MOYENNE DES MAX  HALLUX G",
            115: "Pression MOYENNE DES MAX  TM5 G",
            118: "DURO SESA D",
            119: "DURO HALLUX D",
            120: "DURO TM5 D",
            122: "DURO SESA G",
            123: "DURO HALLUX G",
            124: "DURO TM5 G",
            142: "Épaisseur TOTALE PARTIES MOLLES  SESA D",
            143: "Épaisseur TOTALE PARTIES MOLLES  HALLUX D",
            144: "Épaisseur TOTALE PARTIES MOLLES  TM5 D",
            146: "Épaisseur TOTALE PARTIES MOLLES  SESA G",
            147: "Épaisseur TOTALE PARTIES MOLLES  HALLUX G",
            148: "Épaisseur TOTALE PARTIES MOLLES  TM5 G",
            150: "ROC SESA D",
            151: "ROC HALLUX D",
            152: "ROC TM5 D",
            154: "ROC SESA G",
            155: "ROC HALLUX G",
            156: "ROC TM5 G",
            212: "SUDOSCAN main D",
            213: "SUDOSCAN main G",
            214: "SUDOSCAN pied D",
            215: "SUDOSCAN pied G",
        }

        summary = []
        deviation_table = {}

        for row_index, label in target_rows.items():
            values = pd.to_numeric(df.iloc[row_index, 1:], errors='coerce').dropna()
            if values.empty:
                continue  

            mean = values.mean()
            std_dev = values.std()
            deviation = values - mean

            summary.append({
                "Measure": label,
                "Mean": round(mean, 2),
                "Standard Deviation": round(std_dev, 2)
            })

            deviation_table[label] = deviation.round(2)

        summary_df = pd.DataFrame(summary)
        deviation_df = pd.DataFrame(deviation_table).dropna() 
        deviation_df.index.name = "Patient Index"

        st.subheader("📊 Summary Table")
        st.dataframe(summary_df)

        st.subheader("📉 Deviation from Mean (Écart à la Moyenne)")
        st.dataframe(deviation_df)

        output_filename = "DIAFOOT_stats_and_deviation.xlsx"
        with pd.ExcelWriter(output_filename) as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            deviation_df.to_excel(writer, sheet_name="Deviation from Mean")

        with open(output_filename, "rb") as f:
            st.download_button("📤 Download Excel File", f, file_name=output_filename)


    elif analysis_type == "🧬 ED Thickness & Hypodermis Ultrasound Analysis":

        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found.")
            st.stop()

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)

        @st.cache_data
        def prepare_data(df, risk_values):
            ed_data = df.iloc[126:134, 1:1 + len(risk_values)]
            ed_data.index = [
                "ED SESA R (mm)", "ED HALLUX R (mm)", "ED TM5 R (mm)", "ED Other R (mm)",
                "ED SESA L (mm)", "ED HALLUX L (mm)", "ED TM5 L (mm)", "ED Other L (mm)"
            ]
            df_ed = ed_data.T.apply(pd.to_numeric, errors='coerce')

            hypo_data = df.iloc[134:142, 1:1 + len(risk_values)]
            hypo_data.index = [
                "US Hypoderme SESA R (mm)", "US Hypoderme HALLUX R (mm)",
                "US Hypoderme TM5 R (mm)", "US Hypoderme Other R (mm)",
                "US Hypoderme SESA L (mm)", "US Hypoderme HALLUX L (mm)",
                "US Hypoderme TM5 L (mm)", "US Hypoderme Other L (mm)"
            ]
            df_hypo = hypo_data.T.apply(pd.to_numeric, errors='coerce')

            df_combined = pd.concat([df_ed, df_hypo], axis=1).dropna()
            df_combined["Grade"] = risk_values.loc[df_combined.index].values
            df_combined["Group"] = df_combined["Grade"].apply(lambda x: "A (Grades 0-1)" if x in [0, 1] else "B (Grades 2-3)")
            return df_combined

        df_combined = prepare_data(df, risk_values)

        selected_group = st.radio("Choose a group for intra-group analysis:", ["A (Grades 0-1)", "B (Grades 2-3)"])
        filtered = df_combined[df_combined["Group"] == selected_group]

        g1, g2 = (0, 1) if selected_group == "A (Grades 0-1)" else (2, 3)

        sub1 = filtered[filtered["Grade"] == g1]
        sub2 = filtered[filtered["Grade"] == g2]

        st.write(f"### 📋 Data Overview - Group {selected_group}")
        st.dataframe(filtered)

        if sub1.empty or sub2.empty:
            st.warning(f"Not enough data to compare Grade {g1} and Grade {g2}. Showing available data.")
            if not sub1.empty:
                st.write(f"**Grade {g1}** ({len(sub1)} patients)")
                st.dataframe(sub1)
            if not sub2.empty:
                st.write(f"**Grade {g2}** ({len(sub2)} patients)")
                st.dataframe(sub2)
        else:
            st.subheader(f"📊 T-test: Grade {g1} vs Grade {g2} in Group {selected_group}")
            ttest_results = []
            for col in filtered.columns[:-2]:
                t_stat, p_val = ttest_ind(sub1[col], sub2[col], equal_var=False, nan_policy='omit')
                ttest_results.append({"Variable": col, "p-value": p_val})
            ttest_df = pd.DataFrame(ttest_results).sort_values("p-value")
            st.dataframe(ttest_df.style.format({"p-value": "{:.4f}"}))

        st.subheader("📈 Correlation with Grade")
        if len(df_combined) >= 2:
            corr_with_grade = df_combined.drop(columns=["Group"]).corr()["Grade"].drop("Grade")
            corr_df = corr_with_grade.reset_index()
            corr_df.columns = ["Feature", "Correlation with Grade"]
            corr_df = corr_df.sort_values("Correlation with Grade", key=abs, ascending=False)
            st.dataframe(corr_df.style.format({"Correlation with Grade": "{:.2f}"}))

            fig_corr_grade, ax_corr_grade = plt.subplots(figsize=(10, 6))
            sns.barplot(data=corr_df, y="Feature", x="Correlation with Grade", palette="coolwarm", ax=ax_corr_grade)
            ax_corr_grade.set_title("Correlation of Each Feature with Grade")
            st.pyplot(fig_corr_grade)
        else:
            st.info("Not enough data to compute correlation with Grade.")

        st.subheader("📈 Correlation Matrix")
        if len(filtered) >= 2:
            corr = filtered.drop(columns=["Grade", "Group"]).corr()
            fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.info("Not enough data to compute correlation matrix.")

        st.subheader("🔁 Linear Regression to Predict Grade")
        if len(filtered) >= 2:
            X = filtered.drop(columns=["Grade", "Group"])
            y = filtered["Grade"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_
            }).sort_values(by="Coefficient", key=abs, ascending=False)

            st.dataframe(coef_df.style.format({"Coefficient": "{:.4f}"}))

            fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
            sns.barplot(data=coef_df, x="Coefficient", y="Feature", ax=ax_coef, palette="viridis")
            ax_coef.set_title("Feature Importance for Grade Prediction")
            st.pyplot(fig_coef)
        else:
            st.warning("Not enough data to perform regression.")

else:
    st.info("Please upload an Excel file containing the 'DIAFOOT' sheet.")
