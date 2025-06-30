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
import io
from scipy.stats import mannwhitneyu
from matplotlib import cm
cm.get_cmap("coolwarm")

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

#"Classification – Predict Risk Group",
#"📌 ED Thickness & Hypodermis Ultrasound Analysis", 
#"📌 GMM Clustering vs Grades", 
    analysis_type = st.sidebar.radio(
        "🧪 Choose Analysis Type:",
        ("Descriptive Analysis", "Normality Tests","Comparison of Left and Right Foot Parameters",
        "IWGDF Risk Grade Summary & Clustering", "Correlation Between Key Parameters", "Intra-Group Comparison")
    )

    # ================================
    # 🧹 Data Preparation Function
    # ================================
    def prepare_data(df):
        row_risk = df.iloc[16]
        if str(row_risk[0]).strip().lower() != "grade de risque iwgdf":
            st.error("Label 'Grade de risque IWGDF' not found in row 17.")
            st.stop()

        risk_values = pd.to_numeric(row_risk[1:], errors='coerce').dropna().astype(int)

        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "ABI R", 38: "ABI L", 94: "ROM MTP1 R", 95: "ROM MTP1 L",
            96: "ROM Ankle R", 97: "ROM Ankle L"
        }

        selected_data = df.loc[target_rows.keys(), 1:1 + len(risk_values) - 1]
        selected_data.index = [target_rows[i] for i in selected_data.index]
        df_selected = selected_data.T.apply(pd.to_numeric, errors='coerce')

        df_selected["Grade"] = risk_values.loc[df_selected.index].values
        df_selected["Group"] = df_selected["Grade"].apply(lambda x: "A (Grades 0-1)" if x in [0, 1] else "B (Grades 2-3)")
        return df_selected

    df_combined = prepare_data(df)

    # ================================
    # 📌 Stat Summary Extractor
    # ================================
    if analysis_type == "Descriptive Analysis":
        st.header("📊Descriptive Analysis")

        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L", 94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L",
            96: "ROM Ankle R", 97: "ROM Ankle L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
            154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L"
        }

        normal_params = []
        non_normal_params = []
        summary_data = []

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

            with st.expander(f"🔍 {label}"):
                st.write(f"**Mean**: {mean:.2f}")
                st.write(f"**Standard Deviation**: {std:.2f}")
                st.write(f"**Shapiro-W**: {w_stat:.4f} | **p-value**: {p_value:.4f}")
                st.write(f"Min: {min_val:.2f}, Q1: {q1:.2f}, Median: {median:.2f}, Q3: {q3:.2f}, Max: {max_val:.2f}")
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
            17: "Height (m)", 18: "Weight (kg)", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L", 94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L",
            96: "ROM Ankle R", 97: "ROM Ankle L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
            154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L"
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
    # 📌 Comparison of Left and Right Foot Parameters
    # ================================
    elif analysis_type == "Comparison of Left and Right Foot Parameters":

        st.header("🦶 Comparison of Left and Right Foot Parameters with Plots")

        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "MESI Big Toe Systolic Pressure Index R", 38: "MESI Big Toe Systolic Pressure Index L", 94: "Dorsal flexion range MTP1 R", 95: "Dorsal flexion range MTP1 L",
            96: "ROM Ankle R", 97: "ROM Ankle L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
            154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L"
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

        for label_r, label_l, idx_r, idx_l in paired_parameters:
            values_r = pd.to_numeric(df.iloc[idx_r, 1:], errors='coerce').dropna()
            values_l = pd.to_numeric(df.iloc[idx_l, 1:], errors='coerce').dropna()

            common_len = min(len(values_r), len(values_l))
            values_r = values_r.iloc[:common_len]
            values_l = values_l.iloc[:common_len]

            # Normality tests
            p_r = shapiro(values_r)[1]
            p_l = shapiro(values_l)[1]

            if p_r > 0.05 and p_l > 0.05:
                stat, p_val = ttest_rel(values_r, values_l)
                test_name = "Paired t-test"
            else:
                stat, p_val = wilcoxon(values_r, values_l)
                test_name = "Wilcoxon signed-rank test"

            st.markdown(f"### {label_r} vs {label_l}")
            st.write(f"**Test used:** {test_name}")
            st.write(f"**Statistic:** {stat:.4f}", f"**p-value:** {p_val:.4f}")

            data_plot = pd.DataFrame({
                'Right': values_r.reset_index(drop=True),
                'Left': values_l.reset_index(drop=True)
            })

            # Plot boxplot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=data_plot, ax=ax, palette=["#90CAF9", "#F48FB1"])

            # Draw mean and median lines
            for i, col in enumerate(data_plot.columns):
                mean_val = data_plot[col].mean()
                median_val = data_plot[col].median()
                ax.hlines(mean_val, i - 0.25, i + 0.25, colors='red', label='Mean' if i == 0 else "", linewidth=2)
                ax.hlines(median_val, i - 0.25, i + 0.25, colors='blue', label='Median' if i == 0 else "", linewidth=2, linestyles='--')

            # Only show one legend entry per label
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.set_title(f"Distribution: {label_r} vs {label_l}")
            st.pyplot(fig)
            st.markdown("---")

    # ================================
    # 📌 IWGDF Risk Grade Summary & KMeans Clustering
    # ================================
    elif analysis_type == "IWGDF Risk Grade Summary & Clustering":

        # Locate IWGDF risk grade row
        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]

        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found in the Excel sheet.")
            st.stop()

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        patient_ids = pd.Series(range(1, len(risk_values) + 1), index=risk_values.index)

        # Basic statistics
        min_iwgdf = risk_values.min()
        max_iwgdf = risk_values.max()
        mean_iwgdf = risk_values.mean()

        low_moderate_patients = risk_values[risk_values.isin([0, 1])].index + 1
        high_very_high_patients = risk_values[risk_values.isin([2, 3])].index + 1

        st.subheader("📈 IWGDF Risk Grade Summary")
        st.markdown(f"""
        - **Grade range**: {min_iwgdf} (low) to {max_iwgdf} (very high)
        - **Average grade**: ~{mean_iwgdf:.1f}
        - **Low/Moderate Risk (0–1)**: Patients {', '.join(map(str, low_moderate_patients.tolist()))}
        - **High/Very High Risk (2–3)**: Patients {', '.join(map(str, high_very_high_patients.tolist()))}
        """)

        # Frequency table
        freq = pd.Series([0, 1, 2, 3]).apply(lambda x: (risk_values == x).sum())
        freq.index = [0, 1, 2, 3]
        risk_labels = {0: "Low risk", 1: "Moderate risk", 2: "High risk", 3: "Very high risk"}

        st.markdown("### 📊 Frequency of Each IWGDF Risk Grade")
        freq_df = pd.DataFrame({
            "Grade": freq.index,
            "Label": [risk_labels[g] for g in freq.index],
            "Count": freq.values
        })
        st.dataframe(freq_df)

        # Clustering
        X = risk_values.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        results = pd.DataFrame({
            "Patient": patient_ids.values,
            "IWGDF_Grade": risk_values.values,
            "Cluster": clusters
        })

        # Plot clustering result
        fig, ax = plt.subplots(figsize=(8, 4))
        scatter = ax.scatter(results["Patient"], results["IWGDF_Grade"], c=results["Cluster"], cmap="viridis", s=100)
        ax.set_xlabel("Patient")
        ax.set_ylabel("IWGDF Grade")
        ax.set_title("Clustering Patients Based on IWGDF Grade")
        ax.set_xticks(results["Patient"])
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(True)
        st.pyplot(fig)

        # Cluster summary
        st.markdown("### 🧾 Cluster Composition")
        cluster_summary = results.groupby("Cluster")["IWGDF_Grade"].agg(["count", "mean", "min", "max"])
        st.dataframe(cluster_summary)

        # ==== Scatter plots for all features ====

        st.markdown("### 🎨 Scatter Plots of Features vs IWGDF Risk Grade")

        target_rows = {
            17: "Height (m)", 18: "Weight (kg)", 35: "MESI Ankle Pressure R", 36: "MESI Ankle Pressure L",
            37: "ABI R", 38: "ABI L", 94: "ROM MTP1 R", 95: "ROM MTP1 L",
            96: "ROM Ankle R", 97: "ROM Ankle L",
            108: "Avg Pressure Max SESA R", 109: "Avg Pressure Max HALLUX R",
            110: "Avg Pressure Max TM5 R", 113: "Avg Pressure Max SESA L",
            114: "Avg Pressure Max HALLUX L", 115: "Avg Pressure Max TM5 L",
            118: "Stiffness SESA R", 119: "Stiffness HALLUX R", 120: "Stiffness TM5 R",
            122: "Stiffness SESA L", 123: "Stiffness HALLUX L", 124: "Stiffness TM5 L",
            142: "Total Tissue Thickness SESA R", 143: "Total Tissue Thickness HALLUX R",
            144: "Total Tissue Thickness TM5 R", 146: "Total Tissue Thickness SESA L",
            147: "Total Tissue Thickness HALLUX L", 148: "Total Tissue Thickness TM5 L",
            150: "ROC SESA R", 151: "ROC HALLUX R", 152: "ROC TM5 R",
            154: "ROC SESA L", 155: "ROC HALLUX L", 156: "ROC TM5 L",
            212: "SUDOSCAN Hand R", 213: "SUDOSCAN Hand L", 214: "SUDOSCAN Foot R", 215: "SUDOSCAN Foot L"
        }

        # Extract feature data matching risk_values length
        selected_data = df.loc[target_rows.keys(), 1:1 + len(risk_values) - 1]
        selected_data.index = [target_rows[i] for i in selected_data.index]
        df_features = selected_data.T.apply(pd.to_numeric, errors='coerce')

        # Add the risk grade column for coloring/scatter Y axis
        df_features["IWGDF_Grade"] = risk_values.values

        # Plot scatter for each feature against IWGDF Grade
        for feature in df_features.columns[:-1]:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            sc = ax2.scatter(df_features[feature], df_features["IWGDF_Grade"], 
                            c=df_features["IWGDF_Grade"], cmap="coolwarm", s=60, edgecolor='k', alpha=0.7)
            ax2.set_xlabel(feature)
            ax2.set_ylabel("IWGDF Grade")
            ax2.set_title(f"{feature} vs IWGDF Grade")
            plt.colorbar(sc, ax=ax2, label="IWGDF Grade")
            st.pyplot(fig2)

        # Optional download
        output_file = "IWGDF_clustering_results.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            results.to_excel(writer, sheet_name="Clustering", index=False)
            freq_df.to_excel(writer, sheet_name="Frequencies", index=False)

        with open(output_file, "rb") as f:
            st.download_button("📤 Download Clustering Results", f, file_name=output_file)

    # ================================
    # Classification – Predict Risk Grou
    # ================================
    elif analysis_type == "Classification – Predict Risk Group":
        st.header("📌 Predict Risk Group using Biomechanical & Clinical Features")

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
        features = df_combined.drop(columns=["Grade", "Group"])
        features = features.dropna(axis=0, how='any')
        target = df_combined.loc[features.index, "Group"].map({"A (Grades 0-1)": 0, "B (Grades 2-3)": 1})
        target.name = "Target"
        if features.empty or target.empty:
            st.error("Features or target data is empty after removing missing values.")
            st.stop()
        X = features
        y = target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Accuracy
        acc = clf.score(X_test, y_test)
        st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")

        # Classification report
        st.text("📊 Classification Report:")
        st.code(classification_report(y_test, y_pred, target_names=["Group A (0–1)", "Group B (2–3)"]))

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Group A", "Group B"], ax=ax_cm)
        st.pyplot(fig_cm)

        # Feature importances
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write("📈 Top Predictive Features:")
        st.bar_chart(importances.head(10))

        # Predict for all data
        full_preds = df_combined.copy()
        full_preds = full_preds.dropna(subset=X.columns)
        full_preds["Predicted Group"] = clf.predict(full_preds[X.columns])
        full_preds["Predicted Label"] = full_preds["Predicted Group"].map({0: "A", 1: "B"})

        st.dataframe(full_preds[["Grade", "Group", "Predicted Label"] + list(X.columns[:5])])
            
    # ================================
    # 📌 KMeans Clustering  *
    # Top 6 features most correlated with Grade
    # ================================
    elif analysis_type == "📌 KMeans Clustering Based on Selected Features":
        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found.")
            st.stop()

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce')

        target_rows = {
            17: "Taille (m)", 18: "Poids (kg)", 35: "MESI PRESSION GO D", 36: "MESI PRESSION GO G",
            37: "IPS GO D", 38: "IPS GO G", 94: "AMPLI MTP1 D", 95: "AMPLI MTP1 G",
            96: "AMPLI TALO CRURALE D", 97: "AMPLI TALO CRURALE G", 108: "Pression MOYENNE DES MAX SESA D",
            109: "Pression MOYENNE DES MAX HALLUX D", 110: "Pression MOYENNE DES MAX  TM5 D",
            113: "Pression MOYENNE DES MAX  SESA G", 114: "Pression MOYENNE DES MAX  HALLUX G",
            115: "Pression MOYENNE DES MAX  TM5 G", 118: "DURO SESA D", 119: "DURO HALLUX D",
            120: "DURO TM5 D", 122: "DURO SESA G", 123: "DURO HALLUX G", 124: "DURO TM5 G",
            142: "Épaisseur TOTALE PARTIES MOLLES  SESA D", 143: "Épaisseur TOTALE PARTIES MOLLES  HALLUX D",
            144: "Épaisseur TOTALE PARTIES MOLLES  TM5 D", 146: "Épaisseur TOTALE PARTIES MOLLES  SESA G",
            147: "Épaisseur TOTALE PARTIES MOLLES  HALLUX G", 148: "Épaisseur TOTALE PARTIES MOLLES  TM5 G",
            150: "ROC SESA D", 151: "ROC HALLUX D", 152: "ROC TM5 D", 154: "ROC SESA G",
            155: "ROC HALLUX G", 156: "ROC TM5 G", 212: "SUDOSCAN main D", 213: "SUDOSCAN main G",
            214: "SUDOSCAN pied D", 215: "SUDOSCAN pied G"
        }

        # 📥 Collect feature data
        feature_data = {}
        for row_index, label in target_rows.items():
            values = pd.to_numeric(df.iloc[row_index, 1:], errors='coerce')
            feature_data[label] = values

        features_df = pd.DataFrame(feature_data)

        # 📊 Merge with Grade and clean
        full_df = features_df.copy()
        full_df["Grade"] = risk_values
        full_df = full_df.dropna()

        X = full_df.drop(columns=["Grade"])
        y = full_df["Grade"]

        # ⚖️ Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        # 🧠 Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        full_df["Cluster"] = kmeans.fit_predict(X_scaled)

        # 📈 Evaluation Metrics
        silhouette = silhouette_score(X_scaled, full_df["Cluster"])
        ari = adjusted_rand_score(y, full_df["Cluster"])
        st.markdown(f"**Silhouette Score**: {silhouette:.3f} — (1 = well-separated, 0 = overlap)<br>**Adjusted Rand Index (ARI)**: {ari:.3f} — (1 = perfect match to true labels)", unsafe_allow_html=True)

        # ✨ Explanation Section
        with st.expander("ℹ️ Explanation of indicators and charts"):
            st.markdown("""
            - **Silhouette Score**: An index to measure the quality of clustering. A value close to 1 means that the clusters are well separated.
            - **Adjusted Rand Index (ARI)**: Shows the degree of alignment between the actual labels (Grade) and the clustering.
            - **PCA Visualization**: A 2D view of the data after dimensionality reduction with PCA to see the formation of clusters.
            """)
            
        # 📋 Contingency Table
        cluster_vs_grade = pd.crosstab(full_df["Grade"], full_df["Cluster"])
        st.write("### 🔍 Cluster vs True Grade (Contingency Table)")
        st.dataframe(cluster_vs_grade)

        fig_kmeans, ax_kmeans = plt.subplots(figsize=(8, 6))
        sns.heatmap(cluster_vs_grade, annot=True, fmt="d", cmap="Blues", ax=ax_kmeans)
        ax_kmeans.set_title("Heatmap: Clusters vs True Grades")
        ax_kmeans.set_xlabel("Cluster")
        ax_kmeans.set_ylabel("True Grade")
        st.pyplot(fig_kmeans)

        # ✨ Explanation Section
        with st.expander("ℹ️ Explanation of indicators and charts"):
            st.markdown("""
            - **The heatmap shows that cluster 0 contains a mix of different true degrees, especially with 4 samples at degree 3.0 which is the most densely populated. 
            - **The other clusters (1, 2 and 3) have fewer samples and generally show limited dispersion. This is consistent with the previous consensus table and the low silhouette and ARI scores (0.097 and 0.101), indicating a low quality of cluster separation.
            """)
            
        # 📊 Cluster Means
        st.subheader("📊 Moyennes par Cluster")
        cluster_means = full_df.groupby("Cluster").mean(numeric_only=True)
        st.dataframe(cluster_means.style.format(precision=2))
    
        # 🌀 PCA for Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = full_df["Cluster"].values
        pca_df["Grade"] = y.values

        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", style="Grade", palette="Set2", ax=ax_pca)
        ax_pca.set_title("2D PCA Visualization of Clusters and Grades")
        st.pyplot(fig_pca)

        # ✨ Explanation Section
        with st.expander("ℹ️ Explanation of indicators and charts"):
            st.markdown("""
            - **Horizontal axis (PC1): The first principal component that explains most of the variance in the data.
            - **Vertical axis (PC2): The second principal component that shows the remaining variance.
            - **The green dots (cluster 0) are scattered at the top and bottom of the plot, indicating the extent of this cluster in space.
            - **The orange dots (cluster 1) are located on the left and bottom and are limited in number.
            - **The blue dots (cluster 2) are seen in the middle and left and are less scattered.
            - **The pink dots (cluster 3) are concentrated at the top and center of the plot.
            - **The true degrees (0.0, 1.0, 2.0, 3.0) are represented by different symbols and do not seem to be well separated into clusters, which is consistent with the low Silhouette (0.097) and ARI (0.101) scores.

            Interpretation:
            The scatter of points and the overlapping of colors indicate that the obtained clustering did not separate the true degrees well. This is consistent with previous analyses (agreement table and heat map) and indicates the need to improve the clustering algorithm or the features used in PCA.
            """)

    # ================================
    # 📌 ED Thickness & Hypodermis Analysis
    # ================================
    elif analysis_type == "📌 ED Thickness & Hypodermis Ultrasound Analysis":
        df_combined = prepare_data(df)

        selected_group = st.radio("Choose a group for intra-group analysis:", ["A (Grades 0-1)", "B (Grades 2-3)"])
        filtered = df_combined[df_combined["Group"] == selected_group]

        g1, g2 = (0, 1) if selected_group == "A (Grades 0-1)" else (2, 3)

        sub1 = filtered[filtered["Grade"] == g1]
        sub2 = filtered[filtered["Grade"] == g2]

        st.write(f"### 📋 Data Overview - Group {selected_group}")
        st.dataframe(filtered)

        if sub1.empty or sub2.empty:
            st.warning(f"Not enough data to compare Grade {g1} and Grade {g2}.")
        else:
            st.subheader(f"📊 T-test: Grade {g1} vs Grade {g2}")
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

        st.subheader("📈 Correlation Matrix")
        if len(filtered) >= 2:
            corr = filtered.drop(columns=["Grade", "Group"]).corr()
            fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

        st.subheader("🔁 Linear Regression to Predict Grade")
        if len(filtered) >= 2:
            X = filtered.drop(columns=["Grade", "Group"])
            y = filtered["Grade"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
            coef_df = coef_df.sort_values(by="Coefficient", key=abs, ascending=False)
            st.dataframe(coef_df.style.format({"Coefficient": "{:.4f}"}))

            fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
            sns.barplot(data=coef_df, x="Coefficient", y="Feature", ax=ax_coef, palette="viridis")
            ax_coef.set_title("Feature Importance for Grade Prediction")
            st.pyplot(fig_coef)

    # ================================
    # 📌 GMM Clustering vs Grades
    # Top 6 features most correlated with Grade
    # ================================
    elif analysis_type == "📌 GMM Clustering vs Grades":
        st.subheader("🔍 GMM Clustering Based on Features Correlated with Grade")

        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import adjusted_rand_score
        from sklearn.decomposition import PCA

        # 🔹 Step 1: Locate risk grades row
        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found.")
            st.stop()

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)
        expected_length = len(risk_values)

        # 🔹 Step 2: Define data rows to extract
        target_rows = {
            17: "Taille (m)", 18: "Poids (kg)", 35: "MESI PRESSION GO D", 36: "MESI PRESSION GO G",
            37: "IPS GO D", 38: "IPS GO G", 94: "AMPLI MTP1 D", 95: "AMPLI MTP1 G",
            96: "AMPLI TALO CRURALE D", 97: "AMPLI TALO CRURALE G",
            108: "Pression MOYENNE DES MAX SESA D", 109: "Pression MOYENNE DES MAX HALLUX D",
            110: "Pression MOYENNE DES MAX  TM5 D", 113: "Pression MOYENNE DES MAX  SESA G",
            114: "Pression MOYENNE DES MAX  HALLUX G", 115: "Pression MOYENNE DES MAX  TM5 G",
            118: "DURO SESA D", 119: "DURO HALLUX D", 120: "DURO TM5 D", 122: "DURO SESA G",
            123: "DURO HALLUX G", 124: "DURO TM5 G", 142: "Épaisseur TOTALE PARTIES MOLLES  SESA D",
            143: "Épaisseur TOTALE PARTIES MOLLES  HALLUX D", 144: "Épaisseur TOTALE PARTIES MOLLES  TM5 D",
            146: "Épaisseur TOTALE PARTIES MOLLES  SESA G", 147: "Épaisseur TOTALE PARTIES MOLLES  HALLUX G",
            148: "Épaisseur TOTALE PARTIES MOLLES  TM5 G", 150: "ROC SESA D", 151: "ROC HALLUX D",
            152: "ROC TM5 D", 154: "ROC SESA G", 155: "ROC HALLUX G", 156: "ROC TM5 G",
            212: "SUDOSCAN main D", 213: "SUDOSCAN main G", 214: "SUDOSCAN pied D", 215: "SUDOSCAN pied G"
        }

        # 🔹 Step 3: Extract and validate features
        feature_dict = {}
        invalid_rows = []

        for idx, name in target_rows.items():
            try:
                values = pd.to_numeric(df.iloc[idx, 1:], errors='coerce')
                if len(values.dropna()) == expected_length:
                    feature_dict[name] = values.values[:expected_length]
                else:
                    invalid_rows.append((idx, name, f"Expected {expected_length}, got {len(values.dropna())}"))
            except Exception as e:
                invalid_rows.append((idx, name, str(e)))

        if not feature_dict:
            st.error("❌ No valid features extracted. Check the data format.")
            st.write("🛠️ Debug Info - Invalid/Mismatched rows:")
            st.write(invalid_rows)
            st.stop()

        df_features = pd.DataFrame(feature_dict)
        df_features = df_features.iloc[:expected_length].dropna()
        df_features["Grade"] = risk_values.values[:len(df_features)]
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

        # 🔹 Step 7: Adjusted Rand Index (ARI)
        ari = adjusted_rand_score(y_true, clusters)
        st.metric("🧮 Adjusted Rand Index (ARI)", f"{ari:.3f}")

        # 🔹 Step 8: PCA Visualization
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["GMM_Cluster"] = clusters
        pca_df["True Grade"] = y_true.values

        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="GMM_Cluster", style="True Grade", palette="deep", ax=ax_pca)
        ax_pca.set_title("PCA View: GMM Clusters vs True Grades")
        st.pyplot(fig_pca)
    
    # ================================
    # 📌 Correlation Between Key Parameters
    # ================================
    elif analysis_type == "Correlation Between Key Parameters":
        st.subheader("🔗 Correlation Between Selected DIAFOOT Parameters")

        target_rows = {
            17: "Taille (m)", 18: "Poids (kg)", 35: "MESI PRESSION GO D", 36: "MESI PRESSION GO G",
            37: "IPS GO D", 38: "IPS GO G", 94: "AMPLI MTP1 D", 95: "AMPLI MTP1 G",
            96: "AMPLI TALO CRURALE D", 97: "AMPLI TALO CRURALE G",
            108: "Pression MOYENNE DES MAX SESA D", 109: "Pression MOYENNE DES MAX HALLUX D",
            110: "Pression MOYENNE DES MAX  TM5 D", 113: "Pression MOYENNE DES MAX  SESA G",
            114: "Pression MOYENNE DES MAX  HALLUX G", 115: "Pression MOYENNE DES MAX  TM5 G",
            118: "DURO SESA D", 119: "DURO HALLUX D", 120: "DURO TM5 D", 122: "DURO SESA G",
            123: "DURO HALLUX G", 124: "DURO TM5 G", 142: "Épaisseur TOTALE PARTIES MOLLES  SESA D",
            143: "Épaisseur TOTALE PARTIES MOLLES  HALLUX D", 144: "Épaisseur TOTALE PARTIES MOLLES  TM5 D",
            146: "Épaisseur TOTALE PARTIES MOLLES  SESA G", 147: "Épaisseur TOTALE PARTIES MOLLES  HALLUX G",
            148: "Épaisseur TOTALE PARTIES MOLLES  TM5 G", 150: "ROC SESA D", 151: "ROC HALLUX D",
            152: "ROC TM5 D", 154: "ROC SESA G", 155: "ROC HALLUX G", 156: "ROC TM5 G",
            212: "SUDOSCAN main D", 213: "SUDOSCAN main G", 214: "SUDOSCAN pied D", 215: "SUDOSCAN pied G"
        }

        # Extract data from rows into a DataFrame
        data_dict = {}
        for row_index, label in target_rows.items():
            values = pd.to_numeric(df.iloc[row_index, 1:], errors='coerce')
            data_dict[label] = values

        df_corr = pd.DataFrame(data_dict).dropna()

        if df_corr.empty or df_corr.shape[0] < 2:
            st.warning("⚠️ Not enough valid data to compute correlation. Please check your Excel file.")
            st.stop()

        st.write("### 📋 Cleaned Data Table (dropna)")
        st.dataframe(df_corr)

        # Correlation matrix
        st.write("### 📊 Correlation Matrix (Pearson)")
        corr_matrix = df_corr.corr()

        cmap = "coolwarm"
        st.dataframe(corr_matrix.style.background_gradient(cmap=cmap, axis=None).format("{:.2f}"))

        # Heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, ax=ax_corr)
        ax_corr.set_title("Correlation Heatmap of Selected Parameters", fontsize=16)
        st.pyplot(fig_corr)

        # Optional: Show correlation of selected variable with others
        selected_ref = st.selectbox("📌 Show correlations with a specific variable:", corr_matrix.columns.tolist())
        ref_corr = corr_matrix[selected_ref].drop(labels=[selected_ref])
        ref_corr_sorted = ref_corr.reindex(ref_corr.abs().sort_values(ascending=False).index)

        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(y=ref_corr_sorted.index, x=ref_corr_sorted.values, palette="vlag", ax=ax_bar)
        ax_bar.set_title(f"Correlation of Other Variables with '{selected_ref}'")
        ax_bar.set_xlabel("Correlation Coefficient")
        ax_bar.set_xlim(-1, 1)
        st.pyplot(fig_bar)

        # Detect high correlation pairs
        st.markdown("### 🔍 Highly Correlated Pairs")
        threshold = st.slider("🔧 Correlation threshold", 0.5, 1.0, 0.8, 0.05)
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    high_corr_pairs.append({
                        "Feature 1": corr_matrix.columns[i],
                        "Feature 2": corr_matrix.columns[j],
                        "Correlation": round(val, 3)
                    })

        if high_corr_pairs:
            st.dataframe(pd.DataFrame(high_corr_pairs))
            st.markdown("💡 *Consider removing one variable from each pair with high correlation to reduce redundancy in models.*")
        else:
            st.success("✅ No pairs with correlation above the threshold.")

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
# 📎 File Not Uploaded Message
# ================================
else:
    st.info("Please upload an Excel file containing the 'DIAFOOT' sheet.")
    
