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

    analysis_type = st.sidebar.radio(
        "🧪 Choose Analysis Type:",
        ("📌 Stat Summary Extractor","📌 IWGDF Risk Grade Summary & Clustering",
        "📌 ED Thickness & Hypodermis Ultrasound Analysis", "📌 GMM Clustering vs Grades", 
        "📌 Correlation Between Key Parameters")
    )

    # ================================
    # 🧹 Data Preparation Function
    # ================================
    def prepare_data(df):
        label_risk = "Grade de risque IWGDF"
        row_risk = df[df[0].astype(str).str.strip().str.lower() == label_risk.lower()]
        if row_risk.empty:
            st.error(f"Label '{label_risk}' not found.")
            st.stop()

        idx_risk = row_risk.index[0]
        risk_values = pd.to_numeric(df.iloc[idx_risk, 1:], errors='coerce').dropna().astype(int)

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

    # ================================
    # 📌 Stat Summary Extractor
    # ================================
    if analysis_type == "📌 Stat Summary Extractor":
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

        summary = []
        deviation_table = {}

        for row_index, label in target_rows.items():
            values = pd.to_numeric(df.iloc[row_index, 1:], errors='coerce').dropna()
            if values.empty:
                continue
            mean = values.mean()
            std_dev = values.std()
            deviation = values - mean

            summary.append({"Measure": label, "Mean": round(mean, 2), "Standard Deviation": round(std_dev, 2)})
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
            
    # ================================
    # 📌 IWGDF Risk Grade Summary & KMeans Clustering
    # ================================
    elif analysis_type == "📌 IWGDF Risk Grade Summary & Clustering":

        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        import streamlit as st

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

        # Optional download
        output_file = "IWGDF_clustering_results.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            results.to_excel(writer, sheet_name="Clustering", index=False)
            freq_df.to_excel(writer, sheet_name="Frequencies", index=False)

        with open(output_file, "rb") as f:
            st.download_button("📤 Download Clustering Results", f, file_name=output_file)

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
    elif analysis_type == "📌 Correlation Between Key Parameters":
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

        data_dict = {}
        for row_index, label in target_rows.items():
            values = pd.to_numeric(df.iloc[row_index, 1:], errors='coerce')
            data_dict[label] = values

        df_corr = pd.DataFrame(data_dict).dropna()

        st.write("### 🧾 Cleaned Data Table (dropna)")
        st.dataframe(df_corr)

        # Correlation matrix
        corr_matrix = df_corr.corr()

        st.write("### 📊 Correlation Matrix")
        cmap = "coolwarm"
        st.dataframe(corr_matrix.style.background_gradient(cmap=cmap, axis=None).format("{:.2f}"))

        fig_corr, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap of Selected Parameters", fontsize=16)
        st.pyplot(fig_corr)

        # Optional: correlation with a specific variable
        selected_ref = st.selectbox("📌 Show correlations with a specific variable:", corr_matrix.columns.tolist())
        ref_corr = corr_matrix[selected_ref].drop(labels=[selected_ref])
        ref_corr_sorted = ref_corr.reindex(ref_corr.abs().sort_values(ascending=False).index)

        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(y=ref_corr_sorted.index, x=ref_corr_sorted.values, palette="vlag", ax=ax_bar)
        ax_bar.set_title(f"Correlation of Other Variables with '{selected_ref}'")
        ax_bar.set_xlabel("Correlation Coefficient")
        ax_bar.set_xlim(-1, 1)
        st.pyplot(fig_bar)

    # ================================
    # 📌 Mechanical Features Correlation Analysis
    # ================================
    elif analysis_type == "📌 Mechanical Features Correlation Analysis":

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
        st.subheader("🔬 Analyse de corrélation entre les paramètres mécaniques")

        data = {}
        labels = []

        for row_idx, label in target_rows.items():
            values = pd.to_numeric(df.iloc[row_idx, 1:], errors="coerce")
            if values.isna().sum() < len(values): 
                data[label] = values
                labels.append(label)

        mech_df = pd.DataFrame(data)

        if mech_df.empty or len(mech_df) < 2:
            st.warning("Pas assez de données valides pour cette analyse.")
        else:
            st.write("### 🧾 Données disponibles")
            st.dataframe(mech_df)

            corr = mech_df.corr()

            st.write("### 🔗 Matrice de corrélation")
            fig_corr, ax_corr = plt.subplots(figsize=(16, 10))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, mask=np.triu(np.ones_like(corr, dtype=bool)))
            st.pyplot(fig_corr)

            threshold = st.slider("Seuil de corrélation élevée", 0.5, 1.0, 0.8, 0.05)
            high_corr_pairs = []

            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) >= threshold:
                        high_corr_pairs.append({
                            "Feature 1": corr.columns[i],
                            "Feature 2": corr.columns[j],
                            "Correlation": round(val, 3)
                        })

            if high_corr_pairs:
                st.write(f"### 🚨 Paires avec corrélation absolue > {threshold}")
                st.dataframe(pd.DataFrame(high_corr_pairs))
            else:
                st.success("Aucune paire avec corrélation élevée selon le seuil défini.")

            st.markdown("💡 *Vous pouvez envisager de supprimer l'une des variables de chaque paire très corrélée pour réduire la redondance.*")


# ================================
# 📎 File Not Uploaded Message
# ================================
else:
    st.info("Please upload an Excel file containing the 'DIAFOOT' sheet.")
