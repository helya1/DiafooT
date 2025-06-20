import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Analyse statistique des données Myoton")

uploaded_file = st.file_uploader("📂 Téléchargez le fichier Excel", type=["xlsx"])

if uploaded_file:
    sheet_name = "Manips resultats"
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

    st.subheader("✅ Données brutes (aperçu)")
    st.write(df.head(21))

    # Index mapping pour les pieds droit et gauche
    index_map = {
        'tone_right':      [i-1 for i in [3,4,5,23,24,25,43,44,45,63,64,65,83,84,85,103,104,105,123,124,125,143,144,145,163,164,165]],
        'stiffness_right': [i-1 for i in [6,7,8,26,27,28,46,47,48,66,67,68,86,87,88,106,107,108,126,127,128,146,147,148,166,167,168]],
        'elasticity_right':[i-1 for i in [9,10,11,29,30,31,49,50,51,69,70,71,89,90,91,109,110,111,129,130,131,149,150,151,169,170,171]],
        
        'tone_left':       [i-1 for i in [13,14,15,33,34,35,53,54,55,73,74,75,93,94,95,113,114,115,133,134,135,153,154,161,173,174,175]],
        'stiffness_left':  [i-1 for i in [16,17,18,36,37,38,56,57,58,76,77,78,96,97,98,116,117,118,136,137,138,156,157,158,176,177,178]],
        'elasticity_left': [i-1 for i in [19,20,21,39,40,41,59,60,61,79,80,81,99,100,101,119,120,121,139,140,141,159,160,161,179,180,181]]
    }

    foot_zones = ['Hallux', '1T', '2-3T', '5T', 'Voute int', 'Voute ext', 'Talon']

    def extract_param(index_list):
        return df.iloc[index_list].reset_index(drop=True).T

    def mean_std_ci(data):
        # محاسبه میانگین، انحراف معیار و 95% CI
        n = data.count()
        mean = data.mean()
        std = data.std()
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else np.nan
        return mean, std, ci95

    def test_normality_and_plot(param_df, title):
        st.markdown(f"### 📈 {title}")
        for col in param_df.columns:
            numeric_data = pd.to_numeric(param_df[col], errors='coerce').dropna()
            if len(numeric_data) < 3:
                st.warning(f"🚫 Trop peu de données numériques dans la colonne {col} pour effectuer le test de Shapiro.")
                continue
            stat, p = stats.shapiro(numeric_data)
            fig, ax = plt.subplots()
            sns.histplot(numeric_data, kde=True, ax=ax, color='skyblue')
            ax.axvline(numeric_data.mean(), color='red', linestyle='--', label='Moyenne')
            ax.set_title(f"{col} - Shapiro p={p:.4f}")
            st.pyplot(fig)

    def wilcoxon_test_and_plot(df1, df2, title):
        st.markdown(f"### 🧪 {title}")
        p_values = []
        cleaned_cols = []

        for col in df1.columns:
            series1 = pd.to_numeric(df1[col], errors='coerce')
            series2 = pd.to_numeric(df2[col], errors='coerce')
            valid = series1.notna() & series2.notna()
            series1 = series1[valid]
            series2 = series2[valid]

            if len(series1) < 3:
                st.warning(f"🚫 Trop peu de données numériques dans la colonne {col} pour effectuer le test de Wilcoxon.")
                p_values.append(np.nan)
                continue

            stat, p = stats.wilcoxon(series1, series2)
            p_values.append(p)
            cleaned_cols.append(col)

        # Calcul des stats descriptives + CI
        stats_list = []
        for col in cleaned_cols:
            right = pd.to_numeric(df1[col], errors='coerce').dropna()
            left = pd.to_numeric(df2[col], errors='coerce').dropna()
            mean_r, std_r, ci_r = mean_std_ci(right)
            mean_l, std_l, ci_l = mean_std_ci(left)
            stats_list.append({
                'Zone': col,
                'Droit (mean ± std)': f"{mean_r:.2f} ± {std_r:.2f}",
                'Droit 95% CI': f"± {ci_r:.2f}" if not np.isnan(ci_r) else 'NA',
                'Gauche (mean ± std)': f"{mean_l:.2f} ± {std_l:.2f}",
                'Gauche 95% CI': f"± {ci_l:.2f}" if not np.isnan(ci_l) else 'NA',
                'p-valeur': p_values[cleaned_cols.index(col)],
                'Significatif': '✅' if p_values[cleaned_cols.index(col)] < 0.05 else ''
            })

        stats_df = pd.DataFrame(stats_list)
        st.write(stats_df)

        # Boxplot avec zones significatives en rouge
        fig, ax = plt.subplots(figsize=(10, 5))
        df_box = pd.concat([
            pd.DataFrame({'Valeur': pd.to_numeric(df1[col], errors='coerce'), 'Groupe': 'Droit', 'Zone': col}) for col in cleaned_cols
        ] + [
            pd.DataFrame({'Valeur': pd.to_numeric(df2[col], errors='coerce'), 'Groupe': 'Gauche', 'Zone': col}) for col in cleaned_cols
        ])
        df_box = df_box.dropna()

        # Highlight zones significatives
        palette = ['red' if p_values[i] < 0.05 else 'gray' for i in range(len(p_values))]

        sns.boxplot(x="Zone", y="Valeur", hue="Groupe", data=df_box, ax=ax, palette=["#1f77b4", "#ff7f0e"])
        for tick, color in zip(ax.get_xticklabels(), palette):
            tick.set_color(color)

        ax.set_title(title)
        st.pyplot(fig)

    def correlation_plots(param_df, thickness):
        st.markdown("### 🔗 Analyse de corrélation avec l'épaisseur")

        numeric_df = param_df.apply(pd.to_numeric, errors='coerce')
        thickness_series = pd.Series(thickness)

        # Scatter plots avec ligne de tendance
        for col in numeric_df.columns:
            x = numeric_df[col]
            y = thickness_series

            valid = x.notna() & y.notna()
            x_valid = x[valid]
            y_valid = y[valid]

            if len(x_valid) < 3:
                st.warning(f"🚫 Trop peu de données numériques dans la colonne {col} pour la corrélation.")
                continue

            corr, p = stats.spearmanr(x_valid, y_valid)
            st.write(f"📌 {col} - Corrélation Spearman : {corr:.2f}, p={p:.4f}")

            fig, ax = plt.subplots()
            sns.regplot(x=x_valid, y=y_valid, ax=ax, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
            ax.set_title(f"Corrélation {col} avec épaisseur (Spearman r={corr:.2f}, p={p:.4f})")
            st.pyplot(fig)

        # Heatmap de corrélation
        combined = numeric_df.copy()
        combined['Thickness'] = thickness_series
        corr_matrix = combined.corr(method='spearman')

        st.markdown("### 🔥 Heatmap des corrélations (Spearman)")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    def plot_plan_theatral(param_df, title):
        st.markdown(f"### 🎭 Graphique plan théâtral - {title}")

        numeric_df = param_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        
        mean = numeric_df.mean()
        std = numeric_df.std()

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(mean))
        ax.errorbar(x, mean, yerr=1.96 * std, fmt='o', capsize=5)
        ax.axhline(mean.mean(), color='red', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(mean.index, rotation=45)
        ax.set_title(title)
        st.pyplot(fig)


    for param in ['tone', 'stiffness', 'elasticity']:
        right_df = extract_param(index_map[f'{param}_right']).iloc[:, :7]
        left_df = extract_param(index_map[f'{param}_left']).iloc[:, :7]

        right_df.columns = foot_zones
        left_df.columns = foot_zones

        test_normality_and_plot(right_df, f'{param.upper()} - Pied droit')
        test_normality_and_plot(left_df, f'{param.upper()} - Pied gauche')

        wilcoxon_test_and_plot(right_df, left_df, f'Test de Wilcoxon pour {param.upper()}')

        plot_plan_theatral(right_df, f'{param.upper()} Pied droit')
        plot_plan_theatral(left_df, f'{param.upper()} Pied gauche')

        # Simulation d'épaisseur (dummy data)
        dummy_thickness = np.random.rand(len(right_df))
        correlation_plots(right_df, dummy_thickness)
