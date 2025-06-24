import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, probplot
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📋 Analyse Myoton")

uploaded_file = st.file_uploader("📂 Charger le fichier Excel", type=["xlsx"])

if uploaded_file:
    sheet_name = "Manips resultats"
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

    # Indices des lignes dans le fichier Excel
    index_map = {
        'tone_right':      [i-1 for i in [3,4,5,23,24,25,43,44,45,63,64,65,83,84,85,103,104,105,123,124,125,143,144,145,163,164,165]],
        'tone_left':       [i-1 for i in [13,14,15,33,34,35,53,54,55,73,74,75,93,94,95,113,114,115,133,134,135,153,154,161,173,174,175]],
        'stiffness_right': [i-1 for i in [6,7,8,26,27,28,46,47,48,66,67,68,86,87,88,106,107,108,126,127,128,146,147,148,166,167,168]],
        'stiffness_left':  [i-1 for i in [16,17,18,36,37,38,56,57,58,76,77,78,96,97,98,116,117,118,136,137,138,156,157,158,176,177,178]],
        'frequency_right': [i-1 for i in [9,10,11,29,30,31,49,50,51,69,70,71,89,90,91,109,110,111,129,130,131,149,150,151,169,170,171]],
        'frequency_left':  [i-1 for i in [19,20,21,39,40,41,59,60,61,79,80,81,99,100,101,119,120,121,139,140,141,159,160,161,179,180,181]]
    }

    # Noms des zones du pied
    foot_zones = ['Hallux', '1T', '2-3T', '5T', 'Voute int', 'Voute ext', 'Talon']

    # Fonction pour extraire les données d’un paramètre
    def extract_param(index_list):
        return df.iloc[index_list].reset_index(drop=True).T.iloc[:, :7]

    # Fonction avec INTERPRÉTATION ajoutée
    def compute_statistics_by_zone(data, param_name):
        stats = []
        for i, zone in enumerate(foot_zones):
            values = pd.to_numeric(data.iloc[:, i], errors='coerce')
            mean = values.mean()
            std = values.std()
            cv = std / mean if mean != 0 else np.nan

            if pd.isna(cv):
                interpretation = "Données insuffisantes"
            elif cv < 0.10:
                interpretation = "✅ Bonne représentativité (CV < 10%)"
            elif cv < 0.20:
                interpretation = "⚠️ Variabilité modérée (10% ≤ CV < 20%)"
            else:
                interpretation = "❌ Forte variabilité (CV ≥ 20%)"

            stats.append({
                "Zone": zone,
                "Moyenne": mean,
                "Écart-type": std,
                "CV (%)": f"{cv * 100:.2f}" if pd.notnull(cv) else "N/A",
                "Interprétation": interpretation
            })
        return pd.DataFrame(stats)

    # Liste des paramètres à afficher
    parameters = [
        ("Tone - Pied Droit",      'tone_right'),
        ("Tone - Pied Gauche",     'tone_left'),
        ("Stiffness - Pied Droit", 'stiffness_right'),
        ("Stiffness - Pied Gauche",'stiffness_left'),
        ("Frequency - Pied Droit", 'frequency_right'),
        ("Frequency - Pied Gauche",'frequency_left')
    ]

    # Affichage de tous les tableaux sur la même page
    for label, key in parameters:
        st.markdown(f"### 📌 {label}")
        data = extract_param(index_map[key])
        stats_df = compute_statistics_by_zone(data, label)
        st.dataframe(stats_df)

        # Interprétation générale
        st.markdown("ℹ️ **Interprétation des CV :**")
        st.markdown("- ✅ **CV < 10%** : faible variabilité, la moyenne est représentative")
        st.markdown("- ⚠️ **CV entre 10% et 20%** : variabilité modérée, interprétation avec prudence")
        st.markdown("- ❌ **CV > 20%** : forte variabilité, la moyenne est peu fiable")


    # --- Partie 2 : Test de normalité de Shapiro-Wilk ---
    st.header("🧪 Normalité par zone, paramètre et côté : Test de normalité (Shapiro-Wilk)")
    def test_normality_and_plot(data, param_name, foot_zones):
        results = []

        for i, zone in enumerate(foot_zones):
            values = pd.to_numeric(data.iloc[:, i], errors='coerce').dropna()

            if len(values) >= 3:  # Shapiro nécessite au moins 3 points
                stat, p_value = shapiro(values)
            else:
                stat, p_value = np.nan, np.nan

            results.append({
                "Zone": zone,
                "p-value": p_value,
                "Conclusion": "✅ Normale" if p_value > 0.05 else "❌ Non normale"
            })

            # Affichage des graphiques (Q-Q plot + boxplot)
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f'{param_name} – {zone}', fontsize=14)

            # Q-Q plot
            probplot(values, dist="norm", plot=axs[0])
            axs[0].set_title("Q-Q Plot")

            # Boxplot
            axs[1].boxplot(values, vert=True)
            axs[1].set_title("Boxplot")
            axs[1].set_ylabel("Valeur")

            st.pyplot(fig)

        result_df = pd.DataFrame(results)
        st.write(f"📋 Résultats du test de Shapiro-Wilk pour : {param_name}")
        st.dataframe(result_df)

    # Appel pour tous les paramètres et les deux côtés
    for label, key in parameters:
        st.markdown(f"### 🔍 {label}")
        data = extract_param(index_map[key])
        test_normality_and_plot(data, label, foot_zones)

    st.markdown("""
    ### 🧠 Interprétation

    Ce tableau affiche la normalité **pour chaque combinaison : zone du pied × paramètre × côté**.

    - ✅ signifie que les données suivent une distribution normale (*tests paramétriques possibles*)
    - ❌ signifie que les données ne sont pas normales (*tests non paramétriques recommandés*)
    """)


    # Explication en français
    st.markdown("### 🧠 Interprétation")
    st.markdown("""
    Le **test de Shapiro-Wilk** permet de vérifier si les données suivent une **loi normale**.  
    - Si la *p-value > 0.05*, on considère que les données sont **normalement distribuées** (*test paramétrique possible* : t-test, ANOVA).  
    - Si la *p-value ≤ 0.05*, les données **ne suivent pas une loi normale** (*test non paramétrique recommandé* : Wilcoxon, Friedman).

    """)





    def plot_planck_hartmann(right_df, left_df, param_name):
        st.markdown(f"### 🎭 Planck-Hartmann: {param_name.upper()}")
        right_numeric = right_df.apply(pd.to_numeric, errors='coerce')
        left_numeric = left_df.apply(pd.to_numeric, errors='coerce')

        centered_right = right_numeric - right_numeric.mean()
        centered_left = left_numeric - left_numeric.mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(foot_zones))
        width = 0.3

        for i in range(len(centered_right)):
            ax.plot(x - width/2, centered_right.iloc[i], marker='o', linestyle='-', color='blue', alpha=0.6)
            ax.plot(x + width/2, centered_left.iloc[i], marker='o', linestyle='--', color='orange', alpha=0.6)

        ax.axhline(0, color='red', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(foot_zones)
        ax.set_ylabel("Valeur centrée (diff. à la moyenne)")
        ax.set_title(f"Paramètre: {param_name.upper()} (centré par zone)")
        ax.legend(["Pied droit", "Pied gauche", "Moyenne globale = 0"], loc='upper right')
        st.pyplot(fig)

        # Intervalle de confiance (CI95) global pour info
        ci_right = 1.96 * right_numeric.std().mean() / np.sqrt(len(right_numeric))
        ci_left = 1.96 * left_numeric.std().mean() / np.sqrt(len(left_numeric))
        st.info(f"CI 95% global estimé: Pied droit ±{ci_right:.2f}, Pied gauche ±{ci_left:.2f}")

    def test_normality_and_plot(param_df, title):
        st.markdown(f"### 📈 Test de normalité - {title}")
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
            ax.legend()
            st.pyplot(fig)

    # --- Exécution pour chaque paramètre ---
    def wilcoxon_test_by_zone(df_right, df_left, param_name):
        st.markdown(f"### 🧪 Test de Wilcoxon - {param_name.upper()}")
        p_values = []
        results = []

        for col in df_right.columns:
            x = pd.to_numeric(df_right[col], errors='coerce')
            y = pd.to_numeric(df_left[col], errors='coerce')
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                st.warning(f"🚫 Pas assez de données pour la zone {col}")
                continue
            stat, p = stats.wilcoxon(x[valid], y[valid])
            p_values.append(p)
            results.append({"Zone": col, "Wilcoxon p-valeur": f"{p:.4f}", "Significatif": "✅" if p < 0.05 else "❌"})

        df_results = pd.DataFrame(results)
        st.dataframe(df_results)

        fig, ax = plt.subplots(figsize=(10, 5))
        df_box = pd.concat([
            pd.DataFrame({"Valeur": pd.to_numeric(df_right[col], errors='coerce'), "Pied": "Droit", "Zone": col}) for col in df_right.columns
        ] + [
            pd.DataFrame({"Valeur": pd.to_numeric(df_left[col], errors='coerce'), "Pied": "Gauche", "Zone": col}) for col in df_left.columns
        ])
        df_box = df_box.reset_index(drop=True)
        sns.boxplot(data=df_box, x="Zone", y="Valeur", hue="Pied", ax=ax)
        ax.set_title(f"Comparaison Wilcoxon - {param_name.upper()}")
        st.pyplot(fig)
    for param in ['tone', 'stiffness', 'frequency']:
        df_right = extract_param(index_map[f'{param}_right'])
        df_left = extract_param(index_map[f'{param}_left'])
        df_right.columns = foot_zones
        df_left.columns = foot_zones

        plot_planck_hartmann(df_right, df_left, param)
        test_normality_and_plot(df_right, f"{param.upper()} - Pied droit")
        test_normality_and_plot(df_left, f"{param.upper()} - Pied gauche")


        wilcoxon_test_by_zone(df_right, df_left, param)  # ✅ ici


    def plot_correlations_by_zone(df_right, df_left, param_name):
        st.markdown(f"### 🔗 Corrélations droite vs gauche - {param_name.upper()}")
        
        for zone in foot_zones:
            x = pd.to_numeric(df_right[zone], errors='coerce')
            y = pd.to_numeric(df_left[zone], errors='coerce')
            valid = x.notna() & y.notna()
            x_valid = x[valid]
            y_valid = y[valid]

            if len(x_valid) < 3:
                st.warning(f"🚫 Pas assez de données pour la zone {zone} pour calculer la corrélation.")
                continue

            # Calcul R et p
            r, p = stats.pearsonr(x_valid, y_valid)

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(x_valid, y_valid, alpha=0.6, color='purple')
            sns.regplot(x=x_valid, y=y_valid, ax=ax, scatter=False, color='red')
            ax.set_xlabel(f"{zone} - Pied droit")
            ax.set_ylabel(f"{zone} - Pied gauche")
            ax.set_title(f"Corrélation {param_name.upper()} - Zone {zone}")

            # Affichage R et p
            ax.text(0.05, 0.95,
                    f"R = {r:.2f}\nP = {p:.5f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            st.pyplot(fig)
            
        plot_correlations_by_zone(df_right, df_left, param)  # ✅ ici
        st.success("### ✅ Analyse terminée")
