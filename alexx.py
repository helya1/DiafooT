import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, probplot
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📋 Analyse Myoton")
page = st.sidebar.radio("Choisir la page :", ["Statistiques par zone", "Test de normalité", "plot_left_right_comparison", "plot planck_hartmann"])

uploaded_file = st.file_uploader("📂 Charger le fichier Excel", type=["xlsx"])

if uploaded_file:
    sheet_name = "Manips resultats"
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
    index_map = {
        'tone_right':      [i-1 for i in [3,4,5,23,24,25,43,44,45,63,64,65,83,84,85,103,104,105,123,124,125,143,144,145,163,164,165]],
        'tone_left':       [i-1 for i in [13,14,15,33,34,35,53,54,55,73,74,75,93,94,95,113,114,115,133,134,135,153,154,161,173,174,175]],
        'stiffness_right': [i-1 for i in [6,7,8,26,27,28,46,47,48,66,67,68,86,87,88,106,107,108,126,127,128,146,147,148,166,167,168]],
        'stiffness_left':  [i-1 for i in [16,17,18,36,37,38,56,57,58,76,77,78,96,97,98,116,117,118,136,137,138,156,157,158,176,177,178]],
        'frequency_right': [i-1 for i in [9,10,11,29,30,31,49,50,51,69,70,71,89,90,91,109,110,111,129,130,131,149,150,151,169,170,171]],
        'frequency_left':  [i-1 for i in [19,20,21,39,40,41,59,60,61,79,80,81,99,100,101,119,120,121,139,140,141,159,160,161,179,180,181]]
    }

    foot_zones = ['Hallux', '1T', '2-3T', '5T', 'Voute int', 'Voute ext', 'Talon']

    parameters = [
        ("Tone - Pied Droit",      'tone_right'),
        ("Tone - Pied Gauche",     'tone_left'),
        ("Stiffness - Pied Droit", 'stiffness_right'),
        ("Stiffness - Pied Gauche",'stiffness_left'),
        ("Frequency - Pied Droit", 'frequency_right'),
        ("Frequency - Pied Gauche",'frequency_left')
    ]

    def extract_param(index_list):
        return df.iloc[index_list].reset_index(drop=True).T.iloc[:, :7]

    if page == "Statistiques par zone":
        st.header("📊 Statistiques par zone")
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

        parameters = [
            ("Tone - Pied Droit",      'tone_right'),
            ("Tone - Pied Gauche",     'tone_left'),
            ("Stiffness - Pied Droit", 'stiffness_right'),
            ("Stiffness - Pied Gauche",'stiffness_left'),
            ("Frequency - Pied Droit", 'frequency_right'),
            ("Frequency - Pied Gauche",'frequency_left')
        ]

        for label, key in parameters:
            st.markdown(f"### 📌 {label}")
            data = extract_param(index_map[key])
            stats_df = compute_statistics_by_zone(data, label)
            st.dataframe(stats_df)

            st.markdown("ℹ️ **Interprétation des CV :**")
            st.markdown("- ✅ **CV < 10%** : faible variabilité, la moyenne est représentative")
            st.markdown("- ⚠️ **CV entre 10% et 20%** : variabilité modérée, interprétation avec prudence")
            st.markdown("- ❌ **CV > 20%** : forte variabilité, la moyenne est peu fiable")


    # --- Partie 2 : Test de normalité de Shapiro-Wilk ---       
    elif page == "Test de normalité":
        st.header("🧪 Normalité par zone, paramètre et côté :(Shapiro-Wilk)")

        def test_normality_only(data, param_name, foot_zones):
            results = []
            for i, zone in enumerate(foot_zones):
                values = pd.to_numeric(data.iloc[:, i], errors='coerce').dropna()

                if len(values) >= 3:
                    stat, p_value = shapiro(values)
                else:
                    stat, p_value = np.nan, np.nan

                results.append({
                    "Zone": zone,
                    "p-value": p_value,
                    "Conclusion": "✅ Normale" if pd.notna(p_value) and p_value > 0.05 else "❌ Non normale"
                })

            result_df = pd.DataFrame(results)
            st.write(f"📋 Résultats du test de Shapiro-Wilk pour : {param_name}")
            st.dataframe(result_df)

        for label, key in parameters:
            st.markdown(f"### 🔍 {label}")
            data = extract_param(index_map[key])
            test_normality_only(data, label, foot_zones)

        st.markdown("""
        ### 🧠 Interprétation
        Ce tableau affiche la normalité **pour chaque combinaison : zone du pied × paramètre × côté**.

        - ✅ signifie que les données suivent une distribution normale (*tests paramétriques possibles*)
        - ❌ signifie que les données ne sont pas normales (*tests non paramétriques recommandés*)

        Le **test de Shapiro-Wilk** permet de vérifier si les données suivent une **loi normale**.  
        - Si la *p-value > 0.05*, on considère que les données sont **normalement distribuées** (*test paramétrique possible* : t-test, ANOVA).  
        - Si la *p-value ≤ 0.05*, les données **ne suivent pas une loi normale** (*test non paramétrique recommandé* : Wilcoxon, Friedman).
        """)

    
    elif page == "plot_left_right_comparison":
        st.header("📊 Comparaison Gauche vs Droit – par Paramètre et Zone")

        def plot_left_right_comparison(param_key_right, param_key_left, param_name, foot_zones, df, index_map):
            data_right = extract_param(index_map[param_key_right])
            data_left = extract_param(index_map[param_key_left])

            for i, zone in enumerate(foot_zones):

                vals_right = pd.to_numeric(data_right.iloc[:, i], errors='coerce').dropna()
                vals_left = pd.to_numeric(data_left.iloc[:, i], errors='coerce').dropna()

                # Q-Q Plots
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"{param_name} – Zone : {zone}", fontsize=16)

                probplot(vals_right, dist="norm", plot=axs[0])
                axs[0].set_title("Q-Q Plot Pied Droit")
                axs[0].grid(True)

                probplot(vals_left, dist="norm", plot=axs[1])
                axs[1].set_title("Q-Q Plot Pied Gauche")
                axs[1].grid(True)

                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bp_right = ax2.boxplot(vals_right, positions=[1], widths=0.6, patch_artist=True,
                                    boxprops=dict(facecolor="lightblue"))
                bp_left = ax2.boxplot(vals_left, positions=[2], widths=0.6, patch_artist=True,
                                    boxprops=dict(facecolor="lightgreen"))

                ax2.set_xticks([1, 2])
                ax2.set_xticklabels(["Pied Droit", "Pied Gauche"])
                ax2.set_title(f"Boxplot comparatif – {param_name} – {zone}")
                ax2.grid(True)

                st.pyplot(fig2)

        for param_name, key_right, key_left in [
            ("Tone", 'tone_right', 'tone_left'),
            ("Stiffness", 'stiffness_right', 'stiffness_left'),
            ("Frequency", 'frequency_right', 'frequency_left')
        ]:
            st.subheader(f"🔹 {param_name}")
            plot_left_right_comparison(key_right, key_left, param_name, foot_zones, df, index_map)

        
    elif page == "plot planck_hartmann":
        st.header("🎭 Planck-Hartmann")
        foot_zones = ['Hallux', '1T', '2-3T', '5T', 'Voute int', 'Voute ext', 'Talon']
        def plot_planck_hartmann(right_df, left_df, param_name):
            right_numeric = right_df.apply(pd.to_numeric, errors='coerce')
            left_numeric = left_df.apply(pd.to_numeric, errors='coerce')

            # Centrer les données par zone (différence à la moyenne par zone)
            centered_right = right_numeric - right_numeric.mean()
            centered_left = left_numeric - left_numeric.mean()

            # Calcul moyenne générale (droite + gauche)
            mean_global = pd.concat([right_numeric, left_numeric]).mean()

            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(foot_zones))
            width = 0.3

            # Tracer chaque individu (droite)
            for i in range(len(centered_right)):
                ax.plot(x - width/2, centered_right.iloc[i], marker='o', linestyle='-', color='blue', alpha=0.6, label="Individu Pied Droit" if i == 0 else "")

            # Tracer chaque individu (gauche)
            for i in range(len(centered_left)):
                ax.plot(x + width/2, centered_left.iloc[i], marker='o', linestyle='--', color='orange', alpha=0.6, label="Individu Pied Gauche" if i == 0 else "")

            # Tracer la moyenne générale (droite + gauche) centrée (ligne épaisse et couleur distincte)
            mean_centered = mean_global - mean_global.mean()
            ax.plot(x, mean_centered, marker='s', linestyle='-', color='green', linewidth=3, label="Moyenne Générale", alpha=0.8)

            # Ligne horizontale 0
            ax.axhline(0, color='red', linestyle='--', label="Moyenne zone = 0")

            ax.set_xticks(x)
            ax.set_xticklabels(foot_zones, fontsize=12)
            ax.set_ylabel("Valeur centrée (différence à la moyenne)")
            ax.set_title(f"Paramètre : {param_name.upper()} (centré par zone)")
            ax.legend(loc='upper right')
            ax.grid(True)

            st.pyplot(fig)

            # Calcul IC 95% global par côté (moyenne des std divisée par racine n)
            ci_right = 1.96 * right_numeric.std().mean() / np.sqrt(len(right_numeric))
            ci_left = 1.96 * left_numeric.std().mean() / np.sqrt(len(left_numeric))
            st.info(f"Intervalle de confiance 95% estimé : Pied droit ±{ci_right:.2f}, Pied gauche ±{ci_left:.2f}")

        # --- Simulation données (à remplacer par tes extractions réelles) ---
        np.random.seed(42)
        n_subjects = 15

        # Générer données factices pour chaque paramètre, 15 sujets, 7 zones
        df_tone_right = pd.DataFrame(np.random.normal(50, 10, (n_subjects, 7)), columns=foot_zones)
        df_tone_left = pd.DataFrame(np.random.normal(48, 9, (n_subjects, 7)), columns=foot_zones)

        df_stiffness_right = pd.DataFrame(np.random.normal(30, 5, (n_subjects, 7)), columns=foot_zones)
        df_stiffness_left = pd.DataFrame(np.random.normal(32, 4, (n_subjects, 7)), columns=foot_zones)

        df_frequency_right = pd.DataFrame(np.random.normal(20, 3, (n_subjects, 7)), columns=foot_zones)
        df_frequency_left = pd.DataFrame(np.random.normal(18, 3, (n_subjects, 7)), columns=foot_zones)

        # Affichage
        st.header("🎭 Planck-Hartmann - Comparaison Pied Droit / Pied Gauche")

        plot_planck_hartmann(df_tone_right, df_tone_left, "Tone")
        plot_planck_hartmann(df_stiffness_right, df_stiffness_left, "Stiffness")
        plot_planck_hartmann(df_frequency_right, df_frequency_left, "Frequency")
        
        
        # --- Interprétation du Plot Planck-Hartmann ---
        st.markdown("### 🧠 Interprétation des graphiques Planck-Hartmann")
        st.markdown("""
        - **Les courbes bleues** représentent les valeurs centrées pour chaque individu au pied droit,  
        tandis que les **courbes orange** représentent celles pour le pied gauche.  
        - Les valeurs sont centrées par zone, c’est-à-dire que l’on montre l’écart à la moyenne de chaque zone,  
        ce qui permet de visualiser les différences relatives de chaque sujet selon la zone du pied.

        - La **courbe verte épaisse** correspond à la moyenne générale (droite + gauche) de tous les sujets,  
        permettant d’évaluer la tendance moyenne globale pour chaque zone.

        - La **ligne rouge en pointillé** indique la référence zéro (moyenne par zone).  
        Des points au-dessus indiquent des valeurs supérieures à la moyenne,  
        en dessous des valeurs inférieures.

        - Les intervalles de confiance 95% (affichés en info) donnent une idée de la variabilité et de la précision de la moyenne.

        ### Comment interpréter ?

        - Si les courbes individuelles (bleues et oranges) sont proches de la moyenne verte, cela signifie une homogénéité des profils entre sujets.  
        - De fortes variations individuelles peuvent indiquer une variabilité importante à prendre en compte.  
        - Des différences systématiques entre pied droit et gauche (décalage entre les courbes bleues et oranges) peuvent révéler des asymétries biomécaniques ou fonctionnelles.

        Cette visualisation permet ainsi d’identifier facilement les zones du pied où la variabilité ou les asymétries sont les plus marquées, et de guider des analyses ou interventions ciblées.
        """)
