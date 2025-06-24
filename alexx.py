import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, probplot, ttest_rel, wilcoxon, friedmanchisquare # Added friedmanchisquare
import matplotlib.pyplot as plt
import io
from statsmodels.formula.api import ols # Not directly used in the provided snippets but often useful with AnovaRM
from statsmodels.stats.anova import AnovaRM # Added AnovaRM


st.set_page_config(layout="wide")
st.title("📋 Analyse Myoton")
page = st.sidebar.radio("Choisir la page :", ["Statistiques par zone", "Test de normalité", "plot_left_right_comparison", "plot planck_hartmann", "Moyenne répétitions", "test_symmetry_with_parametric_choice", "Test inter-zone (paramétrique/non-paramétrique)"])

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
        df_param = df.iloc[index_list].reset_index(drop=True).T.iloc[:, :7]
        df_param.columns = foot_zones  # 👈 ici on renomme les colonnes
        return df_param


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

        st.markdown("""
        **Objectif :** Vérifier si les données suivent une distribution normale dans chaque zone.
        """)

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


        def test_symmetry_with_normality(key_right, key_left, param_name, foot_zones):
            data_right = extract_param(index_map[key_right])
            data_left = extract_param(index_map[key_left])

            results = []
            for i, zone in enumerate(foot_zones):
                vals_right = pd.to_numeric(data_right.iloc[:, i], errors='coerce').dropna()
                vals_left = pd.to_numeric(data_left.iloc[:, i], errors='coerce').dropna()

                # Indices communs pour test apparié
                common_idx = vals_right.index.intersection(vals_left.index)
                vals_right = vals_right.loc[common_idx]
                vals_left = vals_left.loc[common_idx]

                # Test normalité
                norm_right = len(vals_right) >= 3 and shapiro(vals_right).pvalue > 0.05
                norm_left = len(vals_left) >= 3 and shapiro(vals_left).pvalue > 0.05

                if norm_right and norm_left:
                    stat, p_value = ttest_rel(vals_right, vals_left)
                    test_used = "Test t apparié"
                else:
                    try:
                        stat, p_value = wilcoxon(vals_right, vals_left)
                        test_used = "Test de Wilcoxon"
                    except ValueError:
                        stat, p_value = np.nan, np.nan
                        test_used = "Test de Wilcoxon (impossible)"

                results.append({
                    "Zone": zone,
                    "Test utilisé": test_used,
                    "Statistique": stat,
                    "p-value": p_value,
                    "Conclusion": "❌ Différence significative" if pd.notna(p_value) and p_value < 0.05 else "✅ Pas de différence significative"
                })

            df_results = pd.DataFrame(results)
            st.write(f"📋 Résultats des tests entre pied droit et gauche pour : {param_name}")
            st.dataframe(df_results)


        # Affichage normalité simple
        for label, key in parameters:
            st.markdown(f"### 🔍 {label}")
            data = extract_param(index_map[key])
            test_normality_only(data, label, foot_zones)

        # Test de comparaison symétrie avec test t ou Wilcoxon selon normalité
        for param_name, key_right, key_left in [
            ("Tone", "tone_right", "tone_left"),
            ("Stiffness", "stiffness_right", "stiffness_left"),
            ("Frequency", "frequency_right", "frequency_left")
        ]:
            test_symmetry_with_normality(key_right, key_left, param_name, foot_zones)


        st.markdown("""
        ### 🧠 Interprétation
        Ce tableau affiche la normalité **pour chaque combinaison : zone du pied × paramètre × côté**.

        - ✅ signifie que les données suivent une distribution normale (*tests paramétriques possibles*)
        - ❌ signifie que les données ne sont pas normales (*tests non paramétriques recommandés*)

        Le **test de Shapiro-Wilk** permet de vérifier si les données suivent une **loi normale**.  
        - Si la *p-value > 0.05*, on considère que les données sont **normalement distribuées** (*test paramétrique possible* : t-test, ANOVA).  
        - Si la *p-value ≤ 0.05*, les données **ne suivent pas une loi normale** (*test non paramétrique recommandé* : Wilcoxon, Friedman).

        Pour comparer pied droit vs pied gauche,  
        - On effectue un test t apparié si les deux côtés ont une distribution normale (Shapiro-Wilk p>0.05).  
        - Sinon on fait un test de Wilcoxon apparié (non paramétrique).  
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
            
            # --- Partie téléchargement ---
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="📥 Télécharger le graphique (PNG)",
                data=buf,
                file_name=f"Planck_Hartmann_{param_name}.png",
                mime="image/png"
            )

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

    elif page == "Moyenne répétitions":
        st.header("📊 Analyse avec moyenne des répétitions par zone, pied droit et pied gauche")

        def plot_planck_hartmann_mean_repetitions(right_df, left_df, param_name, foot_zones):
            st.markdown(f"### 🎭 Planck-Hartmann : {param_name.upper()}")

            right_numeric = right_df.apply(pd.to_numeric, errors='coerce')
            left_numeric = left_df.apply(pd.to_numeric, errors='coerce')

            # Moyenne des répétitions par zone (axe=0 = colonne)
            mean_right = right_numeric.mean(axis=0)
            mean_left = left_numeric.mean(axis=0)

            # Centrer chaque répétition par rapport à sa moyenne zone
            centered_right = right_numeric.subtract(mean_right, axis=1)
            centered_left = left_numeric.subtract(mean_left, axis=1)

            # Moyenne globale (droite + gauche)
            mean_global = pd.concat([right_numeric, left_numeric]).mean(axis=0)
            mean_global_centered = mean_global - mean_global.mean()

            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(foot_zones))
            width = 0.3

            # Tracer répétitions pied droit
            for i in range(len(centered_right)):
                ax.plot(x - width/2, centered_right.iloc[i], marker='o', linestyle='-', color='blue', alpha=0.6, label="Répétition Pied Droit" if i == 0 else "")

            # Tracer répétitions pied gauche
            for i in range(len(centered_left)):
                ax.plot(x + width/2, centered_left.iloc[i], marker='o', linestyle='--', color='orange', alpha=0.6, label="Répétition Pied Gauche" if i == 0 else "")

            # Tracer moyenne globale
            ax.plot(x, mean_global_centered, marker='s', linestyle='-', color='green', linewidth=3, label="Moyenne Générale", alpha=0.8)

            ax.axhline(0, color='red', linestyle='--', label="Moyenne zone = 0")
            ax.set_xticks(x)
            ax.set_xticklabels(foot_zones)
            ax.set_ylabel("Valeur centrée (différence à la moyenne)")
            ax.set_title(f"Paramètre : {param_name.upper()} (centré par zone)")
            ax.legend(loc='upper right')
            ax.grid(True)
            
            # --- Partie téléchargement ---
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="📥 Télécharger le graphique (PNG)",
                data=buf,
                file_name=f"Planck_Hartmann_{param_name}.png",
                mime="image/png"
            )

            st.pyplot(fig)

            ci_right = 1.96 * right_numeric.std().mean() / np.sqrt(len(right_numeric))
            ci_left = 1.96 * left_numeric.std().mean() / np.sqrt(len(left_numeric))
            st.info(f"Intervalle de confiance 95% estimé : Pied droit ±{ci_right:.2f}, Pied gauche ±{ci_left:.2f}")

        # Liste des paramètres à analyser et afficher
        parameters = [
            ("Tone", "tone_right", "tone_left"),
            ("Stiffness", "stiffness_right", "stiffness_left"),
            ("Frequency", "frequency_right", "frequency_left")
        ]

        for param_name, key_right, key_left in parameters:
            st.markdown(f"## Analyse du paramètre : {param_name}")
            data_right = extract_param(index_map[key_right])
            data_left = extract_param(index_map[key_left])
            plot_planck_hartmann_mean_repetitions(data_right, data_left, param_name, foot_zones)


    elif page == "test_symmetry_with_parametric_choice":       
        def test_symmetry_with_parametric_choice(param_key_right, param_key_left, foot_zones):
            data_right = extract_param(index_map[param_key_right])
            data_left = extract_param(index_map[param_key_left])

            results = []

            for i, zone in enumerate(foot_zones):
                vals_right = pd.to_numeric(data_right.iloc[:, i], errors='coerce').dropna()
                vals_left = pd.to_numeric(data_left.iloc[:, i], errors='coerce').dropna()

                # Indices communs pour test apparié
                common_idx = vals_right.index.intersection(vals_left.index)
                vals_right = vals_right.loc[common_idx]
                vals_left = vals_left.loc[common_idx]

                # Vérification normalité avec Shapiro-Wilk (min 3 données)
                norm_right = len(vals_right) >= 3 and shapiro(vals_right).pvalue > 0.05
                norm_left = len(vals_left) >= 3 and shapiro(vals_left).pvalue > 0.05

                # Choix du test selon normalité
                if norm_right and norm_left:
                    stat, p_value = ttest_rel(vals_right, vals_left)
                    test_used = "Test t apparié (paramétrique)"
                else:
                    try:
                        stat, p_value = wilcoxon(vals_right, vals_left)
                        test_used = "Test de Wilcoxon (non paramétrique)"
                    except ValueError:
                        stat, p_value = np.nan, np.nan
                        test_used = "Test de Wilcoxon (impossible)"

                # Interprétation du résultat
                if pd.isna(p_value):
                    conclusion = "⚠️ Données insuffisantes"
                elif p_value < 0.05:
                    conclusion = "❌ Différence significative"
                else:
                    conclusion = "✅ Pas de différence significative"

                results.append({
                    "Zone": zone,
                    "Test utilisé": test_used,
                    "Statistique": round(stat, 4) if not pd.isna(stat) else np.nan,
                    "p-value": round(p_value, 4) if not pd.isna(p_value) else np.nan,
                    "Conclusion": conclusion
                })

            return pd.DataFrame(results)

        # --- Intégration dans Streamlit ---

        page = st.sidebar.radio("Choisir la page :", ["Test de symétrie Gauche/Droite", "Autres pages..."])

        if uploaded_file:  # Ton code d'import des données

            if page == "Test de symétrie Gauche/Droite":
                st.header("⚖️ Test de symétrie Gauche vs Droit (t-test ou Wilcoxon selon normalité)")

                for param_name, key_right, key_left in [
                    ("Tone", "tone_right", "tone_left"),
                    ("Stiffness", "stiffness_right", "stiffness_left"),
                    ("Frequency", "frequency_right", "frequency_left")
                ]:
                    st.subheader(f"🔹 {param_name}")
                    df_results = test_symmetry_with_parametric_choice(key_right, key_left, foot_zones)
                    st.dataframe(df_results)

                st.markdown("""
                **Interprétation :**  
                - Si **p-value < 0.05**, il existe une différence significative entre pied droit et gauche pour cette zone et paramètre.  
                - Si **p-value ≥ 0.05**, pas de différence significative (symétrie).  
                - Le test utilisé est choisi selon la normalité des données (test de Shapiro-Wilk) :  
                - Test t apparié si données normales (paramétrique)  
                - Test de Wilcoxon apparié sinon (non paramétrique)
                """)


    elif page == "Test inter-zone (paramétrique/non-paramétrique)": # Consolidated and renamed page
        st.header("🧪 Comparaison inter-zones (ANOVA RM / Friedman)")

        # Simplified function for the main page display
        def test_inter_zone_summary(param_key, param_name, foot_zones):
            data = extract_param(index_map[param_key])
            # Ensure data is numeric and drop NaNs for analysis. Important for 'per subject' analysis later.
            data_num = data.apply(pd.to_numeric, errors='coerce').dropna()

            if data_num.empty:
                st.warning(f"Pas assez de données complètes pour analyser {param_name}.")
                return

            # Test de normalité sur chaque zone
            normalities = []
            for col in data_num.columns: # Iterate over the actual columns of the cleaned data_num
                vals = data_num[col].dropna()
                if len(vals) >= 3:
                    p_val = shapiro(vals).pvalue
                else:
                    p_val = np.nan
                normalities.append(p_val)

            all_normal = all([(p > 0.05) for p in normalities if not pd.isna(p)])

            st.write(f"### Résultats pour : {param_name}")
            norm_df = pd.DataFrame({
                "Zone": foot_zones, # Use global foot_zones for display, assuming data_num columns map to it
                "p-value normalité": [f"{p:.4f}" if not pd.isna(p) else "N/A" for p in normalities],
                "Distribution": ["Normale" if (p > 0.05) else "Non normale" if not pd.isna(p) else "N/A" for p in normalities]
            })
            st.dataframe(norm_df)

            # Prepare data for ANOVA RM or Friedman: long format for AnovaRM, list of arrays for Friedman
            df_long = data_num.reset_index().melt(id_vars='index', value_vars=data_num.columns,
                                                     var_name='Zone', value_name='Valeur')
            df_long = df_long.rename(columns={'index': 'Sujet'})

            if all_normal:
                st.write("✅ Toutes les zones considérées comme normales. Effectue une ANOVA à mesures répétées.")
                try:
                    # Ensure 'Sujet' and 'Zone' are strings as AnovaRM expects
                    df_long['Sujet'] = df_long['Sujet'].astype(str)
                    df_long['Zone'] = df_long['Zone'].astype(str)
                    aovrm = AnovaRM(df_long, 'Valeur', 'Sujet', within=['Zone'])
                    res = aovrm.fit()
                    st.text(res.summary().as_text()) # Use as_text() for better display in Streamlit
                    p_val = res.anova_table["Pr > F"][0]

                    if p_val < 0.05:
                        st.success(f"Différence significative entre les zones (p = {p_val:.4f})")
                    else:
                        st.info(f"Aucune différence significative détectée entre les zones (p = {p_val:.4f})")
                except Exception as e:
                    st.error(f"Erreur lors de l'ANOVA à mesures répétées : {e}")
                    st.warning("Assurez-vous que chaque sujet a des données pour toutes les zones pour l'ANOVA RM.")
            else:
                st.write("❌ Au moins une zone ne suit pas une distribution normale. Effectue un test de Friedman.")
                # Friedman takes *args, each arg is a data array for a group/zone
                vals_for_friedman = [data_num[col].dropna().values for col in data_num.columns]
                # Friedman requires at least 3 groups, and each group needs valid data
                if len(vals_for_friedman) < 3 or any(len(v) == 0 for v in vals_for_friedman):
                    st.warning("Pas assez de groupes ou de données valides par groupe pour le test de Friedman.")
                else:
                    try:
                        stat, p_val = friedmanchisquare(*vals_for_friedman)
                        st.write(f"Statistique Friedman : {stat:.4f}")
                        if p_val < 0.05:
                            st.success(f"Différence significative entre les zones (p = {p_val:.4f})")
                        else:
                            st.info(f"Aucune différence significative détectée entre les zones (p = {p_val:.4f})")
                    except Exception as e:
                        st.error(f"Erreur lors du test de Friedman : {e}")
                        st.warning("Vérifiez que toutes les zones ont des observations appariées (même nombre de sujets avec des données complètes pour toutes les zones).")

        # Loop through parameters (Tone, Stiffness, Frequency) for both right and left feet
        for param_name, key_right, key_left in parameters_comparison_keys: # Use comparison keys as they hold (param_name, right_key, left_key)
            st.markdown(f"## Analyse des zones pour {param_name}")
            st.subheader(f"Pied Droit - {param_name}")
            test_inter_zone_summary(key_right, f"{param_name} (Droit)", foot_zones) # Pass param_name with side
            st.subheader(f"Pied Gauche - {param_name}")
            test_inter_zone_summary(key_left, f"{param_name} (Gauche)", foot_zones) # Pass param_name with side

            st.markdown("---")
            st.markdown("### 🧠 Interprétation des tests statistiques inter-zones")

            st.markdown("""
            - **But de ces tests** : déterminer si les différentes **zones du pied** présentent des **valeurs significativement différentes** pour un paramètre donné (tone, stiffness, frequency, etc.).

            #### 🔬 ANOVA à mesures répétées (test paramétrique)
            - Utilisé lorsque **toutes les zones ont une distribution normale** (test de Shapiro-Wilk p > 0.05).
            - Ce test compare les moyennes entre les zones, en tenant compte des mesures répétées sur les mêmes sujets.
            - **Interprétation :**
            - Si *p-value < 0.05* → **différences significatives** entre au moins deux zones.
            - Si *p-value ≥ 0.05* → **pas de différence significative**, le paramètre est distribué de façon homogène entre les zones.

            #### 🧪 Test de Friedman (test non paramétrique)
            - Utilisé quand **au moins une zone n’a pas une distribution normale**.
            - C’est l’alternative non paramétrique à l’ANOVA pour données appariées.
            - **Interprétation :**
            - Si *p-value < 0.05* → **différences significatives** entre les zones.
            - Si *p-value ≥ 0.05* → **pas de différence significative**.

            👉 Ces tests permettent de détecter des **asymétries fonctionnelles ou structurelles** dans le pied.
            """)



    else:
        st.info("Veuillez charger un fichier Excel pour commencer l'analyse.")
