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

# Consolidated sidebar radio button for page navigation
page = st.sidebar.radio(
    "Choisir la page :",
    [
        "Statistiques par zone",
        "Test de normalité",
        "Comparaison Gauche vs Droit (Q-Q Plot & Boxplot)", # Renamed for clarity
        "Plot Planck-Hartmann (Individuel)", # Renamed for clarity
        "Plot Planck-Hartmann (Moyenne Répétitions)", # Renamed for clarity
        "Test de symétrie Gauche/Droite", # Renamed for clarity and consolidated
        "Test inter-zone (paramétrique/non-paramétrique)", # Renamed for clarity
        "Compare Inter-Zone (Detailed)" # Renamed for clarity
    ]
)

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

    # This 'parameters' list is used for iterating through parameters for display/analysis
    parameters_display_keys = [
        ("Tone - Pied Droit",      'tone_right'),
        ("Tone - Pied Gauche",     'tone_left'),
        ("Stiffness - Pied Droit", 'stiffness_right'),
        ("Stiffness - Pied Gauche",'stiffness_left'),
        ("Frequency - Pied Droit", 'frequency_right'),
        ("Frequency - Pied Gauche",'frequency_left')
    ]

    # This list is used for right/left comparison tests
    parameters_comparison_keys = [
        ("Tone", "tone_right", "tone_left"),
        ("Stiffness", "stiffness_right", "stiffness_left"),
        ("Frequency", "frequency_right", "frequency_left")
    ]

    def extract_param(index_list):
        # Ensure the extracted data is properly converted to numeric, handling potential non-numeric entries
        extracted_df = df.iloc[index_list].reset_index(drop=True).T
        # Select only the first 7 columns as per original logic, assuming they correspond to foot_zones
        return extracted_df.iloc[:, :7].apply(pd.to_numeric, errors='coerce')


    if page == "Statistiques par zone":
        st.header("📊 Statistiques par zone")
        def compute_statistics_by_zone(data, param_name):
            stats = []
            for i, zone in enumerate(foot_zones):
                values = pd.to_numeric(data.iloc[:, i], errors='coerce').dropna() # Ensure values are numeric and drop NaNs for stats
                mean = values.mean()
                std = values.std()
                cv = std / mean if mean != 0 else np.nan

                if pd.isna(cv):
                    interpretation = "Données insuffisantes ou CV indéfini"
                elif cv < 0.10:
                    interpretation = "✅ Bonne représentativité (CV < 10%)"
                elif cv < 0.20:
                    interpretation = "⚠️ Variabilité modérée (10% ≤ CV < 20%)"
                else:
                    interpretation = "❌ Forte variabilité (CV ≥ 20%)"

                stats.append({
                    "Zone": zone,
                    "Moyenne": f"{mean:.2f}" if pd.notnull(mean) else "N/A",
                    "Écart-type": f"{std:.2f}" if pd.notnull(std) else "N/A",
                    "CV (%)": f"{cv * 100:.2f}" if pd.notnull(cv) else "N/A",
                    "Interprétation": interpretation
                })
            return pd.DataFrame(stats)

        for label, key in parameters_display_keys:
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
                    "p-value": f"{p_value:.4f}" if pd.notna(p_value) else "N/A",
                    "Conclusion": "✅ Normale" if pd.notna(p_value) and p_value > 0.05 else "❌ Non normale"
                })

            result_df = pd.DataFrame(results)
            st.write(f"📋 Résultats du test de Shapiro-Wilk pour : {param_name}")
            st.dataframe(result_df)

        # Affichage normalité simple for all parameters (right and left)
        for label, key in parameters_display_keys:
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

    elif page == "Comparaison Gauche vs Droit (Q-Q Plot & Boxplot)": # Updated page name
        st.header("📊 Comparaison Gauche vs Droit – par Paramètre et Zone")

        def plot_combined_zonewise_boxplot(param_key_right, param_key_left, param_name, foot_zones, df_excel, index_map):
            data_right = extract_param(index_map[param_key_right])
            data_left = extract_param(index_map[param_key_left])

            box_data = []
            positions = []
            labels = []
            width = 0.35  # Half-width for side-by-side boxplots

            zone_indexes = range(len(foot_zones))

            fig, ax = plt.subplots(figsize=(12, 6))
            for i, zone in enumerate(foot_zones):
                vals_right = pd.to_numeric(data_right.iloc[:, i], errors='coerce').dropna()
                vals_left = pd.to_numeric(data_left.iloc[:, i], errors='coerce').dropna()

                if len(vals_right) < 2 or len(vals_left) < 2:
                    st.warning(f"Pas assez de données pour la zone '{zone}' ({param_name}) – ignorée.")
                    continue

                # Pied droit
                bp1 = ax.boxplot(vals_right, positions=[i - width/2], widths=width,
                                patch_artist=True, boxprops=dict(facecolor='lightblue'), showfliers=False)
                # Pied gauche
                bp2 = ax.boxplot(vals_left, positions=[i + width/2], widths=width,
                                patch_artist=True, boxprops=dict(facecolor='lightgreen'), showfliers=False)

                labels.append(zone)

            ax.set_xticks(list(zone_indexes))
            ax.set_xticklabels(foot_zones, rotation=20)
            ax.set_title(f"📦 Comparaison Pied Droit vs Gauche – {param_name}")
            ax.set_ylabel(param_name)
            ax.grid(True)
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Pied Droit", "Pied Gauche"], loc="upper right")
            st.pyplot(fig)
            plt.close(fig)
            

        def plot_all_zones_comparison(param_key_right, param_key_left, param_name, foot_zones, df_excel, index_map):
            data_right = extract_param(index_map[param_key_right])
            data_left = extract_param(index_map[param_key_left])

            # Flatten all zone values into one array per side
            vals_right_all = []
            vals_left_all = []

            for i in range(len(foot_zones)):
                vals_right_zone = pd.to_numeric(data_right.iloc[:, i], errors='coerce').dropna().tolist()
                vals_left_zone = pd.to_numeric(data_left.iloc[:, i], errors='coerce').dropna().tolist()

                vals_right_all.extend(vals_right_zone)
                vals_left_all.extend(vals_left_zone)

            if len(vals_right_all) < 2 or len(vals_left_all) < 2:
                st.warning(f"Pas assez de données globales pour tracer '{param_name}'.")
                return

            # Boxplot global
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([vals_right_all, vals_left_all], patch_artist=True,
                    labels=["Pied Droit", "Pied Gauche"],
                    boxprops=dict(facecolor="lightblue"),
                    medianprops=dict(color='black'))

            ax.set_title(f"📦 Boxplot global – {param_name} (toutes zones)")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)


        for param_name, key_right, key_left in parameters_comparison_keys:
            st.subheader(f"🔹 {param_name}")
            plot_combined_zonewise_boxplot(key_right, key_left, param_name, foot_zones, df, index_map)


        for param_name, key_right, key_left in parameters_comparison_keys:
            st.subheader(f"🔹 {param_name}")
            plot_all_zones_comparison(key_right, key_left, param_name, foot_zones, df, index_map)
            

    elif page == "Plot Planck-Hartmann (Individuel)": # Updated page name
        st.header("🎭 Planck-Hartmann - Comparaison Pied Droit / Pied Gauche (Individuel)")

        def plot_planck_hartmann(right_df_raw, left_df_raw, param_name, foot_zones):
            # Ensure data is numeric
            right_numeric = right_df_raw.apply(pd.to_numeric, errors='coerce')
            left_numeric = left_df_raw.apply(pd.to_numeric, errors='coerce')

            # Drop rows (subjects) that have any NaN values across all zones
            # This is important for consistent centering across subjects/zones
            right_numeric_clean = right_numeric.dropna()
            left_numeric_clean = left_numeric.dropna()

            if right_numeric_clean.empty or left_numeric_clean.empty:
                st.warning(f"Pas assez de données complètes pour tracer le graphique Planck-Hartmann pour {param_name}.")
                return

            mean_right_zones = right_numeric_clean.mean(axis=0)
            mean_left_zones = left_numeric_clean.mean(axis=0)

            # Center individual data points by subtracting the *overall mean of that specific zone*
            centered_right = right_numeric_clean - mean_right_zones
            centered_left = left_numeric_clean - mean_left_zones


            # Moyenne globale (droite + gauche) - this mean should also be across zones
            mean_global = pd.concat([right_numeric_clean, left_numeric_clean]).mean(axis=0) # Mean of each zone across all subjects, both sides
            mean_global_centered = mean_global - mean_global.mean() # Center this global mean itself

            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(foot_zones))
            width = 0.15 # Reduced width for clarity in plots

            # Tracer chaque individu (droite)
            for i in range(len(centered_right)):
                # Adjust x-position for right side points slightly to the left of the zone center
                ax.plot(x - width, centered_right.iloc[i], marker='o', linestyle='-',
                        color='dodgerblue', alpha=0.6, label="Individu Pied Droit" if i == 0 else "")

            # Tracer chaque individu (gauche)
            for i in range(len(centered_left)):
                # Adjust x-position for left side points slightly to the right of the zone center
                ax.plot(x + width, centered_left.iloc[i], marker='o', linestyle='--',
                        color='darkorange', alpha=0.6, label="Individu Pied Gauche" if i == 0 else "")

            # Tracer la moyenne générale (droite + gauche) centrée (ligne épaisse et couleur distincte)
            ax.plot(x, mean_global_centered, marker='s', linestyle='-', color='green', linewidth=3, label="Moyenne Générale Centrée", alpha=0.8)

            # Ligne horizontale 0
            ax.axhline(0, color='red', linestyle='--', label="Écart à la moyenne = 0")

            ax.set_xticks(x)
            ax.set_xticklabels(foot_zones, fontsize=12)
            ax.set_ylabel("Valeur centrée (différence à la moyenne globale de la zone)")
            ax.set_title(f"Graphique de Planck-Hartmann pour {param_name.upper()}")
            ax.legend(loc='upper right')
            ax.grid(True)

            # --- Partie téléchargement ---
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Télécharger le graphique (PNG)",
                data=buf,
                file_name=f"Planck_Hartmann_Individuel_{param_name}.png",
                mime="image/png"
            )
            st.pyplot(fig)
            plt.close(fig) # Close plot to free memory

            # Calculate CI based on clean data
            if not right_numeric_clean.empty:
                ci_right = 1.96 * right_numeric_clean.stack().std() / np.sqrt(len(right_numeric_clean.stack().dropna()))
                st.info(f"Intervalle de confiance 95% estimé pour le Pied Droit (globalement) : ±{ci_right:.2f}")
            if not left_numeric_clean.empty:
                ci_left = 1.96 * left_numeric_clean.stack().std() / np.sqrt(len(left_numeric_clean.stack().dropna()))
                st.info(f"Intervalle de confiance 95% estimé pour le Pied Gauche (globalement) : ±{ci_left:.2f}")

        # Use actual extracted data, not simulated data
        for param_name, key_right, key_left in parameters_comparison_keys:
            st.subheader(f"🔹 {param_name}")
            data_right = extract_param(index_map[key_right])
            data_left = extract_param(index_map[key_left])
            plot_planck_hartmann(data_right, data_left, param_name, foot_zones)


        # --- Interprétation du Plot Planck-Hartmann ---
        st.markdown("### 🧠 Interprétation des graphiques Planck-Hartmann")
        st.markdown("""
        - **Les courbes bleues** représentent les valeurs centrées pour chaque individu au pied droit,
        tandis que les **courbes orange** représentent celles pour le pied gauche.
        - Les valeurs sont centrées par zone, c’est-à-dire que l’on montre l’écart à la moyenne **de chaque zone (tous sujets confondus)**,
        ce qui permet de visualiser les différences relatives de chaque sujet selon la zone du pied, par rapport à la moyenne de cette zone.

        - La **courbe verte épaisse** correspond à la moyenne générale (droite + gauche) de tous les sujets **centrée par rapport à sa propre moyenne**,
        permettant d’évaluer la tendance moyenne globale pour chaque zone.

        - La **ligne rouge en pointillé** indique la référence zéro (moyenne par zone).
        Des points au-dessus indiquent des valeurs supérieures à la moyenne,
        en dessous des valeurs inférieures.

        - Les intervalles de confiance 95% (affichés en info) donnent une idée de la variabilité et de la précision de la moyenne.

        ### Comment interpréter ?

        - Si les courbes individuelles (bleues et oranges) sont proches de la ligne rouge zéro et de la moyenne verte, cela signifie une homogénéité des profils entre sujets et une faible variabilité autour de la moyenne de la zone.
        - De fortes variations individuelles peuvent indiquer une variabilité importante à prendre en compte.
        - Des différences systématiques entre pied droit et gauche (décalage entre les nuages de points bleus et oranges pour une même zone) peuvent révéler des asymétries biomécaniques ou fonctionnelles.

        Cette visualisation permet ainsi d’identifier facilement les zones du pied où la variabilité ou les asymétries sont les plus marquées, et de guider des analyses ou interventions ciblées.
        """)

    elif page == "Plot Planck-Hartmann (Moyenne Répétitions)": # Updated page name
        st.header("📊 Analyse avec moyenne des répétitions par zone, pied droit et pied gauche")

        def plot_planck_hartmann_mean_repetitions(right_df_raw, left_df_raw, param_name, foot_zones):
            st.markdown(f"### 🎭 Planck-Hartmann : {param_name.upper()}")

            right_numeric = right_df_raw.apply(pd.to_numeric, errors='coerce')
            left_numeric = left_df_raw.apply(pd.to_numeric, errors='coerce')

            # Drop rows (subjects) that have any NaN values across all zones
            right_numeric_clean = right_numeric.dropna()
            left_numeric_clean = left_numeric.dropna()

            if right_numeric_clean.empty or left_numeric_clean.empty:
                st.warning(f"Pas assez de données complètes pour tracer le graphique Planck-Hartmann pour {param_name}.")
                return

            # Moyenne des répétitions par zone (axis=0 = colonne) - this calculates the mean across subjects for each zone
            mean_right_per_zone = right_numeric_clean.mean(axis=0)
            mean_left_per_zone = left_numeric_clean.mean(axis=0)

            # Centrer chaque répétition (individu) par rapport à la moyenne de sa ZONE (pas sa propre moyenne)
            # This is the core of a Planck-Hartmann plot: showing deviation from the overall zone mean
            centered_right = right_numeric_clean - mean_right_per_zone
            centered_left = left_numeric_clean - mean_left_per_zone

            # Moyenne globale (droite + gauche) across zones
            mean_global_across_zones = pd.concat([right_numeric_clean, left_numeric_clean]).mean(axis=0)
            mean_global_centered = mean_global_across_zones - mean_global_across_zones.mean() # Centered by its own mean

            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(foot_zones))
            width = 0.15

            # Tracer répétitions pied droit
            for i in range(len(centered_right)):
                ax.plot(x - width, centered_right.iloc[i], marker='o', linestyle='-',
                        color='dodgerblue', alpha=0.6, label="Individu Pied Droit" if i == 0 else "")

            # Tracer répétitions pied gauche
            for i in range(len(centered_left)):
                ax.plot(x + width, centered_left.iloc[i], marker='o', linestyle='--',
                        color='darkorange', alpha=0.6, label="Individu Pied Gauche" if i == 0 else "")

            # Tracer moyenne globale
            ax.plot(x, mean_global_centered, marker='s', linestyle='-', color='green', linewidth=3, label="Moyenne Générale Centrée", alpha=0.8)

            ax.axhline(0, color='red', linestyle='--', label="Écart à la moyenne = 0")
            ax.set_xticks(x)
            ax.set_xticklabels(foot_zones)
            ax.set_ylabel("Valeur centrée (différence à la moyenne globale de la zone)")
            ax.set_title(f"Graphique de Planck-Hartmann pour {param_name.upper()}")
            ax.legend(loc='upper right')
            ax.grid(True)

            # --- Partie téléchargement ---
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Télécharger le graphique (PNG)",
                data=buf,
                file_name=f"Planck_Hartmann_Moyenne_Repetitions_{param_name}.png",
                mime="image/png"
            )
            st.pyplot(fig)
            plt.close(fig) # Close plot to free memory

            # Calculate CI based on clean data
            if not right_numeric_clean.empty:
                ci_right = 1.96 * right_numeric_clean.stack().std() / np.sqrt(len(right_numeric_clean.stack().dropna()))
                st.info(f"Intervalle de confiance 95% estimé pour le Pied Droit (globalement) : ±{ci_right:.2f}")
            if not left_numeric_clean.empty:
                ci_left = 1.96 * left_numeric_clean.stack().std() / np.sqrt(len(left_numeric_clean.stack().dropna()))
                st.info(f"Intervalle de confiance 95% estimé pour le Pied Gauche (globalement) : ±{ci_left:.2f}")

        # List of parameters to analyze and display
        for param_name, key_right, key_left in parameters_comparison_keys:
            st.markdown(f"## Analyse du paramètre : {param_name}")
            data_right = extract_param(index_map[key_right])
            data_left = extract_param(index_map[key_left])
            plot_planck_hartmann_mean_repetitions(data_right, data_left, param_name, foot_zones)


    elif page == "Test de symétrie Gauche/Droite": # Consolidated and renamed page
        st.header("⚖️ Test de symétrie Gauche vs Droit (t-test ou Wilcoxon selon normalité)")

        def test_symmetry_with_parametric_choice(param_key_right, param_key_left, foot_zones):
            data_right = extract_param(index_map[param_key_right])
            data_left = extract_param(index_map[param_key_left])

            results = []

            for i, zone in enumerate(foot_zones):
                vals_right = pd.to_numeric(data_right.iloc[:, i], errors='coerce').dropna()
                vals_left = pd.to_numeric(data_left.iloc[:, i], errors='coerce').dropna()

                # Find common indices for paired test
                common_idx = vals_right.index.intersection(vals_left.index)
                vals_right_paired = vals_right.loc[common_idx]
                vals_left_paired = vals_left.loc[common_idx]

                if len(vals_right_paired) < 2 or len(vals_left_paired) < 2: # Paired test needs at least 2 pairs
                    results.append({
                        "Zone": zone,
                        "Test utilisé": "N/A",
                        "Statistique": "N/A",
                        "p-value": "N/A",
                        "Conclusion": "⚠️ Données insuffisantes pour test apparié"
                    })
                    continue

                # Check normality with Shapiro-Wilk (min 3 data points for Shapiro-Wilk)
                norm_right = len(vals_right_paired) >= 3 and shapiro(vals_right_paired).pvalue > 0.05
                norm_left = len(vals_left_paired) >= 3 and shapiro(vals_left_paired).pvalue > 0.05

                # Choose test based on normality
                if norm_right and norm_left:
                    stat, p_value = ttest_rel(vals_right_paired, vals_left_paired)
                    test_used = "Test t apparié (paramétrique)"
                else:
                    try:
                        stat, p_value = wilcoxon(vals_right_paired, vals_left_paired, zero_method='wilcox') # Added zero_method for consistency
                        test_used = "Test de Wilcoxon (non paramétrique)"
                    except ValueError as e:
                        stat, p_value = np.nan, np.nan
                        test_used = f"Test de Wilcoxon (erreur: {e})"

                # Interpretation of the result
                if pd.isna(p_value):
                    conclusion = "⚠️ Calcul impossible"
                elif p_value < 0.05:
                    conclusion = "❌ Différence significative"
                else:
                    conclusion = "✅ Pas de différence significative"

                results.append({
                    "Zone": zone,
                    "Test utilisé": test_used,
                    "Statistique": round(stat, 4) if not pd.isna(stat) else "N/A",
                    "p-value": round(p_value, 4) if not pd.isna(p_value) else "N/A",
                    "Conclusion": conclusion
                })

            return pd.DataFrame(results)

        for param_name, key_right, key_left in parameters_comparison_keys:
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


    
else:
    st.info("Veuillez charger un fichier Excel pour commencer l'analyse.")
