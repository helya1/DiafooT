# DiafooT

DiafooT is a research initiative focused on improving the management and prevention of diabetic foot complications, particularly those at risk of amputation due to neuropathy and vascular issues. Our goal is to enhance patient outcomes by identifying and treating the underlying causes of foot injuries, ultimately preserving mobility and quality of life. 🌍

## Project Overview

Globally, approximately 500 million people live with diabetes, many of whom face severe complications such as neuropathy (loss of sensation in the feet) and arteriopathy (impaired blood flow). These conditions contribute to chronic wounds and amputations, with a lower limb amputation occurring every 20 seconds worldwide. ⏰ Current healthcare systems often lack effective prevention and treatment strategies, leading to suboptimal outcomes.

Our research aims to:

- Understand the intrinsic (e.g., skin thickness, bone geometry 🦴) and extrinsic (e.g., gait mechanics, pressure distribution 🚶) factors contributing to foot injuries.
- Develop methods to give patients more time before complications escalate ⏳, preserving their body and enabling a more active life.
- Prevent new wounds and recurrences by addressing root causes through clinical observation and data-driven analysis 📊.

We analyze parameters such as skin stiffness, pressure points, and vascular properties to identify correlations and develop predictive models. By clustering patients based on risk grades (IWGDF Grades 0–3) 📊, we aim to tailor interventions to specific patient profiles 🎯.

## Data and Tools 📊

The project leverages clinical data stored in Excel file 📑 with a dedicated "DIAFOOT" sheet. We use advanced analytical tools to process and visualize this data, including:

- **Streamlit 🌐**: For interactive dashboards to explore data and analysis results.
- **Python Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and SciPy for data processing ⚙️, statistical analysis 📉, clustering 🧩, and visualization 🎨.

### Website

- **Analysis Dashboard**: [Diafoot Analysis](https://diafoot-analysis.streamlit.app/)

## Quick Setup for Streamlit

To set up the Streamlit application locally, follow these steps:

```bash
pip install streamlit pandas matplotlib seaborn
```

Run the app with:

```bash
streamlit run dashboard.py
```


## Dashboard Code Explanation (`dashboard.py`) 🖥️📊

The `dashboard.py` script powers the DIAFOOT Analysis Dashboard, providing a user-friendly interface for uploading and analyzing clinical data. Below is a breakdown of its functionality:


### Imports and Setup ⚙️📚

- **Libraries**: Imports Streamlit for the UI 🧑‍💻, Pandas for data handling 🐼, NumPy for numerical operations ➗, Scikit-learn for machine learning (clustering, regression, PCA) 🤖, SciPy for statistical tests 📈, and Matplotlib/Seaborn for visualization 🎨.
- **Page Configuration**: Sets the page title and layout for a wide display.

### File Upload 📂⬆️

- Users upload an Excel file containing a "DIAFOOT" sheet, which is read into a Pandas DataFrame.📑


### Analysis Options 🔍

The sidebar offers six analysis types:

1. **Stat Summary Extractor 📊**: Computes mean and standard deviation for selected parameters (e.g., height, weight, pressure) and generates a downloadable Excel file with summary statistics and deviations.
2. **KMeans Clustering 🧩**: Clusters patients based on features correlated with risk grades using KMeans (4 clusters). Visualizes clusters with PCA and evaluates with silhouette score and ARI.
3. **ED Thickness & Hypodermis Analysis 🩺**: Analyzes epidermis and hypodermis thickness for specific risk groups (Grades 0–1 or 2–3) using t-tests, correlations, and linear regression.
4. **GMM Clustering 🔄**: Applies Gaussian Mixture Models (GMM) to cluster patients based on top features correlated with grades, visualized with PCA and evaluated with ARI.
5. **Correlation Between Key Parameters 🔗**: Computes and visualizes correlations between parameters (e.g., pressure, skin thickness) using heatmaps and bar plots.
6. **Mechanical Features Correlation Analysis ⚙️**: Focuses on mechanical parameters, identifying high correlations to reduce redundancy.

### Visualizations 

- **Heatmaps 🌈**: Display correlation matrices and cluster-grade contingencies.
- **Scatter Plots ✨**: Show PCA-transformed data for clustering visualization.
- **Bar Plots 📊**: Illustrate feature importance (regression coefficients) and correlations with specific variables.

### Key Features ⭐

- **Automation**: Calculations update automatically when new patient data is added to the Excel file.
- **Interactivity**: Users can select analysis types, groups, and thresholds (e.g., correlation threshold).
- **Explanations**: Expander sections provide insights into metrics (e.g., silhouette score, ARI) and chart interpretations.


## Research Insights 🩺🔍

Our clinical observations highlight: 

- **Neuropathy 🦶⚡**: Patients with reduced foot sensation are prone to unnoticed injuries, exacerbated by continuous pressure on wounds.
- **Arteriopathy ❤️🩸**: Poor blood flow hinders wound healing, worsening outcomes.
- **Mechanical Factors 🦴🛠️**: Skin thickness, stiffness, and bone geometry influence injury risk. Thicker, more elastic skin may absorb shocks better, while rigid tissue is more prone to damage.
- **Gait Analysis 🚶‍♂️📏**: Measures maximum pressure and pressure distribution to identify at-risk areas (e.g., first metatarsal head, sesamoids, fifth metatarsal).



the website for analysis : [Diafoot Analysis](https://diafoot-analysis.streamlit.app/)
