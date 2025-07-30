# DiafooT

DiafooT is a research initiative focused on improving the management and prevention of diabetic foot complications, particularly those at risk of amputation due to neuropathy and vascular issues. Our goal is to enhance patient outcomes by identifying and treating the underlying causes of foot injuries, ultimately preserving mobility and quality of life. 🌍

## Project Overview

Globally, approximately 500 million people live with diabetes, many of whom face severe complications such as neuropathy (loss of sensation in the feet) and arteriopathy (impaired blood flow). These conditions contribute to chronic wounds and amputations, with a lower limb amputation occurring every 20 seconds worldwide. Current healthcare systems often lack effective prevention and treatment strategies, leading to suboptimal outcomes.

Our research aims to:

- Understand the intrinsic (e.g., skin thickness, bone geometry) and extrinsic (e.g., gait mechanics, pressure distribution 🚶) factors contributing to foot injuries.
- Develop methods to give patients more time before complications escalate , preserving their body and enabling a more active life.
- Prevent new wounds and recurrences by addressing root causes through clinical observation and data-driven analysis.

We analyze parameters such as skin stiffness, pressure points, and vascular properties to identify correlations and develop predictive models. By clustering patients based on risk grades (IWGDF Grades 0–3) and without it, we aim to tailor interventions to specific patient profiles.

## Data and Tools 📊

The project leverages clinical data stored in Excel file 📑 with a dedicated "DIAFOOT" sheet. We use advanced analytical tools to process and visualize this data, including:

- **Streamlit 🌐**: For interactive dashboards to explore data and analysis results.
- **Python Libraries**:  
  - **Core**: `datetime`, `io`, `re`  
  - **Data Handling**: `pandas`, `numpy`  
  - **Visualization**: `matplotlib`, `seaborn`  
  - **Statistical Analysis**: `scipy`, `statsmodels` (including `MANOVA`)  
  - **Machine Learning & Clustering**: `scikit-learn` (`KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `LogisticRegression`, `LinearRegression`, `RandomForestClassifier`, `IsolationForest`, `PCA`, `SimpleImputer`, `StandardScaler`)  
  - **Metrics & Evaluation**: `silhouette_score`, `adjusted_rand_score`, `normalized_mutual_info_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `confusion_matrix`  
  - **Web App Development**: `streamlit`  

  Used for data processing ⚙️, statistical analysis 📉, clustering 🧩, modeling 🤖, and visualization 🎨.


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


## Dashboard Code Explanation (`dashboard.py`) 

The `dashboard.py` script powers the DIAFOOT Analysis Dashboard, providing a user-friendly interface for uploading and analyzing clinical data. Below is a breakdown of its functionality:


### File Upload 📂

- Users upload an Excel file containing a "DIAFOOT" sheet, which is read into a Pandas DataFrame.


### Analysis Options 🔍

The sidebar now offers multiple analysis types, selected via a radio button menu:  

1. **Basic Analysis **: Runs quick preliminary checks on the dataset.  
2. **Descriptive Analysis**: Generates detailed descriptive statistics for all parameters.  
3. **Normality Tests**: Applies tests such as Shapiro–Wilk to assess data distribution.  
4. **L/R Comparison by Anatomical Zone **: Compares left vs right foot data within anatomical zones.  
5. **Comparison of Left and Right Foot Parameters **: Performs paired comparisons for overall foot parameters.  
6. **Diabetic vs Control**: Compares parameters between diabetic and control groups.  
7. **IWGDF Risk Grade Summary & Clustering **: Summarizes risk grades and performs clustering.  
8. **Clustering (Important Parameters)**: Applies clustering algorithms (KMeans, Agglomerative, GMM) using selected key features.  
9. **Clustering (All Parameters)**: Runs clustering on the full set of parameters and evaluates metrics (silhouette, ARI, etc.).  
10. **Correlation Between Key Parameters**: Computes and visualizes correlations between biomechanical and clinical parameters.  
11. **Bland–Altman Plots by Parameter and Side**: Generates Bland–Altman plots for each parameter by left/right side.  
12. **Bland–Altman Pooled Plots for All Parameters**: Creates pooled Bland–Altman plots across all parameters.  


### Key Features

- **Automation**: Calculations update automatically when new patient data is added to the Excel file.
- **Interactivity**: Users can select analysis types, groups, and thresholds (e.g., correlation threshold).
- **Explanations**: Expander sections provide insights into metrics (e.g., silhouette score, ARI) and chart interpretations.

## Research Insights 🧠 

The **DIAFOOT study** focused on middle-aged adults with diabetes (mean age ≈57 years; mean BMI ~25.8 kg/m²), a population where foot pressure and ulcer risk typically begin to rise. Most participants were classified at the extremes of the **IWGDF risk scale** (Grades 0 or 3), providing valuable insight into early- and late-stage risk profiles.  

###  Key Findings 🔑

- **Neuropathy & Soft Tissue Integrity**: Peripheral nerve dysfunction and reduced plantar fat pad thickness remain core predictors of **diabetic foot ulcer (DFU)** risk. Neuropathy prevalence was strikingly high in at-risk groups.  
- **Biomechanical & Thermal Asymmetries**: Significant left/right differences were observed in **plantar pressure** (notably under the 5th metatarsal), **big toe stiffness**, and **plantar temperature**. These imbalances can signal uneven load distribution, vascular stress, or subclinical inflammation.  
- **Correlations**:  
  - Plantar tissue thickness ↔ plantar temperature: *ρ ≈ 0.99* (thermal insulation effect)  
  - IWGDF risk grade ↔ Michigan Neuropathy Score: *ρ ≈ -0.83* (neuropathy severity aligns with higher risk)  

Using **LASSO regression**, **Random Forests**, and **ANOVA-based feature ranking**, the top predictors of DFU risk were:  
1. **Michigan Neuropathy Score**  
2. **Patient height**  
3. **Plantar tissue thickness**  

This convergence across algorithms highlights the robustness of these markers. Simple, non-invasive measurements (neuropathy tests, anthropometrics, ultrasound) could enable scalable DFU screening.  

---

**Bottom line:**  
Our results demonstrate that readily measurable foot parameters **neuropathy, plantar pressure, stiffness, tissue thickness, and temperature** can powerfully stratify DFU risk. Integrating these metrics into clinical workflows may enable **earlier detection** and **personalized prevention strategies**, especially in primary care or resource-limited settings.  

