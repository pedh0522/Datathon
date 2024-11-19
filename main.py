import pandas as pd
from data_processing import load_data, preprocess_csv, get_diabetes_status, split_by_diabetes_status, add_diabetes_status
from analysis import perform_pca, perform_lda, multivariate_analysis
from visualization import plot_boxplots, plot_distributions, plot_correlation_matrix
from utils import overview, check_missing_values

# Load data
df_dia, df_vars, df_alcohol = load_data()

# Preprocess alcohol data
df_alcohol = preprocess_csv(df_alcohol)

# Get diabetes status
dia_patients, non_dia_patients = get_diabetes_status(df_dia)
dia_al, nonDia_al = split_by_diabetes_status(df_alcohol, dia_patients, non_dia_patients)
print(len(non_dia_patients))
print(len(dia_al))
# Plot Boxplots
plot_boxplots(nonDia_al)
plot_boxplots(dia_al)

# Perform multivariate analysis
multivariate_analysis(df_alcohol)

# Add diabetes status to the alcohol dataset
df_alcohol_with_status = add_diabetes_status(df_alcohol, df_dia, 'EverTold_Diabetes')

# Perform PCA and LDA
pca_result = perform_pca(df_alcohol, n_components=2)
lda_result = perform_lda(df_alcohol_with_status, target_column='EverTold_Diabetes')
