import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#Import features dataframe and Diabetes dataframe
path = 'D:\\Downloads\\raw_datasets'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

path = "D:\\Downloads\\raw_datasets\\"
df = pd.read_sas(path + 'diabetes.XPT', format='xport')
with open("D:\\Downloads\\variables.csv", mode='r', encoding='utf-8', errors='replace') as f:
    df_vars = pd.read_csv(f)

diabetes = pd.read_sas(path + 'diabetes.XPT', format='xport')

relevant_rows = df_vars[(df_vars['Variable Name'].isin(diabetes.columns)) & (df_vars['Data File Name'] == 'DIQ_L')]
df_dia = diabetes.rename(columns=dict(zip(relevant_rows['Variable Name'], relevant_rows['Renamed_variables'])))

df_alcohol = pd.read_sas(path + 'alcohol_use.XPT', format='xport')

#Rename coloumns
relevant_rows = df_vars[(df_vars['Variable Name'].isin(df_alcohol.columns)) & (df_vars['Data File Name'] == 'ALQ_L')]
df_alcohol = df_alcohol.rename(columns=dict(zip(relevant_rows['Variable Name'], relevant_rows['Renamed_variables'])))


# Preprocessing functions
def preprocess_csv(df):
    """
    Preprocess the given DataFrame by removing NaN values and duplicates.
    """
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def standardize_data(df):
    """
    Standardize the DataFrame (zero mean, unit variance).
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def normalize_data(df):
    """
    Normalize the DataFrame (min-max scaling).
    """
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

# Extract Diabetic and Non-Diabetic Patients
def get_diabetes_status(diabetes_df):
    """
    Extract diabetic and non-diabetic patient sequence numbers from the diabetes DataFrame.
    """
    # Filter patients based on diabetes status
    diabetic_patients = diabetes_df[diabetes_df['DIQ010'] == 1]['SEQN'].tolist()
    non_diabetic_patients = diabetes_df[diabetes_df['DIQ010'] == 2]['SEQN'].tolist()
    
    return diabetic_patients, non_diabetic_patients

def split_by_diabetes_status(feature_df, diabetic_patients, non_diabetic_patients):
    """
    Split the feature DataFrame into diabetic and non-diabetic groups.
    """
    # Separate based on sequence_no
    diabetic_df = feature_df[feature_df['sequence_no'].isin(diabetic_patients)]
    non_diabetic_df = feature_df[feature_df['sequence_no'].isin(non_diabetic_patients)]
    
    return diabetic_df, non_diabetic_df

def overview(df):
    """
    Provides a quick overview of the DataFrame.
    """
    print("Shape of DataFrame:", df.shape)
    print("\nColumn Names:\n", df.columns.tolist())
    print("\nBasic Info:\n")
    print(df.info())
    print("\nFirst 5 rows:\n", df.head())
    print("\nStatistical Summary:\n", df.describe())
    print("\nNumber of Duplicates:", df.duplicated().sum())

def check_missing_values(df):
    """
    Checks for missing values in the DataFrame.
    """
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0])
    print("\nPercentage of Missing Values:\n", (missing_values[missing_values > 0] / len(df) * 100).round(2))

# Box Plot and correlation Matrix
def plot_distributions(df, cols=None):
    """
    Plots the distribution of numeric features.
    """
    if cols is None:
        cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[cols].hist(figsize=(15, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()

# Correlation Matrix
def plot_correlation_matrix(df):
    """
    Plots the correlation matrix.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Class Balance
def check_class_balance(df, label_column):
    """
    Checks the balance of classes in the dataset.
    """
    class_counts = df[label_column].value_counts()
    print("\nClass Distribution:\n", class_counts)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title(f"Class Balance of {label_column}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def plot_boxplots(df, cols=None):
    """
    Plots boxplots to check for outliers.
    """
    if cols is None:
        cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[cols].plot(kind='box', subplots=True, layout=(len(cols)//3 + 1, 3), figsize=(15, 10), sharex=False, sharey=False)
    plt.suptitle("Boxplots for Outlier Detection")
    plt.show()

def remove_null_columns(df):
    """Remove columns where all entries are NaN."""
    return df.dropna(axis=1, how='all')


# Multivariate Analysis
def multivariate_analysis(df):
    """
    Perform multivariate analysis on the entire DataFrame.
    """
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

    # Pair plot for visualizing distributions and relationships
    sns.pairplot(df)
    plt.show()

# Principal Component Analysis and Linear Discriminant Analysis
def perform_pca(df, n_components=2):
    """
    Perform PCA on the given DataFrame.
    
    Parameters:
    - df: Input DataFrame with features.
    - n_components: Number of principal components to retain.
    
    Returns:
    - Transformed DataFrame with principal components.
    """
    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Standardizing the data
    df_std = (df_numeric - df_numeric.mean()) / df_numeric.std()
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_std)
    
    # Create DataFrame for principal components
    pca_df = pd.DataFrame(
        principal_components, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Plot explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components+1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.title("Explained Variance by Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.grid(True)
    plt.show()
    
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    
    return pca_df

def perform_lda(df, target_column, n_components=2):
    """
    Perform LDA on the given DataFrame.
    
    Parameters:
    - df: Input DataFrame with features and target.
    - target_column: The name of the target column.
    - n_components: Number of linear discriminants to retain.
    
    Returns:
    - Transformed DataFrame with linear discriminants.
    """
    # Separate features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Drop non-numeric columns from features
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Standardizing the data
    X_std = (X_numeric - X_numeric.mean()) / X_numeric.std()
    
    # Perform LDA
    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(X_std, y)
    
    # Create DataFrame for LDA components
    lda_df = pd.DataFrame(
        lda_components, 
        columns=[f'LD{i+1}' for i in range(n_components)]
    )
    lda_df[target_column] = y.reset_index(drop=True)
    
    # Plot the LDA components
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=lda_df, x='LD1', y='LD2', hue=target_column, palette='Set1')
    plt.title("LDA: Linear Discriminants")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return lda_df

def add_diabetes_status(df_feature, df_diabetes, target_column='DIQ010'):
    """
    Merge the diabetes status into the feature DataFrame based on SEQN.
    
    Parameters:
    df_feature (pd.DataFrame): DataFrame containing features with 'SEQN'.
    df_diabetes (pd.DataFrame): DataFrame containing diabetes status with 'SEQN'.
    target_column (str): Column in df_diabetes that represents the diabetes status.
    
    Returns:
    pd.DataFrame: The feature DataFrame with an added column for diabetes status.
    """
    # Ensure that 'SEQN' column is present in both DataFrames
    if 'sequence_no' not in df_feature.columns or 'sequence_no' not in df_diabetes.columns:
        raise ValueError("Both DataFrames must have a 'sequence_no' column.")
    
    # Merge the DataFrames on 'sequence_no' to get the diabetes status
    merged_df = df_feature.merge(df_diabetes[['sequence_no', target_column]], on='sequence_no', how='left')
    
    # Return the updated DataFrame
    return merged_df


df_alcohol = preprocess_csv(df_alcohol)
dia, non_dia = get_diabetes_status(diabetes)
dia_al, nonDia_al = split_by_diabetes_status(df_alcohol, dia, non_dia)

# Make Plots
plot_boxplots(nonDia_al)
plot_boxplots(dia_al)

multivariate_analysis(df_alcohol)

add_alc = add_diabetes_status(df_alcohol, df_dia, 'EverTold_Diabetes')

red_df = perform_pca(df_alcohol, 2)
vis_df = perform_lda(add_alc, 'EverTold_Diabetes')