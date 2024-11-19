import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns

def perform_pca(df, n_components=2):
    """Perform PCA on DataFrame."""
    df_std = (df - df.mean()) / df.std()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_std.select_dtypes(include=[np.number]))
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_, marker='o')
    plt.title("PCA Explained Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.show()
    
    return pca_df

def perform_lda(df, target_column, n_components=2):
    """Perform LDA."""
    X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    y = df[target_column]
    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(X, y)
    lda_df = pd.DataFrame(lda_components, columns=[f'LD{i+1}' for i in range(n_components)])
    lda_df[target_column] = y.reset_index(drop=True)
    
    sns.scatterplot(data=lda_df, x='LD1', y='LD2', hue=target_column)
    plt.title("LDA: Linear Discriminants")
    plt.show()
    
    return lda_df

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