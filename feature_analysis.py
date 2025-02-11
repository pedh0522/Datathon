import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class DimReductAndFeatureSelect:
    def __init__(self, X_train, y_train, X_test, n_top_features=15):
        """
        Initialize with training and testing data.
        
        Parameters:
        - X_train: Training features.
        - y_train: Training target labels.
        - X_test: Testing features.
        - n_top_features: Number of top features to select based on feature importance.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.n_top_features = n_top_features
        self.rf = RandomForestClassifier(random_state=42)
        self.top_features = None
    
    def select_top_features(self):
        """Train a RandomForest and select the top features based on importance."""
        self.rf.fit(self.X_train, self.y_train)
        importances = self.rf.feature_importances_
        self.top_features = np.argsort(importances)[-self.n_top_features:]
        self.X_train_selected = self.X_train[:, self.top_features]
        self.X_test_selected = self.X_test[:, self.top_features]
        print(f"Top {self.n_top_features} features selected.")
    
    def perform_pca(self, df, n_components=2):
        """Perform PCA on the DataFrame and plot explained variance."""
        df_numeric = df.select_dtypes(include=[np.number])
        df_std = (df_numeric - df_numeric.mean()) / (df_numeric.std() + 1e-10)
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(df_std)
        pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(n_components)])
        
        plt.plot(range(1, n_components+1), pca.explained_variance_ratio_, marker='o', linestyle='--')
        plt.title("Explained Variance by Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.grid(True)
        plt.show()
        
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)
        return pca_df
    
    def perform_lda(self, df, target_column, n_components=2):
        """Perform LDA on the DataFrame and plot the results."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_numeric = X.select_dtypes(include=[np.number])
        X_std = (X_numeric - X_numeric.mean()) / X_numeric.std()
        lda = LDA(n_components=n_components)
        lda_components = lda.fit_transform(X_std, y)
        lda_df = pd.DataFrame(lda_components, columns=[f'LD{i+1}' for i in range(n_components)])
        lda_df[target_column] = y.reset_index(drop=True)
        
        sns.scatterplot(data=lda_df, x='LD1', y='LD2', hue=target_column, palette='Set1')
        plt.title("LDA: Linear Discriminants")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        return lda_df
