import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots(df, cols=None):
    """Plot boxplots for outlier detection."""
    if cols is None:
        cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[cols].plot(kind='box', subplots=True, layout=(len(cols)//3 + 1, 3), figsize=(15, 10), sharex=False)
    plt.suptitle("Boxplots for Outlier Detection")
    plt.show()

def plot_correlation_matrix(df):
    """Plot correlation matrix."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
def plot_distributions(df, cols=None):
    """
    Plots the distribution of numeric features.
    """
    if cols is None:
        cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[cols].hist(figsize=(15, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()