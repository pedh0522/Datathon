def overview(df):
    """Provide an overview of the DataFrame."""
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.info())
    print(df.head())

def check_missing_values(df):
    """Check for missing values."""
    missing = df.isnull().sum()
    print(missing[missing > 0])
