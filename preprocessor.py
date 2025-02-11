import pandas as pd
from sklearn.impute import KNNImputer

class Preprocessor:
    """
    A class for preprocessing SAS XPT files, renaming columns based on metadata,
    and merging DataFrames on the 'sequence_no' column.
    """
    def __init__(self, df_vars):
        """
        Initializes the Preprocessor with metadata DataFrame.

        Args:
            df_vars (pd.DataFrame): Metadata DataFrame containing column renaming information.
        """
        self.df_vars = df_vars

    def process_xpt_files(self, file_info):
        """
        Processes a list of SAS XPT files, renames their columns based on provided metadata,
        and merges all DataFrames on the 'sequence_no' column.

        Args:
            file_info (list of tuple): List of tuples where each tuple contains the file path (str) 
                                      and corresponding data file name (str).

        Returns:
            pd.DataFrame: A merged DataFrame containing all processed DataFrames.
        """
        processed_dfs = []

        for file_path, data_file_name in file_info:
            # Load the XPT file into a DataFrame
            df = pd.read_sas(file_path, format='xport')

            # Filter the metadata to find relevant rows
            relevant_rows = self.df_vars[(self.df_vars['Variable Name'].isin(df.columns)) & (self.df_vars['Data File Name'] == data_file_name)]

            # Rename columns based on the metadata
            rename_dict = dict(zip(relevant_rows['Variable Name'], relevant_rows['Renamed_variables']))
            df = df.rename(columns=rename_dict)

            # Add the processed DataFrame to the list
            processed_dfs.append(df)

        # Merge all processed DataFrames on 'sequence_no'
        merged_df = processed_dfs[0]
        i = 0
        for df in processed_dfs[1:]:
            i += 1
            if i == 7:
                print(df.columns) 
            merged_df = pd.merge(merged_df, df, on='sequence_no', how='inner')
            print(len(merged_df))

        return merged_df
    
    def drop_nan_columns(self, df, threshold=2500):
        """
        Drops columns with NaN values exceeding a specified threshold.

        Args:
            df (pd.DataFrame): Input DataFrame.
            threshold (int): Maximum allowed NaN values per column.

        Returns:
            pd.DataFrame: DataFrame with dropped columns.
        """
        return df.loc[:, df.isnull().sum() < threshold]
    
    def impute_missing_values(self, merged_dataframe):
        """
        Imputes missing values in categorical columns using mode and numerical columns using KNN.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with imputed values.

        """
        df = merged_dataframe.copy()
        df_encoded = pd.get_dummies(df, drop_first=True)
        knn_imputer = KNNImputer(n_neighbors=5)
        df_imputed = knn_imputer.fit_transform(df_encoded)
        
        return pd.DataFrame(df_imputed, columns=df_encoded.columns)
    
    def hot_deck_imputation(self, df):
        """
        Removes duplicates and imputes missing values using hot deck imputation.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        df.drop_duplicates(inplace=True)
        
        def impute(col):
            valid = col.dropna()
            return col.apply(lambda x: x if pd.notna(x) else valid.sample(1).values[0] if not valid.empty else x)
        
        for col in df.columns:
            df[col] = impute(df[col])
        
        return df

    def add_diabetes_status(self, df_feature, df_diabetes, target_column='DIQ010'):
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
        print(df_diabetes.columns)
        if 'sequence_no' not in df_feature.columns or 'sequence_no' not in df_diabetes.columns:
            raise ValueError("Both DataFrames must have a 'sequence_no' column.")
        
        # Merge the DataFrames on 'sequence_no' to get the diabetes status
        merged_df = df_feature.merge(df_diabetes[['sequence_no', target_column]], on='sequence_no', how='left')
        
        # Return the updated DataFrame
        return merged_df