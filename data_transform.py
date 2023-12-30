import pandas as pd

class DataTransform:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def convert_to_numeric(self, column):
        self.dataframe[column] = pd.to_numeric(self.dataframe[column], errors='coerce')

    def convert_to_datetime(self, column, format='%Y-%m-%d'):
        self.dataframe[column] = pd.to_datetime(self.dataframe[column], format=format, errors='coerce')

    def drop_missing_values(self):
        self.dataframe.dropna(inplace=True)

    def uppercase_column(self, column):
        self.dataframe[column] = self.dataframe[column].str.upper()

from data_transform import DataTransform

data_transformer = DataTransform(df)

data_transformer.convert_to_numeric('numeric_column')
data_transformer.convert_to_datetime('date_column')
data_transformer.drop_missing_values()
data_transformer.uppercase_column('text_column')

transformed_df = data_transformer.dataframe

import pandas as pd

class DataFrameInfo:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def describe_columns(self):
        return self.dataframe.dtypes

    def extract_statistics(self):
        return self.dataframe.describe()

    def count_distinct_values(self, column):
        return self.dataframe[column].nunique()

    def print_shape(self):
        print(f"Number of rows: {self.dataframe.shape[0]}, Number of columns: {self.dataframe.shape[1]}")

    def count_null_values(self):
        return self.dataframe.isnull().sum()

from data_frame_info import DataFrameInfo

df_info = DataFrameInfo(df)

column_types = df_info.describe_columns()
print("Column Types:")
print(column_types)

statistics = df_info.extract_statistics()
print("\nStatistics:")
print(statistics)

distinct_values_count = df_info.count_distinct_values('categorical_column')
print(f"\nNumber of distinct values in 'categorical_column': {distinct_values_count}")

df_info.print_shape()

null_values_count = df_info.count_null_values()
print("\nNull Values Count:")
print(null_values_count)

import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    @staticmethod
    def visualize_nulls(dataframe):
        plt.figure(figsize=(10, 6))
        sns.heatmap(dataframe.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Missing Values in DataFrame')
        plt.show()

import pandas as pd

class DataFrameTransform:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def check_nulls(self):
        return self.dataframe.isnull().sum()

    def drop_columns(self, columns_to_drop):
        self.dataframe.drop(columns=columns_to_drop, inplace=True)

    def impute_nulls(self, column, strategy='median'):
        if strategy == 'median':
            fill_value = self.dataframe[column].median()
        elif strategy == 'mean':
            fill_value = self.dataframe[column].mean()
        else:
            raise ValueError("Invalid imputation strategy. Use 'median' or 'mean'.")

        self.dataframe[column].fillna(fill_value, inplace=True)

import numpy as np

class DataFrameTransform:

    def identify_skewed_columns(self, skew_threshold=1):
        skewed_columns = self.dataframe.apply(lambda x: abs(x.skew()) > skew_threshold)
        return skewed_columns[skewed_columns].index.tolist()
    
class DataFrameTransform:

    def transform_skewed_columns(self, method='log1p'):
        for column in self.skewed_columns:
            if method == 'log1p':
                self.dataframe[column] = np.log1p(self.dataframe[column])
            elif method == 'sqrt':
                self.dataframe[column] = np.sqrt(self.dataframe[column])

class DataFrameTransform:

    def save_copy(self, filename):
        self.dataframe.to_csv(filename, index=False)

class Plotter:
    @staticmethod
    def visualize_outliers(dataframe, column):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=dataframe[column])
        plt.title(f'Outliers in {column}')
        plt.show()

class DataFrameTransform:
    def remove_outliers(self, column, method='z-score', threshold=3):
        if method == 'z-score':
            z_scores = (self.dataframe[column] - self.dataframe[column].mean()) / self.dataframe[column].std()
            outliers = abs(z_scores) > threshold
        elif method == 'IQR':
            Q1 = self.dataframe[column].quantile(0.25)
            Q3 = self.dataframe[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (self.dataframe[column] < Q1 - 1.5 * IQR) | (self.dataframe[column] > Q3 + 1.5 * IQR)
        else:
            raise ValueError("Invalid outlier removal method. Use 'z-score' or 'IQR'.")

        self.dataframe = self.dataframe[~outliers]

class Plotter:
    @staticmethod
    def visualize_correlation(dataframe):
        correlation_matrix = dataframe.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()


class DataFrameTransform:
    def identify_highly_correlated_columns(self, threshold=0.8):
        correlation_matrix = self.dataframe.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
        highly_correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        return highly_correlated_columns
    
class DataFrameTransform:
    def remove_highly_correlated_columns(self, threshold=0.8):
        correlation_matrix = self.dataframe.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
        highly_correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.dataframe = self.dataframe.drop(columns=highly_correlated_columns)
