import pandas as pd
import yaml
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    def __init__(self, credentials_path):
        self.credentials_path = credentials_path
        self.credentials = None
        self.engine = None

    def load_credentials(self):
        with open(self.credentials_path, "r") as file:
            self.credentials = yaml.safe_load(file)

    def init_engine(self):
        if not self.engine and self.credentials:
            try:
                connection_string = f"postgresql+psycopg2://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
                self.engine = create_engine(connection_string)
                print("Engine initialized successfully.")
            except Exception as e:
                print(f"Error initializing engine: {str(e)}")

    def extract_data(self, table_name="loan_payments"):
        try:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"Error extracting data: {str(e)}")

    def save_to_csv(self, data_frame, file_path="output_data.csv"):
        try:
            data_frame.to_csv(file_path, index=False)
            print(f"Data saved to {file_path} successfully.")
        except Exception as e:
            print(f"Error saving data to CSV: {str(e)}")

credentials_path = "C:/Users/Lenovo/Training/credentials.yaml"

rds_connector = RDSDatabaseConnector(credentials_path)
rds_connector.load_credentials()
rds_connector.init_engine()

data = rds_connector.extract_data()
rds_connector.save_to_csv(data)

import pandas as pd

def load_data(file_path="output_data.csv"):
    try:
        data = pd.read_csv(file_path)

        print("Data Shape:", data.shape)
        print("\nSample Data:")
        print(data.head())

        return data

    except Exception as e:
        print(f"Error loading data: {str(e)}")


output_data_path = "output_data.csv"

loaded_data = load_data(output_data_path)
