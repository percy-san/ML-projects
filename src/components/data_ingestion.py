"""
Data Ingestion Module

This module handles the process of loading raw data and splitting it into
training and testing datasets. It's the first step in the machine learning pipeline.
"""

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.
    
    This class uses @dataclass decorator to automatically generate
    __init__, __repr__, and other special methods.
    
    Attributes:
        train_data_path: Path where training data will be saved
        test_data_path: Path where test data will be saved  
        raw_data_path: Path where raw data will be saved
    """
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")


class DataIngestion:
    """
    Data ingestion class that handles loading and splitting the dataset.
    
    This class is responsible for:
    1. Loading the raw dataset
    2. Splitting it into training and testing sets
    3. Saving the split datasets to files
    """

    def __init__(self):
        """
        Initialize the DataIngestion class with configuration.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Main method to perform data ingestion process.
        
        This method:
        1. Loads the dataset from the source file
        2. Creates necessary directories
        3. Saves the raw data
        4. Splits data into training and testing sets
        5. Saves the split datasets
        
        Returns:
            tuple: Paths to train data, test data, and raw data files
        
        Raises:
            CustomException: If any error occurs during the process
        """
        logging.info("Initiating data ingestion")
        try:
            # Load the dataset from the CSV file
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as a dataframe')

            # Create the directory for saving data if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            # Split the dataset into training (80%) and testing (20%) sets
            # random_state=42 ensures reproducible results
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing datasets to separate CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')

            # Return the paths to the saved files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )
        except Exception as e:
            # If any error occurs, raise a custom exception with details
            raise CustomException(e, sys)


# This block only runs if this file is executed directly (not imported)
if __name__ == '__main__':
    # Create a DataIngestion object
    obj = DataIngestion()

    # Perform data ingestion and get the file paths
    train_data, test_data, _ = obj.initiate_data_ingestion()

    # Create a DataTransformation object and perform data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
