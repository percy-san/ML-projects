"""
Data Transformation Module

This module handles the preprocessing and transformation of data for machine learning.
It creates preprocessing pipelines for numerical and categorical features,
and applies them to both training and testing datasets.
"""

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation paths.
    
    This class defines where the preprocessor object will be saved
    for later use in prediction.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    Data transformation class that handles preprocessing of features.
    
    This class is responsible for:
    1. Creating preprocessing pipelines for numerical and categorical features
    2. Applying transformations to training and testing data
    3. Saving the preprocessor for later use
    """
    
    def __init__(self):
        """
        Initialize the DataTransformation class with configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create preprocessing pipelines for numerical and categorical features.
        
        This method creates separate preprocessing pipelines:
        - Numerical features: Handle missing values and scale the data
        - Categorical features: Handle missing values, encode categories, and scale
        
        Returns:
            ColumnTransformer: A preprocessor object that can transform data
        
        Raises:
            CustomException: If any error occurs during pipeline creation
        """
        try:
            # Define which columns are numerical (continuous values)
            numerical_columns = ["writing_score", "reading_score"]
            
            # Define which columns are categorical (discrete categories)
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    # Replace missing values with median (middle value)
                    ("imputer", SimpleImputer(strategy="median")),
                    # Scale the features to have mean=0 and standard deviation=1
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            
            # Create pipeline for categorical features
            cat_pipline = Pipeline(
                steps=[
                    # Replace missing values with most frequent value
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # Convert categorical variables to numerical using one-hot encoding
                    ("onehot", OneHotEncoder()),
                    # Scale the encoded features
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns : {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine both pipelines into a single preprocessor
            # This will apply the appropriate pipeline to each type of column
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipline", cat_pipline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Apply preprocessing to training and testing datasets.
        
        This method:
        1. Loads the training and testing datasets
        2. Separates features (X) from target variable (y)
        3. Applies the preprocessor to transform the features
        4. Combines transformed features with target variable
        5. Saves the preprocessor for later use
        
        Args:
            train_path (str): Path to the training data file
            test_path (str): Path to the testing data file
        
        Returns:
            tuple: Transformed training array, transformed testing array, and preprocessor path
        
        Raises:
            CustomException: If any error occurs during transformation
        """
        try:
            # Load the training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            # Define the target variable (what we want to predict)
            target_column_name = "math_score"
            
            # Define numerical columns (these will be used as features)
            numerical_columns = ["writing_score", "reading_score"]

            # Separate features (X) from target variable (y) for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features (X) from target variable (y) for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessor object on training dataframe and testing dataframe")

            # Apply preprocessing to training data (fit and transform)
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            
            # Apply preprocessing to testing data (only transform, don't fit)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target variable
            # np.c_ concatenates arrays along the second axis (columns)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            # Save the preprocessor object for later use in prediction
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )
            
            # Return the transformed data and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
