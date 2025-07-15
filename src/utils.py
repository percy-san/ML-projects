"""
Utility Functions Module

This module contains helper functions used throughout the project,
primarily for saving Python objects to files.
"""

import os
import sys
import numpy as np
import pandas as pd
import dill  # dill is like pickle but can handle more complex Python objects
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill serialization.
    
    This function is useful for saving machine learning models, 
    preprocessors, or any other Python objects that need to be 
    reused later.
    
    Args:
        file_path (str): The path where the object should be saved
        obj: The Python object to save (can be any serializable object)
    
    Raises:
        CustomException: If there's an error during the saving process
    """
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # If any error occurs, raise a custom exception with details
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
