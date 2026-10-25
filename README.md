# Chlorophyll Concentration Predictor

This project contains machine learning training scripts to find the best parameters and compare three ML models. The primary goal is to predict today's chlorophyll concentration based on historical data. 

## Model Performance

The best-performing model achieved the following evaluation metrics on the test set:
* **$R^2$ Score:** 0.6262
* **Mean Absolute Error (MAE):** 0.1912
* **Mean Squared Error (MSE):** 0.1723
* **Root Mean Squared Error (RMSE):** 0.4151

*Note: The best model is automatically saved after training for future inference.*

## Project Structure

* **`train.py`**  
  The main training script. It evaluates and compares three different models to find the most accurate one:  
  * Linear Regression
  * Ridge Regression
  * Random Forest

* **`utils.py`**  
  Contains utility functions that support the main training pipeline:
  * **Lag Features:** Creates time-lagged variables to assist in time-series forecasting.
  * **Data Splitting:** Functions to safely separate the dataset into training and testing sets while respecting time dependency.
  * **Performance Assessment:** Calculates evaluation metrics such as $R^2$, MAE, and MSE.

* **`*.csv`**  
  The raw dataset utilized for training and testing the models.

* **`*.pkl`**  
  The final trained model exported as a pickle file, ready to be loaded for future predictions.
