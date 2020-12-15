# CS-3080-TFProject

#Final Report#
TensorFlow project
Derrick Kluck, George Race, Cameron Schlonski

API for weather data = http://api.worldweatheronline.com
Basis for TF model = https://keras.io/examples/timeseries/timeseries_weather_forecasting/

Takes in user input for range of dates and gathers weather data.
Reads weather data into a dictionary with keys about dates and temperatures
Reads dictionary into a dataframe to be used for model training.
Graphs weather data by total and first year of data
Splits data into groups for training and validation
Using parameters, trains the model using the training data set
  Documents loss and validation loss
Displays loss/val loss in console when training model using parameters
Applies model to validation set and graphs error in the prediction
Displays model predictions and the loss/val loss in console
