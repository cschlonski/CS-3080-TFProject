"""
TensorFlow project

Note that my interpreter is anaconda, the
command to install tensorflow 2.3 from terminal:
pip install tensorflow==2.3.0

I was having a hard time getting everything to function correctly with
the API directly so I instead loaded all the data into a .csv file
that I access later. The problem I'm running into now is from lack of
data since I can only get access to 35 days of information at a time.
This is why my step, past, future and batch_size values are so low.

I got everything to function correctly but the only data I am using
is the maxtempF, mintempF, avgtempF from each day in the range
"""

import csv
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# this is the API that we will be down loading our data from
def api_search(code, start, end):
    url = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6230647e6745477995c00355202410&q=" + code + "&format=json&date=" + start + "&enddate=" + end
    return url


"""
code=input('Enter a zip code:')
start=input('Enter a start date in the format yyyy-mm-dd:')
end=input('Enter an end date in the format yyyy-mm-dd:')
"""
# to make everything move faster i hard coded in some values
code = str('80922')
start = str('2018-09-09')
end = str('2020-09-09')

url = api_search(code, start, end)
obj = urllib.request.urlopen(url)
rawdata = json.load(obj)
data = rawdata['data']

# now we will open a file for writing
data_file = open('data_file.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(data_file)

# counter variable used for writing
# headers to the CSV file
count = 0

# loads the data from the API into the .csv file
for item in data['weather']:
    if count == 0:
        header = item.keys()
        csv_writer.writerow(header)
        count += 1

    csv_writer.writerow(item.values())
data_file.close()

csv_path = "data_file.csv"
df = pd.read_csv(csv_path)

# these are the titles for our columns that we created in our CSV file
titles = ["date", "astronomy", "maxtempC", "maxtempF",
          "mintempC", "mintempF", "avgtempC", "avgtempF",
          "totalSnow_cm", "sunHour", "uvIndex", "hourly"
          ]
# these are the keys that we will later be using to access the data
feature_keys = ["date", "astronomy", "maxtempC", "maxtempF",
                "mintempC", "mintempF", "avgtempC", "avgtempF",
                "totalSnow_cm", "sunHour", "uvIndex", "hourly"
                ]
# colors that we can use when graphing the data later on
colors = ["blue", "orange", "green", "red", "purple",
          "brown", "pink", "gray", "olive", "cyan",
          ]

# split_fraction is the percentage of the incoming data that we will be using to train our model with
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))  # df.shape[0] is the number of total data inputs per a column
step = 1  # step is the rate at which we move through our samples
past = 3  # 3 timestamps of previous data at 24hrs per timestamp
future = 1  # predict 1 timestamp
learning_rate = 0.001
# note that the batch_size MUST divide df.shape[0] with 0 as a remainder
batch_size = 1  # batch_size is number of samples being propagated through the network
epochs = 10  # epochs are the number of times we go trough the our training set


# this is a function that will normalize our data, the reason we are doing this is to reduce the
# the complexity and increase the accuracy of our training
def normalize(data, train_split_fun):
    data_mean = data[:train_split_fun].mean(axis=0)
    data_std = data[:train_split_fun].std(axis=0)
    return (data - data_mean) / data_std


# here we are printing out the data fields we are going to be analyzing
print("\nThe selected data fields are:", ", ".join([titles[i] for i in [3, 5, 7]]), )
print()
# note that I am just inserting 3, 5, 7 because that is the maxtempF, mintempF, avgtempF
dataField = [feature_keys[i] for i in [3, 5, 7]]
averages = df[dataField]  # we assign averages to get the min max and avg from our CSV
averages.index = df["date"]  # this will organize our data by date
averages = normalize(averages.values, train_split)
averages = pd.DataFrame(averages)  # this will organize our normalized data into a 2 dimensional array
# the sample of data we are inserting into the model for training
train_data = averages.loc[0: train_split - 1]
# val-data is a data set held back from our training model that we use to estimate the model skill
val_data = averages.loc[train_split:]
##############################################################################################


# note that since matrices are being used the len(x_train) must equal len(y_train) in order
# for matrix multiplication to take place
# these are the variables needed for our dataset_train
start = past + future
end = start + train_split
x_train = train_data[[i for i in range(len(dataField))]].values
y_train = averages.iloc[start:end][[1]]
sequence_length = int(past / step)

# this function takes in the train_data sequence of data points gathered at equal intervals
# to produce batches of timeseries inputs and targets.
dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,  # length of the output sequence
    sampling_rate=step,
    batch_size=batch_size,
)

# these are the variables needed for our dataset_val
x_end = len(val_data) - past - future
label_start = train_split + past + future
x_val = val_data.iloc[:x_end][[i for i in range(len(dataField))]].values
y_val = averages.iloc[label_start:][[1]]

# this function takes in our val_data sequence of data-points gathered at equal intervals
# to produce batches of timeseries inputs and targets.
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,  # length of the output sequence
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

print()
print("Input shape:", inputs.numpy().shape)  # numpy array containing consecutive data points, timesteps
print("Target shape:", targets.numpy().shape)  # targets correspond to timestep in data, should have same length as data
#######################################################################################

# we are creating the first layer of our network to receive the data for training
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)  # calling the Long Short Term Memory function
outputs = keras.layers.Dense(1)(lstm_out)  # the dense layer implements the training

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()
#######################################################################################


path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)


######################################################


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")
###########################################################
