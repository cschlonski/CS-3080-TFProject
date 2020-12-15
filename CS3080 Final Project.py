#Pulls the data from the api with a start and end date
def api_search(code,start,end):
    url="http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6230647e6745477995c00355202410&q="+code+"&format=json&date="+start+"&enddate="+end +"&tp=24"
    return url

import json
import urllib.request
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy

#allows user to enter what they want to search
code=input('Enter a zip code:')
start=input('Enter a start date in the format yyyy-mm-dd:')
end=input('Enter an end date in the format yyyy-mm-dd:')
graph_type=input('Do you want the data displayed daily, monthly, or yearly?')

# Recommend using large data set I.e. 2010-2019 daily
#code = "80924"
#start = "2010-01-01"
#end = "2019-01-01"
#graph_type = "daily"

minTemp={}
maxTemp={}
windSpeed={}
humidity={}
totTemp = {}

#splits the start and end date into groups
dateRegex=re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
mo=dateRegex.search(start)

dateRegex=re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
mo2=dateRegex.search(end)

start2=start
month=int(mo.group(2))+1
if month>9:
    end2 = mo.group(1) + '-' + str(month) + '-' + mo.group(3)
else:
    end2=mo.group(1)+'-'+'0'+str(month)+'-'+'01'
j=2

#pulls the data from the api seperately since the max amount of dates is 35. This allows for unlimited time
while(end2!=end):
    url = api_search(code, start2, end2)
    obj = urllib.request.urlopen(url)

    rawdata = json.load(obj)

    data = rawdata['data']

    # pulls the data and places it into dictionaries
    # Modified to allow for a dictionary with the total temperatures
    for item in data['weather']:
        i = 0
        date = item['date']
        minTemp.update({date: item['mintempF']})
        maxTemp.update({date: item['maxtempF']})
        totTemp.setdefault("date", []).append(date)
        totTemp.setdefault("MinTemp", []).append(int(item['mintempF']))
        totTemp.setdefault("MaxTemp", []).append(int(item['maxtempF']))
        for item2 in item['hourly']:
            test = item['hourly']
            test2 = test[i]
            windSpeed.update({date: test2['windspeedMiles']})
            humidity.update({date: test2['humidity']})
            i = i + 1

    dateRegex = re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
    mo3 = dateRegex.search(end2)
    day = int(mo3.group(3)) + 1
    if day > 9:
        start2 = mo3.group(1) + '-' + mo3.group(2) + '-' + str(day)
    else:
        start2 = mo3.group(1) + '-' + mo3.group(2) + '-' + '0' + str(day)

    month = int(mo.group(2)) + j

    if month > 9:
        end2 = mo3.group(1) + '-' + str(month) + '-' + '01'
    else:
        end2 = mo3.group(1) + '-' + '0' + str(month) + '-' + '01'
    j = j + 1

    dateRegex = re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
    mo4 = dateRegex.search(start2)
    if (mo4.group(2) == '12'):
        end2 = str(int(mo4.group(1)) + 1) + '-01-' + '01'
        j = 1

    if (mo2.group(1) == mo4.group(1) and mo2.group(2) == mo4.group(2)):
        end2 = end

url = api_search(code,start2,end2)

obj = urllib.request.urlopen(url)

rawdata = json.load(obj)

data = rawdata['data']

print("\n")

#pulls the data and places it into dictionaries
for item in data['weather']:
    i=0
    date=item['date']
    minTemp.update({date: item['mintempF']})
    maxTemp.update({date: item['maxtempF']})
    for item2 in item['hourly']:
        test = item['hourly']
        test2 = test[i]
        windSpeed.update({date: test2['windspeedMiles']})
        humidity.update({date: test2['humidity']})
        i = i + 1


finalMinTemp=[]
finalMaxTemp=[]
finalwindSpeed=[]
finalHumidity=[]

#puts the data into an array so it can be graphed. Will also average the data based on month or year if needed
if(graph_type=='daily'):
    for key, value in minTemp.items():
        finalMinTemp.append(int(value))
    for key, value in maxTemp.items():
        finalMaxTemp.append(int(value))
    for key, value in windSpeed.items():
        finalwindSpeed.append(int(value))
    for key, value in humidity.items():
        finalHumidity.append(int(value))

elif(graph_type=='monthly'):
    total = 0
    count=0
    temp = ''
    for key, value in minTemp.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMinTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMinTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in maxTemp.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMaxTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMaxTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in windSpeed.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalwindSpeed.append(average)
            total=int(value)
            count=1
    average = total / count
    finalwindSpeed.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in humidity.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalHumidity.append(average)
            total=int(value)
            count=1
    average = total / count
    finalHumidity.append(average)

else:
    total = 0
    count=0
    temp = ''
    for key, value in minTemp.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMinTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMinTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in maxTemp.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMaxTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMaxTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in windSpeed.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalwindSpeed.append(average)
            total=int(value)
            count=1
    average = total / count
    finalwindSpeed.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in humidity.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalHumidity.append(average)
            total=int(value)
            count=1
    average = total / count
    finalHumidity.append(average)


# Graph the min and max temp data
plt.plot(list(range(0,len(finalMinTemp))), finalMinTemp, label='Min Temperature')
plt.plot(list(range(0,len(finalMaxTemp))), finalMaxTemp, label='Max Temperature')

# naming the x axis
plt.xlabel('Date - (Days)')
# naming the y axis
plt.ylabel('Temperature - (F)')
plt.legend()
plt.show()
plt.close()

# Graph the min and max temp data for 1 year
plt.figure()
plt.plot(list(range(0,365)), finalMinTemp[0:365], label='Min Temperature')
plt.plot(list(range(0,365)), finalMaxTemp[0:365], label='Max Temperature')

# naming the x axis
plt.xlabel('Date - (Days)')
# naming the y axis
plt.ylabel('Temperature - (F)')
plt.legend()
plt.show()


# The dictionary totTemp will apply its data to the tensorflow program below
###############################################################

# Creation of dataframe from the dictionary defined
df = pd.DataFrame(totTemp)

# these are the titles for our columns that we created in our dictionary defined earlier
titles = ["MinTemp", "MaxTemp"]
# these are the keys that we will later be using to access the data
feature_keys = ["MinTemp", "MaxTemp"]
# colors that we can use when graphing the data later on
colors = ["blue", "green"]
date_time_key = "date"
###############################################################


# split_fraction is the percentage of the incoming data that we will be using to train our model with
# The smaller the split_fraction, less data that will apply to the model
# With a larger split fraction, you will likely have higher val_loss with a smaller sample size
split_fraction = 0.7
train_split = int(split_fraction * int(df.shape[0]))    # df.shape[0] is the number of total data inputs per a column
step = 1    # step is the rate at which we move through our samples

past = 400  # 400 timestamps of previous data, preferably less than half of the total samples
future = 50  # predict 50 timestamp
learning_rate = 0.001

batch_size = round(int(df.shape[0])/100)  # batch_size is number of samples being propagated through the network
epochs = 15  # epochs are the number of times we go through the our training set
######################################################


# this is a function that will standardize our data, the reason we are doing this is to reduce the
# the complexity and increase the accuracy of our training by reducing the range of our values
def standardize(data, train_split_fun):
    data_mean = data[:train_split_fun].mean(axis=0)
    data_std = data[:train_split_fun].std(axis=0)
    return (data - data_mean) / data_std


# here we are printing out the data fields we are going to be analyzing
print(
    "\nThe selected data fields are:",
    ", ".join([titles[i] for i in [0]]),)
print()
# Note that I am just inserting 0, 1 because that is the maxtempF and mintempF
# If we had more keys, we would need to specify which keys we want to apply to dataField below
dataField = [feature_keys[i] for i in [0, 1]]
averages = df[dataField]    # we assign averages to get the min and max from the dictionary
averages.index = df[date_time_key]  # this index our data by date
print(averages.values)

averages = standardize(averages.values, train_split)
averages = pd.DataFrame(averages)   # this will organize our normalized data into a 2 dimensional array

# the sample of data we are inserting into the model for training
train_data = averages.loc[0: train_split - 1]
# val-data is a data set held back from our training model that we use to estimate the model skill
val_data = averages.loc[train_split:]
##################################################


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
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

##################################################
# these are the variables needed to create dataset_val
x_end = len(val_data) - past - future
label_start = train_split + past + future
x_val = val_data.iloc[:x_end][[i for i in range(len(dataField))]].values
y_val = averages.iloc[label_start:][[1]]

# this function takes in our val_data sequence of data-points gathered at equal intervals
# to produce batches of timeseries inputs and targets.
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

#################################################

# we are creating the first layer of our network to receive the data for training
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)    # calling the Long Short Term Memory function
outputs = keras.layers.Dense(1)(lstm_out)   # the dense layer implements the training

# Creating the model from the inputs and outputs defined from the batches of dataset_train
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()
####################################################


path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=3)

# A callback defined to stop running training the model if the val_loss does not improve 5 times total
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch'
)

# Training the model against the training data set and the validation set
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)
######################################################

# Visualize the loss
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


# Show plot function:
# Takes data and graphs the history of the data and the prediction
# Shows how well the prediction did compared to the true future
def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    print(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


# Takes 5 points from the validation data set and tests the model on the date based on the past and graphs true/prediction points
for x, y in dataset_val.take(6):
    show_plot(
        [x[0][0].numpy(), y[0].numpy(), model.predict(x)[0]],
        12,
        "Single Step Prediction",
    )

# Prints the predictions of the 5 data points in the validation set selected
for x, y in dataset_val.take(6):
    print(x[0][0].numpy())
    print(model.predict(x)[0])

# Prints the final losses
# Loss = difference between predicted values and the training dataset
# Val loss = difference between predictions and the validation dataset
for w in range(len(history.history["loss"])):
    print("Training set loss for epoch " + str(w) + " : " + str(history.history["loss"][w]))
    print("Validation set loss for epoch " + str(w) + " : " + str(history.history["val_loss"][w]))
