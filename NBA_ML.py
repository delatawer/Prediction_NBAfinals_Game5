import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#This function takes a dictionary and a array to convert names to numbers
#For example months to numbers or the name of the teams
def tokenize_names(dict, names):
    cont = 0
    cont2 = 0
    for team in names:
        if team in dict:
            pass
        else:
            dict[team] = cont
            cont += 1
        names[cont2] = dict[team]
        cont2 += 1
    return dict, names

def format_output(data):
    y1 = data.pop('PTS')
    y1 = np.array(y1).reshape((len(y1), 1))
    y2 = data.pop('PTS.1')
    y2 = np.array(y2).reshape((len(y2), 1))
    return y1, y2

#We read the data we make from https://www.basketball-reference.com/leagues/NBA_2021_games-december.html
#we copy the data from the page and save it on the csv
data = pd.read_csv('data.csv')

#I'm gonna drop the notes, and two Unnamed columns, next we get rid of the "P" and ":" on the start hour
data.drop(columns='Notes', inplace=True)
data.drop(columns='Unnamed: 6', inplace=True)
data.drop(columns='Unnamed: 7', inplace=True)
data['Start (ET)'] = data['Start (ET)'].str.strip("p")
data['Start (ET)'] = data['Start (ET)'].str.replace(":", "").astype("int")

#We start the dictionary i'm going to use for the teams
dict = {}
data2 = data["Visitor/Neutral"]
data3 = data["Home/Neutral"]

#Call the function we make to pass the names to numbers
dict, data2 = tokenize_names(dict, data2)
dict, data3 = tokenize_names(dict, data3)

#Change the type to integer
data["Visitor/Neutral"] = data["Visitor/Neutral"].astype("int")
data["Home/Neutral"] = data["Home/Neutral"].astype("int")

#Split the Date to 3 different columns and drop the original column
date = data["Date"].str.split(" ", expand = True)
data["Day"] = date[2]
data["Month"] = date[1]
#data["Year"] = date[3]
data.drop(columns='Date', inplace=True)

#We declare the month dictionary and clean the data
dates_dict = {"Jan":1, "Feb": 2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Dec":12}
dates_dict, data["Month"] = tokenize_names(dates_dict, data["Month"])
data["Day"] =  data["Day"].astype("int")
data["Month"] = data["Month"].astype("int")
#data["Year"] = data["Year"].astype("int")
data["Attend."] = data["Attend."].fillna(0)
data["Attend."] = data["Attend."].astype("int")
data["PTS"] = data["PTS"].astype("int")
data["PTS.1"] = data["PTS.1"].astype("int")


#Here we normalize the data
#print(data["Attend."].max())
data["Day"] = data["Day"] / 31
data["Start (ET)"] = data["Start (ET)"] / 100
data["Attend."] = (data["Attend."] / data["Attend."].max()) * 50


#We shuffle the dataset because we are going to split the data into train/test
data = data.sample(frac=1).reset_index(drop=True)

#We divide the data into train/test
train_X, test_X = train_test_split(data, test_size=0.1)

#we convert the data to numpy arrays
train_Y = format_output(train_X)
test_Y = format_output(test_X)

#Define model layers
input_layer = Input(shape=(len(train_X .columns),));
first_dense = Dense(units='64', activation='relu')(input_layer);
second_dense = Dense(units='128', activation='relu')(first_dense);
third_dense = Dense(units='32', activation='relu')(second_dense);
fourth_dense = Dense(units='32', activation='relu')(second_dense);
y1_output = Dense(units='1', name='y1_output')(fourth_dense);
y2_output = Dense(units='1', name='y2_output')(third_dense);
model = Model(inputs=input_layer, outputs=[y1_output, y2_output]);
print(model.summary())

#optimizer = tf.keras.optimizers.SGD
model.compile(optimizer='Adam',
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})


history = model.fit(train_X, train_Y,
                    epochs=500, batch_size=5, validation_data=(test_X, test_Y))


#Make the prediction
#Time is 9 pm
#Bucks is away (7)
#Suns is home (23)
#Expected attendance (average of previous two suns home games)
#Day -> 0.54.. (17 / 31)
#Month -> 7
today_game = [9.0, 7, 23, 44.4856, 0.54838, 7]
today_game = np.array(today_game).reshape((1, 6))
Y_pred = model.predict(today_game)
print(Y_pred)



