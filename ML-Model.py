
#Import Modules 
!pip install -q seaborn

!pip install boto3

import boto3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

#Importing Dataset --> DATA-01 from UCI

from google.colab import drive 
drive.mount('/content/drive')

import pandas as pd
import requests
from io import StringIO

orig_url='https://drive.google.com/file/d/1thTX13xrV21klv1MZxZLpbmzE8heIDDF/view?usp=sharing'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)

column_names = ['Date','Time','HGlucose','SGlucose']

raw_dataset = pd.read_csv(csv_raw, names=column_names,
                          na_values='?',comment ='\t',
                          sep='\t', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

#drop code column 

dataset = dataset.loc[(dataset["Date"] == 100220170000)]
print(dataset)
dataset.drop(["SGlucose"], axis = 1, inplace = True) 
dataset.drop(["Date"], axis = 1, inplace = True)
dataset.drop_duplicates(subset=['Time'], inplace = True)
dataset.sort_values(by=['Time', 'HGlucose'], inplace = True)


print(dataset)
dataset.()

"""Clean Data Set for Training """

dataset.isna().sum()

dataset = dataset.dropna()

"""Splitting Dataset into train and test

"""

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['HGlucose']], diag_kind='kde')

"""Overall Statistics of HGlucose distribution"""

train_dataset.describe().transpose()

"""Split features from labels"""

#training algorithim to predict HGlucose 

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('HGlucose')
test_labels = test_features.pop('HGlucose')

"""Normalization"""

train_dataset.describe().transpose()[['mean', 'std']]

"""Normalization Layer 

"""

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

"""Linear Regression

Normalization Layer for One Variable --> Time
"""

Time = np.array(train_features['Time'])
normalizer = preprocessing.Normalization(input_shape=[1,])
normalizer.adapt(Time)

"""Build Seqeuntial Model:"""

Time_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

Time_model.summary()

Time_model.predict(Time[:10])

Time_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = Time_model.fit(
#     train_features['Time'], train_labels,
#     epochs= 1000,
#     # suppress logging
#     verbose=2,
#     # Calculate validation results on 20% of the training data
#     validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 300])
  plt.xlabel('Epoch')
  plt.ylabel('Error [HGlucose]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

test_results = {}

test_results['Time_model'] = Time_model.evaluate(
    test_features['Time'],
    test_labels, verbose=2)

x = tf.linspace(0.0, 2500, 2501)
y = Time_model.predict(x)

def plot_time(x, y):
  plt.scatter(train_features['Time'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Time')
  plt.ylabel('HGlucose')
  plt.legend()

plot_time(x,y)

"""Multiple Inputs 

"""

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = linear_model.fit(
#     train_features, train_labels, 
#     epochs=100,
#     # suppress logging
#     verbose=0,
#     # Calculate validation results on 20% of the training data
#     validation_split = 0.2)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

"""DNN Regression """

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(128, activation='swish'),
      layers.Dense(128, activation='sigmoid'),
      layers.Dense(128, activation='tanh'),
      layers.Dense(64, activation='gelu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_Time_model = build_and_compile_model(normalizer)

dnn_Time_model.summary()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = dnn_Time_model.fit(
#     train_features['Time'], train_labels,
#     validation_split=0.2,
#     verbose=0, epochs= 6000)

plot_loss(history)

x = tf.linspace(0.0, 2999, 3000)
y = dnn_Time_model.predict(x)

plot_time(x, y)

test_results['dnn_Time_model'] = dnn_Time_model.evaluate(
    test_features['Time'], test_labels,
    verbose=2)

"""Full Model"""

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = dnn_model.fit(
#     train_features, train_labels,
#     validation_split=0.2,
#     verbose=2, epochs=6000)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [HGlucose]']).T

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [HGlucose]')
plt.ylabel('Predictions [HGlucose]')
lims = [0, 200]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [HGlucose]')
_ = plt.ylabel('Count')

dnn_model.save('dnn_model')

reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [HGlucose]']).T
