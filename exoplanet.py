# importing libraries and custom classes required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import *
# %matplotlib inline
# loading test and train data
data = pd.read_csv(r'C:\Users\manas\Downloads\ExoTrain.csv')
print ('Data Loaded!')

from sklearn.model_selection import train_test_split
y = np.asarray(data.iloc[:, 0])
X = np.asarray(data.iloc[:, 1:])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_test=y_test.reshape(-1,1)
dfTest = pd.DataFrame(np.concatenate((y_test, X_test), axis = 1))

y_train=y_train.reshape(-1,1)
data = pd.DataFrame(np.concatenate((y_train, X_train), axis = 1))

data.head()

# Visualize data before noise removal
i = 4
plt.figure(figsize=(15, 8))
plt.plot(np.array(data.iloc[i,1:]))
print('Visualized data before noise removal')
# Removing noise from the data using custom made funtions to
# detect the outliers and normalize them.

f, k = 2.5, 1
def cal(mean, std, f):
    return mean + f * std

def modify(data, f, k):
    y_train = data.iloc[:, 0]
    X_train = data.iloc[:, 1:]
    y_train_2 = np.array(y_train, copy = False, subok = True, ndmin = 2).T
    X_train_lim = cal(X_train.mean(axis = 1), X_train.std(axis = 1), f)
    dump, n = X_train.shape
    masks = np.zeros((dump, n), dtype = np.bool_)
    for i in range(n):
        masks[:, i] = np.greater_equal(X_train.iloc[:, i], X_train_lim)
    X_out = np.copy(X_train)
    for i in range(n):
        if (i >= k and i < (n - k)):
            X_out[:, i] = (np.logical_not(masks[:, i]) * X_out[:, i]) + \
            masks[:, i] * (X_train.iloc[:, range(i - k, i)].mean(axis = 1) + \
                           X_train.iloc[:, range(i + 1, i + k + 1)].mean(axis = 1)) / 2
                        
        else:
            X_out[:, i] = (np.logical_not(masks[:, i]) * X_out[:, i]) +\
            masks[:, i] * (X_train.mean(axis = 1))
    return pd.DataFrame(np.concatenate((y_train_2, X_out), axis = 1))

data_n = modify(data, f, k)
dfTest_n = modify(dfTest, f, k)
print('detected the outliers and normalized them.')
# deleting the initial data structures.
del data
del dfTest

#data after noise removal
plt.figure(figsize=(15, 8))
plt.plot(data_n.iloc[i,1:])

# Oversampling the data using SMOTE from imblearn to
# create synthetic data to the balance the under represented class 
y_train = np.array(data_n.iloc[:, 0])
X_train = np.array(data_n.iloc[:, 1:])
#Resample using SMOTE
X_r, y_r = SMOTE(k_neighbors = 8, random_state = 32).fit_sample(X_train, y_train)
y_2 = np.array(y_r, copy=False, subok=True, ndmin=2).T
data_n_s = pd.DataFrame(np.concatenate((y_2, X_r), axis = 1))

# deleting the initial data structures.
del data_n

# Finding the first order differences
def diff(data):
    y = np.array(data.iloc[:, 0])
    X = np.array(data.iloc[:, 1:])
    y_2 = np.array(y, copy=False, subok=True, ndmin=2).T
    X = np.diff(X)
    return pd.DataFrame(np.concatenate((y_2, X), axis = 1))

data_n_s_1d = diff(data_n_s)
dfTest_n_1d = diff(dfTest_n)

# deleting the initial data structures.
del data_n_s
del dfTest_n

# synthetic data appended to the initial data
data_n_s_1d.tail(2)

# data after 1st order difference (for the 0th index time series flux data)
plt.figure(figsize=(15, 8))
plt.plot(data_n_s_1d.iloc[0,1:])

# setting up the features and Labels to be put to the classifier
y_test = np.array(dfTest_n_1d.iloc[:, 0])
X_test = np.array(dfTest_n_1d.iloc[:, 1:])
y = np.array(data_n_s_1d.iloc[:, 0])
X = np.array(data_n_s_1d.iloc[:, 1:])

# applying the classifier and predicting on the amplified feature.
clf_r = RandomForestClassifier(random_state = 0)
clf_r = clf_r.fit(X, y)
prediction_r = clf_r.predict(X_test)
(pd.DataFrame(prediction_r)).head()

# np.savetxt('Expected_Labels_FINAL(SRSLY).csv', prediction_r, fmt = '%.1u', header = 'LABEL', delimiter = ',')

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction_r)

from sklearn.metrics import classification_report
print (classification_report(y_test,prediction_r))

accuracy_score(y_test, prediction_r)

# index [starting form 0] which was predicted to be the exoplanets
np.nonzero(prediction_r - 1)