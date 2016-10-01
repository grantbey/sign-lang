import os
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def interpolate(data):

    def nanhelper(x):
        return np.isnan(x), lambda z: z.nonzero()[0]
    holder = np.ndarray(0)
    for i in range(1,23):
        scaffold = np.full(135,np.nan)
        dim = data.shape[0]/22
        current_var = data[i*dim-dim:i*dim]
        randpts = np.sort(np.random.choice(np.linspace(0,134,135,dtype=np.intp),dim))
        scaffold[randpts] = current_var[:]
        nans,x = nanhelper(scaffold)
        scaffold[nans] = np.interp(x(nans),x(~nans),scaffold[~nans])
        holder = np.append(holder,scaffold)
    return holder

def get_data():
    x = np.empty(0)
    y = np.empty(0)
    for i in os.listdir(os.getcwd() + '/tctodd'):
        if not i.endswith('.DS_Store'):
            for fn in os.listdir(os.getcwd() + '/tctodd/' + i):
                if fn.endswith("tsd"):
                    action = re.search('(.+?)-[0-9]',fn).group(1)
                    data_current = np.loadtxt(os.getcwd() + '/tctodd/' + i + '/' + fn,delimiter='\t').ravel(order='F')
                    interp_data = interpolate(data_current)
                    data = np.append(data,interp_data)
                    y = np.append(y,action)
            print 'Done with directory ' + i # Provide status updates
    return data.reshape((len(data)/2970,2970)), y

x,y = get_data()
X_train, X_test, y_train, y_test = train_test_split(x, y)

scaler = StandardScaler() # instantiate the scaler
scaler.fit(X_train) # Fit the model to the training data
X_train_scaled = scaler.transform(X_train) # transform the training data
X_test_transformed = scaler.transform(X_test) # transform the testing data

pca = PCA(100) # instantiate the model
pca.fit(X_train_scaled) # Fit the model to training data
X_train_PCA = pca.transform(X_train) # transform the training data
X_test_PCA = pca.transform(X_test) # transform the testing data

from sklearn.svm import LinearSVC
svm = LinearSVC(C=0.1)
svm.fit(X_train_PCA, y_train)
svm.score(X_train_PCA, y_train)
svm.score(X_test_PCA, y_test)