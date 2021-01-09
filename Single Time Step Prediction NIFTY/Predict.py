import csv
import numpy as np
from numpy import random
from random import choices
from random import random

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.externals import joblib

def get_random_number():
    m=random()
    choice=[25,75,125,175,225,275,325,375,425,525]
    probabilities=[0.24,0.23,0.18,0.14,0.11,0.036,0.02,0.0103,0.0103,0.005155]
    return(float(choices(choice,probabilities)[0])*m)



scaler_filename='C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/scalar.save'
scaler=joblib.load(scaler_filename)
model=load_model('C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/Trained_Model.h5')

f=open('C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/Data/Read.csv','r')
data=csv.reader(f)
i=0
series=list()
for row in data:
    if(i==0):
        i=i+1
        continue
    series.append([row[1]])
    i=i+1
f.close()


test_P=series[:i-1]
test_N=series[:i-1]


f=open('C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/Data/Write.csv','a',newline='')
wr=csv.writer(f,delimiter=',')

for i in range(0,50):

    test_P=scaler.transform(test_P)
    test_N=scaler.transform(test_N)
    
    test_PX=list()
    test_NX=list()
    
    for j in range(i-51,i-1):
        test_PX.append(test_P[j])
        test_NX.append(test_P[j])
        
    test_PX=np.asarray(test_PX)
    test_PX=np.reshape(test_PX,(1,1,len(test_PX)))
    
    test_NX=np.asarray(test_NX)
    test_NX=np.reshape(test_NX,(1,1,len(test_NX)))
    
    prediction_P=model.predict(test_PX)
    prediction_N=model.predict(test_NX)
    
    prediction_P=np.reshape(prediction_P,(len(prediction_P),1))
    prediction_N=np.reshape(prediction_N,(len(prediction_N),1))
    
    prediction_P=scaler.inverse_transform(prediction_P)
    prediction_N=scaler.inverse_transform(prediction_N)
    
    prediction_P=prediction_P.reshape(len(prediction_P),)
    prediction_N=prediction_P.reshape(len(prediction_N),)
    
    rn=get_random_number()
    
    prediction_P=round(prediction_P[0]+rn,2)
    prediction_N=round(prediction_N[0]-rn,2)
    
    row=[prediction_P,prediction_N]
    wr.writerow(row)
    
#    print(prediction_P,prediction_N)
    
    
    test_P=scaler.inverse_transform(test_P)
    test_N=scaler.inverse_transform(test_N)
    
    test_P=test_P.tolist()
    test_N=test_N.tolist()
    
    test_P.append([prediction_P])
    test_N.append([prediction_N])

f.close()