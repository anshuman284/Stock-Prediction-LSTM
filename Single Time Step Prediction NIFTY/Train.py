import csv
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib


f=open('C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/Data/NIFTY_50_Data.csv','r')
data=csv.reader(f)
i=0
series=list()
for row in data:
    if(i==0):
        i=i+1
        continue
    series.append([row[1]])
    i=i+1
train=series[:i-1]

scaler = MinMaxScaler(feature_range=(0,1))
scaler=scaler.fit(train)
train=scaler.transform(train)

trainX=list()
trainY=list()

for j in range(51,i):
    trainX.append(train[j-51:j-1])
    trainY.append(train[j-1])
  
trainX=np.asarray(trainX)
trainY=np.asarray(trainY)


trainX=np.reshape(trainX,(len(trainX),1,trainX.shape[1]))
trainY=np.reshape(trainY,(len(trainY),1,1))

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN

keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model=Sequential()
model.add(LSTM(4,input_shape=(1,trainX.shape[2]),return_sequences=True))
#model.add(LSTM(100,return_sequences=True))
#model.add(Dense(1))
#model.add(LSTM(100,return_sequences=True))
#model.add(Dense(1))
#model.add(LSTM(100,return_sequences=True))
#model.add(Dense(1))
#model.add(LSTM(100,return_sequences=True))
#model.add(Dense(1))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX,trainY,epochs=500,batch_size=50,verbose=1)

prediction=model.predict(trainX)
prediction=np.reshape(prediction,(len(prediction),1))
prediction=scaler.inverse_transform(prediction)


test=series[i-1-200:i-1]
test=np.array(test)
test=test.reshape(len(test),)


prediction=prediction.reshape(len(prediction),)

f=open('C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/Data/Train.csv','w',newline='')
row=['Test','Predicted']
wr = csv.writer(f,delimiter=',')
wr.writerow(row)
diff=list()
k=len(prediction)
for p in range(0,len(test)):
    diff.append(float(test[p])-float(prediction[k-200+p]))
    row=[test[p],prediction[k-200+p]]
    wr.writerow(row)
f.close()

diff=np.array(diff)

from matplotlib import pyplot as plt

plt.plot(diff)
#plt.plot(prediction)
plt.show()

from sklearn.externals import joblib
scaler_filename='C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/scalar.save'
joblib.dump(scaler,scaler_filename)
model.save('C:/Users/Anshuman/Desktop/Anshuman/Machine_Learning/LSTM_2/Trained_Model.h5')
