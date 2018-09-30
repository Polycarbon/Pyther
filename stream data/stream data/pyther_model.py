from keras.models import model_from_json
from keras.models import Model
from keras.layers import Dense,Input,Conv1D,MaxPooling1D,Flatten,LSTM,Dropout,BatchNormalization,Activation,Concatenate
from keras.preprocessing import sequence
import numpy as np


class pyther_model(object):

    def __init__(self):
        json_file = open("model_pyther-LSTM.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model_pyther-LSTM.h5")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #self.model.summary()

    def preprocess(self,dl,dr,batch_size):
        T1 =  find_active_time(dl.Time,dl.loc[:,['gx','gy','gz']])
        T2 =  find_active_time(dr.Time,dr.loc[:,['gx','gy','gz']])
        if (len(T1)==2):
            T = (np.array(T1)+np.array(T2))/2
            xl = dl[dl.Time>=T[0]].T.values[1:]
            xr = dr[dr.Time>=T[0]].T.values[1:]
        #xl = dl.T.values[1:]
        if(len(xl[0])<batch_size):
            xl = sequence.pad_sequences(xl, padding="post",maxlen=batch_size,dtype='float32')
        #xr = dr.T.values[1:]
        if(len(xr[0])<batch_size):
            xr = sequence.pad_sequences(xr, padding="post",maxlen=batch_size,dtype='float32')
        return np.concatenate((xl, xr), axis=0).reshape(1,36,batch_size)

        
    def predict_gesture(self,x):
        array_p=self.model.predict([x])
        predict=[np.argmax(i) for i in array_p]
        return  (np.array(predict)[0],array_p)


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def find_active_time(time,emg):
    T1 = []
    T2 = []
    for i in emg:
        x = time
        y = emg[i]

        lag=int(len(y)*1/5)
        threshold=3.6
        influence=0.6
        result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
        v=np.abs(result["signals"])
        if(len(np.where(v==1)[0])>2):
            t1=x[np.where(v==1)[0][0]]
            t2=x[np.where(v==1)[0][-1]]
            T1.append(t1)
            T2.append(t2)
    t1 = np.mean(T1)
    t2 = np.mean(T2)
    return [t1,t2]         