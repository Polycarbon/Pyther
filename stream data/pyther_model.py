from keras.models import model_from_json
from keras.models import Model
from keras.layers import Dense,Input,Conv1D,MaxPooling1D,Flatten,LSTM,Dropout,BatchNormalization,Activation,Concatenate
from keras.preprocessing import sequence
import numpy as np


class pyther_model(object):

    def __init__(self):
        json_file = open("model_pyther-107sample-178dp.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model_pyther-107sample-178dp.h5")
        #self.model.summary()

    def preprocess(self,dl,dr,batch_size):
        xl = dl.T.values
        if(len(xl[0])<batch_size):
            xl = sequence.pad_sequences(xl, padding="post",maxlen=batch_size,dtype='float32')
        xr = dr.T.values
        if(len(xr[0])<batch_size):
            xr = sequence.pad_sequences(xr, padding="post",maxlen=batch_size,dtype='float32')
        return np.concatenate((xl, xr), axis=0).reshape(1,36,batch_size)
    
    def predict_gesture(self,x):
        array_p=self.model.predict([x])
        predict=[np.argmax(i) for i in array_p]
        return  np.array(predict)


         