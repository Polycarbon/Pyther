from Servers import  Server
import os
import time
import pickle
import hickle as hkl
from multiprocessing import Process,Queue
from threading import Thread
import pandas as pd
from pyther_model import pyther_model
from time import sleep
import queue as queue
from sklearn import preprocessing
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
def handle(ports,queue):
    print(ports)
    s = Server(("192.168.30.150", ports),queue)
    connection, client_address = s.getClient()
    print("Process at ", os.getpid())
    print("CLear")

def update_data(ql,qr):
    colnames = ['Batt','Time',
               'Raw-Ch1','Raw-Ch2','Raw-Ch3','Raw-Ch4','Raw-Ch5','Raw-Ch6','Raw-Ch7','Raw-Ch8',
               'Rect-Ch0','Rect-Ch1','Rect-Ch2','Rect-Ch3','Rect-Ch4','Rect-Ch5','Rect-Ch6','Rect-Ch7',
               'Smooth-Ch0','Smooth-Ch1','Smooth-Ch2','Smooth-Ch3','Smooth-Ch4','Smooth-Ch5','Smooth-Ch6','Smooth-Ch7',
               'q1','q2','q3','q4',
               'ax','ay','az',
               'gx','gy','gz',
               'mx','my','mz']
    feature_select = ["Time",'Smooth-Ch0','Smooth-Ch1','Smooth-Ch2','Smooth-Ch3','Smooth-Ch4','Smooth-Ch5','Smooth-Ch6','Smooth-Ch7',
            'q1','q2','q3','q4','ax','ay','az','gx','gy','gz']
    dl = pd.DataFrame([],columns=colnames)
    dr = pd.DataFrame([],columns=colnames)
    gp = pyther_model()
    gesture_name=['1', '10', '11', '12', '13', '14l', '14r', '15l', '15r', '2', '3',
       '4', '5', '6', '7', '8', '9']
    #gesture_name=['1', '10', '5', '6', '9']
    e=preprocessing.LabelEncoder()
    e.fit(gesture_name)
    batch_size = 2000
    step_size = 2000
    print("build predict")
    ans=[]
    while(True):
        if(not ql.empty() and not qr.empty()):
            l=ql.get()
            dl=dl.append(pd.Series(l, index=colnames),ignore_index=True)
            r=qr.get()
            dr=dr.append(pd.Series(r, index=colnames),ignore_index=True)
            if(dl.shape[0]>=500):
                x = gp.preprocess(dl[feature_select],dr[feature_select],batch_size)
                (gesture,p) = gp.predict_gesture(x)

                ans.append(gesture)
                print(e.inverse_transform(gesture))
                dl = pd.DataFrame([],columns=colnames)
                dr = pd.DataFrame([],columns=colnames)
                # dl=dl.tail(batch_size-step_size)
                # dr=dr.tail(batch_size-step_size)
            if(len(ans)==15 ):
                g,count = stats.mode(ans)
                print("answer : "+str(e.inverse_transform(g)))
                ans.clear()

            
if __name__ == "__main__":
    ql = Queue()
    L = Thread(target=handle, args=(9003,ql))
    L.daemon = True
    L.start()
    qr = Queue()
    R = Thread(target=handle, args=(9004,qr))
    R.daemon = True
    R.start()
    t = Thread(target=update_data , args=(ql,qr))
    t.start()
    R.join()
    L.join()
