from __future__ import division, print_function
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
import itertools
def read_emg(filename):
    df= pd.read_csv(filename,names=['Batt','Time','C',
                                           'Raw-Ch1','Raw-Ch2','Raw-Ch3','Raw-Ch4','Raw-Ch5','Raw-Ch6','Raw-Ch7','Raw-Ch8','L',
                                           'Rect-Ch0','Rect-Ch1','Rect-Ch2','Rect-Ch3','Rect-Ch4','Rect-Ch5','Rect-Ch6','Rect-Ch7','U',
                                           'Smooth-Ch0','Smooth-Ch1','Smooth-Ch2','Smooth-Ch3','Smooth-Ch4','Smooth-Ch5','Smooth-Ch6','Smooth-Ch7','AD',
                                           'q1','q2','q3','q4','AI',
                                           'ax','ay','az','AM',
                                           'gx','gy','gz','AQ',
                                           'mx','my','mz',"AU"])
    d=df.loc[:,['Time','Raw-Ch1','Raw-Ch2','Raw-Ch3','Raw-Ch4','Raw-Ch5','Raw-Ch6','Raw-Ch7','Raw-Ch8','Rect-Ch0','Rect-Ch1','Rect-Ch2','Rect-Ch3','Rect-Ch4',
            'Rect-Ch5','Rect-Ch6','Rect-Ch7','Smooth-Ch0','Smooth-Ch1','Smooth-Ch2','Smooth-Ch3','Smooth-Ch4','Smooth-Ch5','Smooth-Ch6','Smooth-Ch7',
            'q1','q2','q3','q4','ax','ay','az','gx','gy','gz']]
    return d
        
def plot_EMG(time,emgs,gesture_name,normalize=False):
    if normalize:
        emgs = normalize_EEG(emgs)
    fig = plt.figure("EMG")
    ticklocs = []
    numRows=8
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(np.arange(10))
    dmin = emgs.min().min()
    dmax = emgs.max().max()
    dr = (dmax - dmin) * 1  # Crowd them a bit.
    y0 = 1
    y1 = (numRows - 1) * dr + dmax
    plt.ylim(y0, y1)
    
    segs = []
    for i,ch in enumerate(emgs[:8]):
        segs.append(np.hstack((time[:, np.newaxis], emgs[ch][:, np.newaxis])))
        ticklocs.append((i+0.25) * dr)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax.add_collection(lines)

    # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)
    ax.set_yticklabels( [ 'Raw-Ch1','Raw-Ch2','Raw-Ch3','Raw-Ch4','Raw-Ch5','Raw-Ch6','Raw-Ch7','Raw-Ch8'])
    ax.set_title(gesture_name)
    ax.set_xlabel('Time (s)')
    
    return ax,emgs

def normalize_EEG(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def confuse_matrix_plot(Y,Pred,list_class,title="confusion matrix"):
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y, Pred)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list_class, normalize=False,
                          title=title)
    plt.show()
      # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list_class, normalize=True,
                          title=title +'(Normalized)')
    plt.show()


