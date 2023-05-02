import numpy as np
import os
from datetime import datetime, timedelta

# define example
labels = {'NONE' : 0,  'L_EYE' : 1, 'R_EYE' : 2, 'JAW_CLENCH' : 3, 'BROW_UP' : 4, 'BROW_DOWN': 5}

def getTitle(recordingFile):
    return recordingFile.split("-")[-1].split(".")[0].translate({ord(k): None for k in '0123456789'})

# pass none if dont want granularity
# pass none to dataLimit if want all the data
def getData(path, granularity, channels, dataLimit):
    dataRaw = []
    dataStartLine = 6
    count = 0
    with open(path, 'r') as data_file:
        for line in data_file:
            if count >= dataStartLine:
                    dataRaw.append(line.strip().split(','))
            else:
                    count += 1
    dataRaw = np.char.strip(np.array(dataRaw))

    dataChannels = dataRaw[:, 1:5]
    timeChannels = dataRaw[:, 15]

    if granularity is None:
        granularity = 1
    # the current channel of data
    if dataLimit is None:
        dataLimit = len(dataChannels)

    channelData = dataChannels[:, channels][:dataLimit:granularity].transpose()
    y_channels = channelData.astype(float)
    inds = np.arange(channelData.shape[1])
    t = np.array([datetime.strptime(time[11:], '%H:%M:%S.%f') for time in timeChannels])
    return y_channels, inds, t

def getLabel(path):
    dataRaw = []
    first = True
    basetime = None
    with open(path, 'r') as label_file:
        for line in label_file:
            if(first):
                dt_obj = datetime.strptime(line[11:].strip(),'%H:%M:%S.%f')
                basetime = dt_obj
                first = False
            else:
                dr = np.char.strip(np.array(line[1:-1].split(", ")))
                for i in range(len(dr)):
                    if dr[i] == '1':
                        dataRaw.append(basetime + timedelta(seconds=i))
    return dataRaw

def groupbyInterval(data, labels, interval, actionType):
    #data tuple (x,y,z). labels: datetimes. interval(ms): int
    y_channels, inds, t = data
    interval_ms = timedelta(milliseconds=interval)

    split_inds = []
    cutoff_times = [t[0] + interval_ms]
    for ind in range(t.shape[0]):
        time = t[ind]
        if time >= cutoff_times[-1]:
            split_inds.append(ind)
            cutoff_times.append(cutoff_times[-1] + interval_ms)

    ind_groups = np.split(inds, split_inds)
    y_channels_groups = np.split(y_channels, split_inds, axis=1)
    t_groups = np.split(t, split_inds)

    #find min group size
    min_group_size = ind_groups[0].shape[0]
    for i in range(len(split_inds) - 1):
        if ind_groups[i].shape[0] < min_group_size:
            min_group_size = ind_groups[i].shape[0]

    #rectangularize jagged arrays
    for i in range(len(split_inds)):
        ind_groups[i] = ind_groups[i][:min_group_size]
        y_channels_groups[i] = y_channels_groups[i][:, :min_group_size]

print("sharika")
print("sharika")
print("sharika")
print("sharika")
print("sharika")