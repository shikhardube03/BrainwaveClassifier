import numpy as np
from datetime import datetime, timedelta


def get_title(recording_file: str) -> str:
    return recording_file.split("-")[-1].split(".")[0].translate({ord(k): None for k in "0123456789"})


def get_data(path: str, granularity: int = 1, channel: int = 0, data_limit: int = None) -> tuple:
    data_raw = np.genfromtxt(path, delimiter=",", skip_header=6, dtype=str)
    data_channels = data_raw[:, 1:5]
    time_channels = data_raw[:, 14]
    data_limit = data_limit or len(data_channels)

    channel_data = data_channels[:, channel][:data_limit:granularity]
    y = channel_data.astype(float)
    x = np.arange(len(channel_data))
    t = [datetime.strptime(time.split()[1], "%H:%M:%S.%f") for time in time_channels]

    return x, y, t


def get_label(path: str) -> list:
    with open(path, "r") as label_file:
        return [datetime.strptime(line.strip(), "%H:%M:%S.%f") for line in label_file]


def group_by_interval(data: tuple, labels: list, interval: int, action_type: str) -> tuple:
    x, y, t = data
    interval_ms = timedelta(milliseconds=interval)
    cutoff_times = [t[0] + interval_ms]
    split_inds = []

    for ind in range(len(t)):
        time = t[ind]
        if time >= cutoff_times[-1]:
            split_inds.append(ind)
            cutoff_times.append(cutoff_times[-1] + interval_ms)

    x_groups = np.array(np.split(x, split_inds))
    y_groups = np.array(np.split(y, split_inds))
    t_groups = np.array(np.split(t, split_inds))
    l_groups = np.zeros(len(x_groups), dtype=bool)

    if action_type:
        l_groups = np.array(["NONE" if time < labels[0] else action_type for time in t_groups[:, 0]])
        l_groups[l_groups != action_type] = "NONE"
        labels = labels[1:]

    for ind in range(len(cutoff_times)):
        if labels and labels[0] < cutoff_times[ind]:
            l_groups[ind] = action_type
            labels.pop(0)

    return (x_groups, y_groups, t_groups), l_groups


def return_millisecond_timestamps(labels: list) -> list:
    return [datetime.strptime(timestamp, "%H:%M:%S.%f").timestamp() * 1000 for timestamp in labels]


def standardise_observations(grouped_data: tuple, group_contains_label: list) -> tuple:
    REQ_NUM_PTS = 190
    x_groups, y_groups, t_groups = grouped_data
    l_groups = group_contains_label

    # remove all instances where not enough sample points
    mask = [len(y) >= REQ_NUM_PTS for y in y_groups] 
    x_groups = x_groups[mask]
    y_groups = y_groups[mask]
    t_groups = t_groups[mask]
    l_groups = l_groups[mask]

    # limit to required observations
    for i in range(len(x_groups)):
        x_groups[i] = x_groups[i][:REQ_NUM_PTS]
        y_groups[i] = y_groups[i][:REQ_NUM_PTS]
        t_groups[i] = t_groups[i][:REQ_NUM_PTS]
        
    return (x_groups, y_groups, t_groups), l_groups

def getObservationSet(dataPath, labelPath, interval, channels, actionType):
	observationSet = {}
	for channel in channels:
		data = get_data(dataPath, None, channel, None)
		action_times = get_label(labelPath)
		observations = group_by_interval(data, action_times, interval, actionType)
		observations = standardise_observations(observations[0], observations[1])
		observationSet[channel] = observations
	
	return observationSet