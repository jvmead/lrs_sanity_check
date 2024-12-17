import numpy as np
import pandas as pd
import json

# set channels as good = 0
channel_status = np.zeros((8, 64))

# open json file
with open('channel_status/channel_status.json') as json_file:
    data = json.load(json_file)
    # print keys
    print(data.keys())
    for i in range(8):
        for j in range(64):
            if data['inactive_channels'][i][j] == 1:
                channel_status[i][j] = -1
            if data['dead_channels'][i][j] == 1:
                channel_status[i][j] = 1
            if data['bad_baseline'][i][j] == 1:
                channel_status[i][j] = 2

## save channel status as a csv file
df = pd.DataFrame(channel_status)
df.to_csv('channel_status/channel_status.csv', index=False, header=False)