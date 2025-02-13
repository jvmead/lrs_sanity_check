import numpy as np
import pandas as pd


# get masks to sum channels in TPC or detector quickly
def get_masks(geom_filename, channel_status_filename, summed, data_shape=[1, 8, 64, 1000]):

    # load geometry
    geom = pd.read_csv(geom_filename)

    # get channel status
    channel_status = pd.read_csv(channel_status_filename, header=None)

    # data shape
    n_evts = data_shape[0]
    n_adcs = data_shape[1]
    n_channels = data_shape[2]
    n_samples = data_shape[3]

    # get masks
    if summed == 'TPC':
        n_masks = len(np.unique(geom[summed]))
        masks = np.zeros((n_masks, n_adcs, n_channels))
        # for each TPC, make an 8x64 mask
        for i in range(n_masks):
            tpc_mask = geom['TPC'] == i
            # get adc and channel numbers for this TPC
            adcs = geom['ADC'][tpc_mask]
            channels = geom['Channel'][tpc_mask]
            # set mask
            masks[i, adcs, channels] = 1

    elif summed == 'Detector':
        n_masks = len(np.unique(geom[summed]))
        masks = np.zeros((n_masks, n_adcs, n_channels))
        # for each detector, make an 8x64 mask
        for i in range(n_masks):
            det_mask = geom['Detector'] == i
            # get adc and channel numbers for this detector
            adcs = geom['ADC'][det_mask]
            channels = geom['Channel'][det_mask]
            # set mask
            masks[i, adcs, channels] = 1

    elif summed == 'TrapType':
        n_masks = len(np.unique(geom['TPC'])) * 2
        masks = np.zeros((n_masks, n_adcs, n_channels))
        # for each TRP, make an 8x64 mask
        for i in range(len(np.unique(geom['TPC']))):
            tpc_mask = geom['TPC'] == i
            acl_mask = geom['TrapType'] == 0
            lcm_mask = geom['TrapType'] == 1
            # set acl masks
            acl_adcs = geom['ADC'][tpc_mask & acl_mask]
            acl_channels = geom['Channel'][tpc_mask & acl_mask]
            masks[2*i, acl_adcs, acl_channels] = 1
            # set lcm masks
            lcm_adcs = geom['ADC'][tpc_mask & lcm_mask]
            lcm_channels = geom['Channel'][tpc_mask & lcm_mask]
            masks[2*i+1, lcm_adcs, lcm_channels] = 1

    # good channels only
    channel_status_mask = channel_status == 0
    masks[:,~channel_status_mask] = 0

    return masks

# Example usage
is_data = False
if is_data:
    geom_filename = '../geom_files/light_module_desc-5.0.0.csv'
else:
    geom_filename = '../geom_files/light_module_desc-4.0.0.csv'
channel_status_filename = '../channel_status/channel_status.csv'
summed = 'TPC'
data_shape = [1, 8, 64, 1000]
masks = get_masks(geom_filename, channel_status_filename, summed, data_shape)

# save masks to file
data = 'data' if is_data else 'MC'
# save as npz file
np.savez('{}_masks_{}.npz'.format(summed, data), masks=masks)