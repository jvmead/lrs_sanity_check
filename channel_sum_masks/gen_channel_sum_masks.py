## STANDARD IMPORTS
import numpy as np
import numpy.ma as ma
import pandas as pd
import yaml

is_data = False
is_TPC = False

# generating masks per tpc so channels can be isolated quickly
def generate_tpc_masks(merged_dict, data_shape):
    # Assuming spes_evt.shape is (N_events, 8, 64, N_samples)
    _, n_adcs, n_channels, _ = data_shape
    tpc_masks = []
    for tpc_key, detectors in merged_dict.items():
        # Initialize a mask for this TPC
        tpc_mask = np.zeros((n_adcs, n_channels), dtype=bool)
        for det_key, info in detectors.items():
            adc = info['ADC']
            channels = info['Channels']
            for ch in channels:
                tpc_mask[adc, ch] = True
        tpc_masks.append(tpc_mask)
    return tpc_masks

# generating masks per det so channels can be isolated quickly
def generate_det_masks(merged_dict, data_shape):
    # Assuming adc_values_evt.shape is (N_events, 8, 64, N_samples)
    _, n_adcs, n_channels, _ = data_shape
    detector_masks = []
    for tpc_key, detectors in merged_dict.items():
        for det_key, info in detectors.items():
            # Initialize a mask for this detector
            detector_mask = np.zeros((n_adcs, n_channels), dtype=bool)
            adc = info['ADC']
            channels = info['Channels']
            for ch in channels:
                detector_mask[adc, ch] = True
            detector_masks.append(detector_mask)
    return detector_masks

def geom_to_masks(geom_filename, data_shape, summed):
    #Load the channel mapping files and organize it to be per TPC
    with open(geom_filename, 'r') as file:
        detector_desc = yaml.safe_load(file)

    #Both these dictionaries contain the same keys namely the TPC key (0-7) and the det key (0-15), which we can loop over
    det_adc = detector_desc['det_adc']
    det_chan = detector_desc['det_chan']
    merged_dict = {}

    #Loop over TPC and the detector IDs
    for tpc, detectors in det_adc.items():
        tpc_key = f"TPC {tpc}"  #Add "TPC" to the key
        merged_dict[tpc_key] = {}
        #Loop over detectors in each TPC
        for detector, adc in detectors.items():
            detector_key = f"det {detector}"  #Add "det" to the detector key
            #Find corresponding channels for the same detector
            channels = det_chan[tpc][detector]
            #Store ADC and channels in the merged dictionary
            merged_dict[tpc_key][detector_key] = {'ADC': adc, 'Channels': channels}

    # get tpc masks
    if summed == 'DET':
        masks = generate_det_masks(merged_dict, data_shape)
    elif summed == 'TPC':
        masks = generate_tpc_masks(merged_dict, data_shape)
    return masks


# load file
data_shape = (0, 8, 64, 1000)

data = ''
sum = ''

if is_data:
    data = 'data'
    geom_filename = 'light_module_desc-5.0.0.yaml'
    if is_TPC:
        sum = 'TPC'
        masks = geom_to_masks(geom_filename, data_shape, 'TPC')
    else:
        sum = 'DET'
        masks = geom_to_masks(geom_filename, data_shape, 'DET')
else:
    data = 'MC'
    geom_filename = 'light_module_desc-4.0.0.yaml'
    if is_TPC:
        sum = 'TPC'
        masks = geom_to_masks(geom_filename, data_shape, 'TPC')
    else:
        sum = 'DET'
        masks = geom_to_masks(geom_filename, data_shape, 'DET')

# save masks to file
np.save(f'channel_sum_masks/{data}_{sum}', masks)