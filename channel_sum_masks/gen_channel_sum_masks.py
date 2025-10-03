## STANDARD IMPORTS
import numpy as np
import numpy.ma as ma
import pandas as pd
import yaml


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


# generating masks per trap type per TPC so channels can be isolated quickly
def generate_trap_masks(merged_dict, data_shape):
    # Assuming adc_values_evt.shape is (N_events, 8, 64, N_samples)
    _, n_adcs, n_channels, _ = data_shape
    # Create a dict to hold masks: {(tpc_key, trap_type): mask}
    trap_masks_dict = {}
    for tpc_key, detectors in merged_dict.items():
        # Initialize empty masks for both trap types for this TPC
        trap_masks = {0: np.zeros((n_adcs, n_channels), dtype=bool),
                      1: np.zeros((n_adcs, n_channels), dtype=bool)}
        for det_key, info in detectors.items():
            adc = info['ADC']
            channels = info['Channels']
            if len(channels) == 6:
                trap_type = 0
            elif len(channels) == 2:
                trap_type = 1
            else:
                continue
            for ch in channels:
                trap_masks[trap_type][adc, ch] = True
        # Append masks for each trap type for this TPC
        for trap_type in [0, 1]:
            trap_masks_dict[(tpc_key, trap_type)] = trap_masks[trap_type]
    # Return as a list in a consistent order
    return [trap_masks_dict[key] for key in sorted(trap_masks_dict.keys())]


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



def generate_epcb_masks(merged_dict, data_shape):
    # Assuming adc_values_evt.shape is (N_events, 8, 64, N_samples)
    _, n_adcs, n_channels, _ = data_shape
    epcb_masks = []

    for tpc_key, detectors in merged_dict.items():
        # Collect all DETs with 2 channels and 6 channels for this TPC
        two_ch_dets = []
        six_ch_dets = []
        for det_key, info in detectors.items():
            channels = sorted(info['Channels'])
            if len(channels) == 2:
                two_ch_dets.append((info['ADC'], channels))
            elif len(channels) == 6:
                six_ch_dets.append((info['ADC'], channels))

        # Build masks for 6-channel DETs
        six_masks = []
        for adc, chs in six_ch_dets:
            m = np.zeros((n_adcs, n_channels), dtype=bool)
            for ch in chs:
                m[adc, ch] = True
            six_masks.append(m)

        # Build masks for groups of three 2-channel DETs
        two_masks = []
        two_ch_dets.sort()  # sort by ADC then channels
        used = set()
        for i in range(len(two_ch_dets)):
            if i in used:
                continue
            adc_i, chs_i = two_ch_dets[i]
            group = [(adc_i, chs_i)]
            used.add(i)
            for j in range(i+1, len(two_ch_dets)):
                if j in used:
                    continue
                adc_j, chs_j = two_ch_dets[j]
                # Same ADC and consecutive channels
                if adc_j == adc_i and chs_i[-1] + 1 == chs_j[0]:
                    group.append((adc_j, chs_j))
                    used.add(j)
                    chs_i = chs_j  # move window
                    if len(group) == 3:
                        break
            # If found a group of 3 DETs (6 consecutive channels)
            if len(group) == 3:
                m = np.zeros((n_adcs, n_channels), dtype=bool)
                for adc, chs in group:
                    for ch in chs:
                        m[adc, ch] = True
                two_masks.append(m)

        # Interleave: 6-channel mask, then 3x2-channel mask, repeat
        i = j = 0
        take_six = True  # start with 6-channel to match your original order
        while i < len(six_masks) or j < len(two_masks):
            if take_six and i < len(six_masks):
                epcb_masks.append(six_masks[i]); i += 1
            elif (not take_six) and j < len(two_masks):
                epcb_masks.append(two_masks[j]); j += 1
            else:
                # If the preferred list is exhausted, take from the other
                if i < len(six_masks):
                    epcb_masks.append(six_masks[i]); i += 1
                elif j < len(two_masks):
                    epcb_masks.append(two_masks[j]); j += 1
            take_six = not take_six

    return epcb_masks



def geom_to_masks(geom_filename, data_shape, summed, data):
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
    elif summed == 'EPCB':
        masks = generate_epcb_masks(merged_dict, data_shape)
    elif summed == 'TrapType':
        masks = generate_trap_masks(merged_dict, data_shape)
    elif summed == 'TPC':
        masks = generate_tpc_masks(merged_dict, data_shape)

    # save masks to file
    np.save(f'{data}_{summed}', masks)
    return masks


# load file
data_shape = (0, 8, 64, 1000)

# data masks
data = 'data'
geom_filename = '/global/homes/j/jvmead/dune/ndlar_flow/data/proto_nd_flow/light_module_desc-5.0.0.yaml'
# TPC masks
masks = geom_to_masks(geom_filename, data_shape, 'TPC', 'data')
# TrapType masks
masks = geom_to_masks(geom_filename, data_shape, 'TrapType', 'data')
# EPCB masks
masks = geom_to_masks(geom_filename, data_shape, 'EPCB', 'data')
# DETector masks
masks = geom_to_masks(geom_filename, data_shape, 'DET', 'data')

# MC masks
geom_filename = '/global/homes/j/jvmead/dune/ndlar_flow/data/proto_nd_flow/light_module_desc-4.0.0.yaml'
# TPC masks
masks = geom_to_masks(geom_filename, data_shape, 'TPC', 'MC')
# TrapType masks
masks = geom_to_masks(geom_filename, data_shape, 'TrapType', 'MC')
# EPCB masks
masks = geom_to_masks(geom_filename, data_shape, 'EPCB', 'MC')
# DETector masks
masks = geom_to_masks(geom_filename, data_shape, 'DET', 'MC')