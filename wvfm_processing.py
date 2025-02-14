### module for preprocssing light readout waveforms

## STANDARD IMPORTS
import os
import time
import h5py
import numpy as np
import pandas as pd
import argparse
import json
import scipy
from scipy.ndimage import uniform_filter1d



import matplotlib.pyplot as plt
def plot_summed_waveform_with_interactions(time_bins, wvfm, height, hits, i_mask=None, i_evt=None, xlim = (0, 16), logy=False, print_int_times=False):
  # plot the waveform
  fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

  # convert i_mask into TPC and TrapType
  if i_mask is not None:
    tpc = i_mask // 2
    trap_type = i_mask % 2
    if trap_type == 0:
      trap_type = 'ACL'
    else:
      trap_type = 'LCM'
    ax1.set_title(f'TPC {tpc} {trap_type}')

  if i_mask is not None and i_evt is not None:
    ax1.set_title(f'TPC {i_mask} Event {i_evt}')

  ylabel = 'SPEs'
  sqrt_height = np.sqrt(height)
  if logy:
    wvfm+=1
    height+=1
    ylabel+=' + 1'
    ax1.set_yscale('log')
    #ax1.set_ylim(0.99, 1.1 * np.max(wvfm))

  # Linear y-axis plot
  ax1.plot(time_bins * 1e6, wvfm, color='black')
  ax1.set_ylabel(ylabel)

  # plot noise floor
  ax1.axhline(height, color='r', linestyle='--')

  # peaks for interactions
  ax1.plot(time_bins[hits] * 1e6, wvfm[hits], 'x', color='red', label='Interactions')
  if print_int_times:
    for hit in hits:
      ax1.text(time_bins[hit] * 1e6, wvfm[hit], f'{time_bins[hit]*1e6:.2f}', color='red')

  # formatting
  #ax1.set_xlim(xlim)
  #ax1.set_ylim(2 * np.min(wvfm), 1.1 * np.max(wvfm))
  ax1.legend()

  plt.show()


# function for getting the data
def bookkeeping(filename, is_data, summed=None, max_evts=None):

    data = 'mc'
    if is_data:
        data = 'data'

    # get dirname
    if max_evts == None:
        dirname = data+'_processed_'+filename+'_'+summed+'_evts_all'
    else:
        dirname = data+'_processed_'+filename+'_'+summed+'_evts_' + str(max_evts)

    # channel status
    channel_status_filename = 'channel_status/channel_status.csv'

    # calibration and geometry
    if is_data:
        geom_filename = 'geom_files/light_module_desc-5.0.0.csv'
        calib_filename = 'calibration/data_calib.csv'
    else: # is MC
        geom_filename = 'geom_files/light_module_desc-4.0.0.csv'
        calib_filename = 'calibration/mc_calib.csv'

    # summing masks
    if summed == None:
        maskfile = None
    else:
        maskfile = 'channel_sum_masks/{}_masks'.format(summed)
        if is_data:
            maskfile += '_data.npz'
        else:
            maskfile += '_mc.npz'

    return dirname, channel_status_filename, geom_filename, calib_filename, maskfile


# function for getting baseline and noise threshold per waveform per channel
def get_baseline_and_noise_threshold(wvfms, n_mad_factor=5.0):
    # Initialize median and MAD
    median = np.median(wvfms, axis=-1)
    mad = np.median(np.abs(wvfms - median[..., np.newaxis]), axis=-1)
    # identify outliers in the waveform
    mad_factor = n_mad_factor * mad
    noise_mask = np.abs(wvfms - median[..., np.newaxis]) < mad_factor[..., np.newaxis]
    print("Noise mask calculated, shape: ", noise_mask.shape)
    # set non mask values to nan
    noise_samples = np.where(noise_mask, wvfms, np.nan)
    print("Noise samples calculated, shape: ", noise_samples.shape)
    # calculate noise as stddev of noise_samples
    noise = np.nanstd(noise_samples, axis=-1)
    print("Noise calculated, shape: ", noise.shape)
    # calculate baseline as mean of noise_samples
    baseline = np.nanmean(noise_samples, axis=-1)
    print("Baseline calculated, shape: ", baseline.shape)

    return baseline, noise


def get_data(filename, calib_filename, geom_filename, channel_status_filename, maskfile, max_evts, n_mad_factor=5.0):

    # load file
    with h5py.File(filename, 'r') as f:

        # load data
        wvfms = f['light/wvfm/data']['samples']
        data_shape = wvfms.shape
        n_evts = data_shape[0]
        n_adcs = data_shape[1]
        n_channels = data_shape[2]
        n_samples = data_shape[3]
        print("Raw wvfms loaded, shape: ", data_shape)

        # get calibration csv file
        calib_csv = pd.read_csv(calib_filename, header=None)
        calib_npy = calib_csv.to_numpy()
        wvfms_calib = wvfms * calib_npy[np.newaxis, :, :, np.newaxis]
        print("Calibrated wvfms loaded, shape: ", wvfms_calib.shape)

        # summing channels by TPC, detector, or trap type
        if maskfile != None:
            masks_file = np.load(maskfile)
            masks = np.array(masks_file['masks'])
            print("Summed channels masks loaded, shape: ", masks.shape)

            # sum channels
            wvfms_summed = np.zeros((n_evts, masks.shape[0], n_samples))
            #noise_summed = np.zeros((n_evts, masks.shape[0]))
            for i in range(masks.shape[0]):
                wvfms_summed[:, i, :] = np.sum(wvfms_calib[:, masks[i]==1, :], axis=(1))
                #noise_summed[:, i] = np.sqrt(np.sum(noise_thresholds[:, masks[i] == 1]**2, axis=1))
        else:
            wvfms_summed = wvfms_calib
            #noise_summed = noise_thresholds
        print("Channels summed, shape: ", wvfms_summed.shape)

        # get baseline and noise threshold per waveform per channel
        baselines, noise_thresholds = get_baseline_and_noise_threshold(wvfms_summed, n_mad_factor=n_mad_factor)
        print("Baselines and noise thresholds calculated, shapes: ", baselines.shape, noise_thresholds.shape)
        wvfms_blsub = wvfms_summed - baselines[..., np.newaxis]
        print("Baseline subtracted wvfms loaded, shape: ", wvfms_blsub.shape)

    return wvfms_blsub, noise_thresholds


def get_truth(filename, file_idx, in_tpc=False):
    # check if file exists
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        return None

    # load file
    with h5py.File(filename, 'r') as f:

        # get the geometry info
        mod_bounds_mm = np.array(f['geometry_info'].attrs['module_RO_bounds'])
        tpc_bounds_mm = []

        for i,mod in enumerate(mod_bounds_mm):
            x_min = mod[0][0]
            x_max = mod[1][0]
            y_min = mod[0][1]
            y_max = mod[1][1]
            z_min = mod[0][2]
            z_max = mod[1][2]

            # split modules in half, using max_drift_distance from outer x-facing edge
            max_drift_distance = f['geometry_info'].attrs['max_drift_distance']
            x_min_adj = x_max - max_drift_distance
            # ordering TPC number in descending x
            tpc_bounds_mm.append(((x_min_adj, y_min, z_min), (x_max, y_max, z_max)))
            x_max_adj = x_min + max_drift_distance
            tpc_bounds_mm.append(((x_min, y_min, z_min), (x_max_adj, y_max, z_max)))

        tpc_bounds_mm = np.array(tpc_bounds_mm)

        # get event in data
        light_wvfms = f['light/wvfm/data']['samples']
        #i_evt_lrs = np.linspace(0, light_wvfms.shape[0]-1, light_wvfms.shape[0], dtype=int)

        # for each unique vertex_id, get the segment with the min time
        first_seg_idxs = []
        first_seg_tpcs = []
        first_seg_file_ids = []
        first_seg_evt_ids = []
        first_seg_x = []
        first_seg_y = []
        first_seg_z = []

        # loop over events
        for i_evt_lrs in range(light_wvfms.shape[0]):

            # get the light waveforms
            light_data = light_wvfms[i_evt_lrs]

            # get segment to vertex matching
            mc_seg_data = f["mc_truth/segments/data"]
            spill_id = np.unique(mc_seg_data["event_id"])[i_evt_lrs]
            ev_seg_ids = np.where(mc_seg_data["event_id"]==spill_id)[0]
            ev_truth_light = f["mc_truth/light/data"][ev_seg_ids]

            # get segment vertex_id
            int_seg_ids = mc_seg_data["vertex_id"][ev_seg_ids]

            # get segment times
            segment_times = mc_seg_data["t0_start"][ev_seg_ids]
            segment_idxs = segment_times%1.2e6*(1000.0/16.0)+100

            # segment coordinates
            seg_x = mc_seg_data["x_start"][ev_seg_ids]
            seg_y = mc_seg_data["y_start"][ev_seg_ids]
            seg_z = mc_seg_data["z_start"][ev_seg_ids]

            for i in np.unique(int_seg_ids):

                idx = np.where(int_seg_ids==i)[0]
                first_seg_time = np.argmin(segment_times[idx])
                first_seg_idx = first_seg_time%1.2e6*(1000.0/16.0)+100
                first_seg_idxs.append(segment_idxs[idx])

                # for each event, find the TPC number
                first_seg_tpc = np.full(len(seg_x), -1, dtype=int)
                for j in range(len(tpc_bounds_mm)):
                    mask = (seg_x > tpc_bounds_mm[j][0][0]) & (seg_x < tpc_bounds_mm[j][1][0]) & \
                        (seg_y > tpc_bounds_mm[j][0][1]) & (seg_y < tpc_bounds_mm[j][1][1]) & \
                        (seg_z > tpc_bounds_mm[j][0][2]) & (seg_z < tpc_bounds_mm[j][1][2])
                    first_seg_tpc[mask] = j
                first_seg_tpcs.append(first_seg_tpc)

                # add file and event ids
                first_seg_file_ids.append(file_idx)
                first_seg_evt_ids.append(spill_id)

                # add vertex coordinates
                first_seg_x.append(seg_x[idx])
                first_seg_y.append(seg_y[idx])
                first_seg_z.append(seg_z[idx])

        # save event number, tpc number and start time
        event_tpc_start = np.column_stack((first_seg_file_ids, first_seg_evt_ids,
                                           first_seg_time, first_seg_idx,
                                           first_seg_tpcs, first_seg_x, first_seg_y, first_seg_z))

        # if first file, create the dataframe
        cols = ['file_idx', 'event_id',
                'start_time', 'start_time_idx',
                'tpc_num', 'v_x', 'v_y', 'v_z']

        if in_tpc:
            mask = (first_seg_tpcs >= 0)
            event_tpc_start = event_tpc_start[mask]

        return event_tpc_start


# interaction finder function
def interaction_finder(wvfm, noise,
                       n_noise_factor = 5.0,
                       n_bins_rolled = 10,
                       n_sqrt_rt_factor = 5.0,
                       pe_weight = 1.0):

  # save hitfinder settings to config
  hit_config = {'n_noise_factor': n_noise_factor,
                'n_bins_rolled': n_bins_rolled,
                'n_sqrt_rt_factor': n_sqrt_rt_factor,
                'pe_weight': pe_weight}

  # height = flat threshold over noise (n*sigma)
  height = n_noise_factor * noise[..., np.newaxis] * np.ones(wvfm.shape[-1])
  print("Height calculated, shapes: ", height.shape)

  # dynamic_threshold = rolling threshold of previous 5 bins + n*sqrt(rolling threshold)
  wvfm_rolled = np.roll(wvfm, n_bins_rolled)
  rolling_average = uniform_filter1d(wvfm_rolled, size=n_bins_rolled)
  sqrt_rolling_average = np.sqrt(np.abs(rolling_average) * pe_weight**2)
  sqrt_rolling_average[sqrt_rolling_average == 0] = 1
  dynamic_threshold = rolling_average + n_sqrt_rt_factor*sqrt_rolling_average
  print("Dynamic threshold, shapes: ", dynamic_threshold.shape)

  # find rising edges
  bins_over_dynamic_threshold = (wvfm > dynamic_threshold) & (wvfm > height)
  # remove consecutive bins, keep only the first
  bins_over_dynamic_threshold[..., 1:] = bins_over_dynamic_threshold[..., 1:] & ~bins_over_dynamic_threshold[..., :-1]

  return bins_over_dynamic_threshold, hit_config



def main(path, is_data, summed, max_evts, run_hitfinder, overwrite_preprocessing, overwrite_hitfinder, save_truth, is_cont):

    # get bookkeeping
    filename = path.split('/')[-1]
    name = filename.split('.hdf5')[0]
    dirname, channel_status_filename, geom_filename, calib_filename, maskfile = bookkeeping(name, is_data, summed, max_evts)

    if not os.path.exists(dirname) or overwrite_preprocessing:
        if overwrite_preprocessing:
            print("Directory exists: ", dirname, ", overwriting data...")
        else:
            print("Directory does not exist, processing...")
            os.makedirs(dirname)
            print("Directory created: ", dirname)

        # make config file json
        config = {'timestamp': str(pd.Timestamp.now()),
                  'filename': path,
                  'is_data': is_data,
                  'save_truth': save_truth,
                  'summed': summed,
                  'max_evts': max_evts,
                  'calib_filename': calib_filename,
                  'geom_filename': geom_filename,
                  'channel_status_filename': channel_status_filename,
                  'maskfile': maskfile}
        with open(dirname+'/config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Config file created: config.json")

        # get data
        spes_evt, noise_evt = get_data(path, calib_filename, geom_filename, channel_status_filename, maskfile, max_evts)
        print("Data processed, shapes: ", spes_evt.shape, noise_evt.shape)

        # save data
        np.savez(dirname+'/spes_evt.npz', spes_evt)
        np.savez(dirname+'/noise_evt.npz', noise_evt)

    else:
        print("Directory exists, loading data...")

        # load data
        spes_file = np.load(dirname+'/spes_evt.npz')
        spes_evt = spes_file['arr_0']
        noise_file = np.load(dirname+'/noise_evt.npz')
        noise_evt = noise_file['arr_0']

        #tpc_masks = np.load(dirname+'/tpc_masks.npz')
        #channel_status = np.load(dirname+'/channel_status.npz')
        #calib = np.load(dirname+'/calib.npz')

        print("Data loaded, shapes: ", spes_evt.shape, noise_evt.shape)

    # run hitfinder
    if run_hitfinder:
        # check if hitfinder has already been run
        if not os.path.exists(dirname+'/hits_evt.npz') or overwrite_hitfinder:
            if os.path.exists(dirname+'/hits_evt.npz'):
                print(dirname+'/hits.npz exists, overwriting...')
            else:
                print(dirname+'/hits.npz does not exists, processing...')
            print("Running hitfinder...")
            hits_evt, hits_config = interaction_finder(spes_evt, noise_evt)
            print("Hitfinder run, shape: ", hits_evt.shape)
            print("Total hits: ", np.sum(hits_evt != -1))
            # save config
            with open(dirname+'/hits_config.json', 'w') as f:
                json.dump(hits_config, f, indent=4)
            # save hits
            np.savez(dirname+'/hits_evt.npz', hits_evt)
        else:
            print(dirname+'/hits.npz exists, exiting.') #loading hits...')
            #hits_file = np.load(dirname+'/hits_evt.npz')
            #hits_evt = hits_file['arr_0']

    # get truth info
    if is_data and save_truth:
        print("No truth information for data...")
    if save_truth and not is_data:
        # WIP
        i = 0
        print("Getting truth info...")
        truth_information = get_truth(path, i, is_cont)
        # save as csv file
        cols = ['file_idx', 'event_id', #'vertex_id',
                'tpc_num', 'start_time', 'start_time_idx',
                'v_x', 'v_y', 'v_z']
        true_int_output = dirname+'/true_nu_int.csv'
        if i==0:
            print("Creating output: ", true_int_output)
            df = pd.DataFrame(truth_information, columns=cols)
            df.to_csv(true_int_output, index=False)
        # for subsequent files, append to the dataframe
        else:
            print("Appending to output: ", true_int_output)
            df = pd.DataFrame(truth_information, columns=cols)
            df.to_csv(true_int_output, index=False, mode='a', header=False)

    else:
        print("Hitfinder not run")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some waveforms.')
    parser.add_argument('path', type=str, help='The name of the file to process')
    parser.add_argument('--is_data', action='store_true', help='Flag to indicate if the file is data')
    parser.add_argument('--summed', type=str, default=None, help='Summing method for channels')
    parser.add_argument('--max_evts', type=int, default=None, help='Maximum number of events to process')
    parser.add_argument('--run_hitfinder', action='store_true', help='Flag to indicate if hitfinder should be run')
    parser.add_argument('--opp', action='store_true', help='Flag to indicate if preprocessing should be overwritten')
    parser.add_argument('--ohf', action='store_true', help='Flag to indicate if hit finder output should be overwritten')
    parser.add_argument('--get_truth', action='store_true', help='Flag to indicate if truth info should be extracted')
    parser.add_argument('--is_cont', action='store_true', help='Flag to indicate if the vertices are contained in active volume')
    args = parser.parse_args()

    ## timing info
    number_of_cpus = os.cpu_count()
    clock_speed = 1.0 / time.get_clock_info('thread_time').resolution
    flops_per_cycle = 8 # for modern CPUs?
    print("Number of CPUs: ", number_of_cpus)
    print("Clock Speed: ", clock_speed)
    print("FLOPS per Cycle: ", flops_per_cycle)

    # start timer
    start_time = time.time()

    # execute main function for preprocessing data
    main(args.path, args.is_data, args.summed, args.max_evts, args.run_hitfinder, args.opp, args.ohf, args.get_truth, args.is_cont)

    # end timer
    end_time = time.time()
    seconds = end_time - start_time
    cpu_seconds = seconds * number_of_cpus
    cpu_flops = cpu_seconds * clock_speed * flops_per_cycle
    print(f"Seconds: {seconds:.4f}")
    print(f"CPU seconds: {cpu_seconds:.4f}")
    print(f"CPU FLOPS: {cpu_flops:.4e}")