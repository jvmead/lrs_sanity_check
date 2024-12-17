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

    # get dirname
    if max_evts == None:
        dirname = 'data_processed_'+filename+'_'+summed+'_evts_all'
    else:
        dirname = 'data_processed_'+filename+'_'+summed+'_evts_' + str(max_evts)

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



def get_baseline_and_noise_threshold(wvfms, n_mad_factor=1.4826, max_iters=10, tol=1e-3):
    """
    Calculate the baseline and noise threshold iteratively using an iterative clipping approach.

    Parameters:
    - wvfms: np.ndarray
        Waveforms array (shape: [n_waveforms, n_samples]).
    - n_mad_factor: float
        Threshold factor for outlier detection (default: 1.5).
    - max_iters: int
        Maximum number of iterations (default: 10).
    - tol: float
        Convergence tolerance for baseline change (default: 1e-3).

    Returns:
    - baseline: np.ndarray
        Baseline estimate per waveform.
    - noise: np.ndarray
        Noise (standard deviation) per waveform.
    """
    # Initialize median and MAD for the first iteration
    median = np.median(wvfms, axis=-1)
    mad = np.median(np.abs(wvfms - median[..., np.newaxis]), axis=-1)

    # Initialize baseline and noise
    baseline = median
    noise = 1.4826 * mad  # Convert MAD to standard deviation

    for i in range(max_iters):
        # Mask out samples outside the noise range
        noise_mask = np.abs(wvfms - baseline[..., np.newaxis]) < n_mad_factor * noise[..., np.newaxis]

        # Update noise samples with mask
        noise_samples = np.where(noise_mask, wvfms, np.nan)

        # Calculate new baseline and noise
        new_baseline = np.nanmean(noise_samples, axis=-1)
        new_noise = np.nanstd(noise_samples, axis=-1)

        # Check for convergence
        if np.all(np.isclose(new_baseline, baseline, atol=tol)):
            print(f"Converged after {i + 1} iterations.")
            break

        # Update baseline and noise for the next iteration
        baseline = new_baseline
        noise = new_noise

    else:
        print("Reached maximum iterations without full convergence.")

    print("Baseline and noise calculated, shapes:", baseline.shape, noise.shape)
    return baseline, noise



'''
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

    # get masks per TPC
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

    # get masks per detector
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

    # get masks per trap type in each TPC
    elif summed == 'TrapType':
        n_masks = len(np.unique(geom['TPC'])) * 2 # len(np.unique(geom[summed]))
        masks = np.zeros((n_masks, n_adcs, n_channels))
        # for each TRP, make an 8x64 mask
        for i in range(len(np.unique(geom['TPC']))):
            tpc_mask = geom['TPC'] == i
            acl_mask = geom['TrapType'] == 0
            lcm_mask = geom['TrapType'] == 1
            # set acl masks
            acl_adcs = geom['ADC'][tpc_mask & acl_mask]
            acl_channels = geom['Channel'][tpc_mask & acl_mask]
            masks[i, acl_adcs, acl_channels] = 1
            # set lcm masks
            lcm_adcs = geom['ADC'][tpc_mask & lcm_mask]
            lcm_channels = geom['Channel'][tpc_mask & lcm_mask]
            masks[2*(i+1)-1, lcm_adcs, lcm_channels] = 1

    # good channels only
    channel_status_mask = channel_status == 0
    masks[:,~channel_status_mask] = 0

    return masks
'''



def get_data(filename, calib_filename, geom_filename, channel_status_filename, maskfile, max_evts):

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
            '''
            # get channel status
            channel_status = pd.read_csv(channel_status_filename, header=None)
            # get geometry
            geom = pd.read_csv(geom_filename, channel_status_filename, summed, data_shape)
            # get masks
            masks = get_masks(geom_filename, channel_status_filename, summed, data_shape)
            '''
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
        baselines, noise_thresholds = get_baseline_and_noise_threshold(wvfms_summed, n_mad_factor=1.5)
        print("Baselines and noise thresholds calculated, shapes: ", baselines.shape, noise_thresholds.shape)
        wvfms_blsub = wvfms_summed - baselines[..., np.newaxis]
        print("Baseline subtracted wvfms loaded, shape: ", wvfms_blsub.shape)

    return wvfms_blsub, noise_thresholds



'''
# hit finder function
def hitfinder(time_bins, wvfm, noise,
            n_noise_factor = 5.0,
            n_sqrt_factor = 1.0):

  # height = flat threshold over noise (n*sigma)
  height = n_noise_factor * noise
  sqrt_height = np.sqrt(height)

  # hits
  hits = []

  # Find indices where waveform exceeds threshold
  bins_over_threshold = np.where(wvfm > height)[0]
  bins_below_threshold = np.where(wvfm < height - n_sqrt_factor*sqrt_height)[0]
  while len(bins_over_threshold) > 0:
    # Find the start of a hit (first index above threshold)
    t_i = bins_over_threshold[0]

    # Find next index below lower threshold that comes after t_i
    bins_below_after_ti = bins_below_threshold[bins_below_threshold > t_i]
    if len(bins_below_after_ti) == 0:
      break  # Exit if no bins are below threshold after t_i
    t_f = bins_below_after_ti[0]

    # if the hit starts at t=0 then skip it
    if t_i == 0:
      bins_over_threshold = bins_over_threshold[bins_over_threshold > t_f]
      continue

    # Calculate start time and time over threshold (ToT)
    t0 = time_bins[t_i]
    tf = time_bins[t_f]
    tot = time_bins[t_f] - t0

    # Calculate height of the highest bin in the ToT window
    hit_window = wvfm[t_i : t_f+1]
    max_height = np.max(hit_window)
    t_max = time_bins[t_i + np.argmax(hit_window)]
    #t_max_idx = t_i + np.argmax(hit_window)

    # Calculate integral of the waveform in the ToT window
    integral = np.sum(hit_window)

    # append hit information with t0, tf, tot, t_max, height, integral
    hits.append((t0, tf, tot, t_max, max_height, integral))

    # Update bins_over_threshold to exclude processed region
    bins_over_threshold = bins_over_threshold[bins_over_threshold > t_f]

    return hits
    '''



# interaction finder function
def interaction_finder(wvfm, noise,
                       n_noise_factor = 40.0,
                       #n_sqrt_factor = 1.0,
                       n_bins_rolled = 1,
                       n_sqrt_rt_factor = 3.0,
                       pe_weight = 1.0):

  # save hitfinder settings to config
  hit_config = {'n_noise_factor': n_noise_factor,
                #'n_sqrt_factor': n_sqrt_factor,
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



def main(path, is_data, summed, max_evts, run_hitfinder, overwrite_preprocessing, overwrite_hitfinder):

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
    main(args.path, args.is_data, args.summed, args.max_evts, args.run_hitfinder, args.opp, args.ohf)

    # end timer
    end_time = time.time()
    seconds = end_time - start_time
    cpu_seconds = seconds * number_of_cpus
    cpu_flops = cpu_seconds * clock_speed * flops_per_cycle
    print(f"Seconds: {seconds:.4f}")
    print(f"CPU seconds: {cpu_seconds:.4f}")
    print(f"CPU FLOPS: {cpu_flops:.4e}")