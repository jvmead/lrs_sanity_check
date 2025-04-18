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
from scipy.ndimage import label
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# function for getting the data
def bookkeeping(filenames, nfiles, is_data, summed=None, max_evts=None):

    data = 'mc'
    if is_data:
        data = 'data'

    # make filename string from common prefix + nfiles
    filename = ''
    if nfiles > 1:
        filename += filenames[0]+'_nfiles_'+str(len(filenames))
    else:
        filename = filenames

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
            maskfile += '_MC.npz'

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


def get_truth(filename, file_idx, in_tpc=False, n_photons_threshold=7500):
    # check if file exists
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        return None

    # load file
    with h5py.File(filename, 'r') as f:

        n_events = f['light/wvfm/data']['samples'].shape[0]

        # for each unique vertex_id, get the segment with the min time per TPC
        first_seg_evt_ids = []
        first_seg_vertex_ids = []
        first_seg_tpcs = []
        first_seg_idxs = []
        first_seg_times = []
        first_seg_tpc_pileup = []

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
            # ordering TPC number in descending x
            x_max_adj = x_min + max_drift_distance
            tpc_bounds_mm.append(((x_min, y_min, z_min), (x_max_adj, y_max, z_max)))
            x_min_adj = x_max - max_drift_distance
            tpc_bounds_mm.append(((x_min_adj, y_min, z_min), (x_max, y_max, z_max)))
        tpc_bounds_mm = np.array(tpc_bounds_mm)

        # load values to be masked
        unique_ids = np.unique(f["mc_truth/segments/data"]["event_id"])
        photons_threshold = (f["mc_truth/segments/data"]["n_photons"] >= n_photons_threshold)

        # load values to be saved
        all_event_ids = f["mc_truth/segments/data"]["event_id"][:]
        all_vertex_id =  f["mc_truth/segments/data"]["vertex_id"][:]
        all_t0_start = f["mc_truth/segments/data"]["t0_start"][:]

        # load segment coordinates
        seg_xs_tot = f["mc_truth/segments/data"]["x_start"][:]
        seg_ys_tot = f["mc_truth/segments/data"]["y_start"][:]
        seg_zs_tot = f["mc_truth/segments/data"]["z_start"][:]

        # which tpc segment is in using tpc_bounds_mm
        tpc_mask = (
            (seg_xs_tot[:, None] > tpc_bounds_mm[:, 0, 0]) & (seg_xs_tot[:, None] < tpc_bounds_mm[:, 1, 0]) &
            (seg_ys_tot[:, None] > tpc_bounds_mm[:, 0, 1]) & (seg_ys_tot[:, None] < tpc_bounds_mm[:, 1, 1]) &
            (seg_zs_tot[:, None] > tpc_bounds_mm[:, 0, 2]) & (seg_zs_tot[:, None] < tpc_bounds_mm[:, 1, 2])
        )
        seg_tpc_tot = np.argmax(tpc_mask, axis=1)
        seg_tpc_tot[~tpc_mask.any(axis=1)] = -1

        # loop over events
        for i_evt_lrs in range(n_events):

            # get segment to vertex matching & filter out segments with less than 7500 photons
            spill_id = unique_ids[i_evt_lrs]
            ev_seg_ids = np.where(all_event_ids==spill_id)[0]
            if len(ev_seg_ids) == 0:
                continue

            # get segment to vertex matching & filter out segments with less than 7500 photons
            ev_seg_ids = ev_seg_ids[photons_threshold[ev_seg_ids]]
            if len(ev_seg_ids) == 0:
                continue

            # get segment's vertex_id and time
            seg_vertex_ids = all_vertex_id[ev_seg_ids]
            unique_vertex_ids = np.unique(seg_vertex_ids)

            # loop over vertex ids, and find which tpc each segment is in
            for vertex_id in unique_vertex_ids:

                # get subset of ev_seg_ids for this vertex
                vertex_segs = np.where(all_vertex_id==vertex_id)[0]
                ev_seg_vertex = np.intersect1d(ev_seg_ids, vertex_segs)

                # get segment times and sample idx
                segment_times = all_t0_start[ev_seg_vertex]
                segment_idx = segment_times%1.2e6*(1000.0/16.0)+100

                # get segment coordinates
                seg_tpcs = seg_tpc_tot[ev_seg_vertex]

                # for each unique seg_tpc, find the argmin of the segment times
                unique_seg_tpcs = np.unique(seg_tpcs)

                for tpc in unique_seg_tpcs:
                    tpc_segs = np.where(seg_tpcs==tpc)[0]
                    if len(tpc_segs) == 0:
                        continue

                    # find the segment with the min time
                    min_idx = np.argmin(segment_times[tpc_segs])

                    # save the segment info
                    first_seg_evt_ids.append(i_evt_lrs)#spill_id)
                    first_seg_vertex_ids.append(vertex_id)
                    first_seg_tpcs.append(tpc)
                    first_seg_idxs.append(segment_idx[min_idx])
                    first_seg_times.append(segment_times[min_idx])


        print("Truth info extracted, shapes: ", len(first_seg_evt_ids),
                                                len(first_seg_vertex_ids),
                                                len(first_seg_tpcs),
                                                len(first_seg_idxs),
                                                len(first_seg_times))


        # save event number, tpc number and start time
        event_tpc_start = np.column_stack((first_seg_evt_ids,
                                           first_seg_vertex_ids,
                                           first_seg_times,
                                           first_seg_idxs,
                                           first_seg_tpcs))

        if in_tpc:
            mask = (first_seg_tpcs >= 0)
            event_tpc_start = event_tpc_start[mask]

        return event_tpc_start


# interaction finder function
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema

def interaction_finder(wvfm, noise,
                       n_noise_factor=5.0,
                       n_bins_rolled=5,
                       n_sqrt_rt_factor=5.0,
                       pe_weight=1.0,
                       use_rising_edge=False,
                       use_local_maxima=True):
    """
    Identifies interaction points (first bins over threshold and peaks) in waveforms.

    Parameters:
    - wvfm: np.ndarray of shape (..., n_bins)
    - noise: np.ndarray of shape (...)
    - n_noise_factor: flat threshold multiplier (height)
    - n_bins_rolled: number of bins for rolling average
    - n_sqrt_rt_factor: multiplier for dynamic threshold above rolling average
    - pe_weight: scaling factor for sqrt(rolling average)
    - use_local_maxima: use scipy's local maxima detection instead of derivative method

    Returns:
    - first_bins_over: boolean array of same shape as wvfm, with first crossings
    - hit_config: dict of parameters used
    - peak_bins: boolean array marking peaks
    """

    # Save config
    hit_config = {
        'n_noise_factor': n_noise_factor,
        'n_bins_rolled': n_bins_rolled,
        'n_sqrt_rt_factor': n_sqrt_rt_factor,
        'pe_weight': pe_weight,
        'use_local_maxima': use_local_maxima
    }

    # Height threshold (flat noise-based)
    height = n_noise_factor * noise[..., np.newaxis] * np.ones(wvfm.shape[-1])

    # Rolling average for dynamic threshold
    wvfm_rolled = np.concatenate([
        np.zeros_like(wvfm[..., :n_bins_rolled]),
        wvfm[..., :-n_bins_rolled]
    ], axis=-1)
    rolling_average = uniform_filter1d(wvfm_rolled, size=n_bins_rolled, axis=-1)
    sqrt_rolling_average = np.sqrt(np.abs(rolling_average)) * pe_weight
    sqrt_rolling_average[sqrt_rolling_average == 0] = 1  # Prevent zeros
    dynamic_threshold = rolling_average + n_sqrt_rt_factor * sqrt_rolling_average

    # Find bins over both thresholds
    bins_over_dynamic_threshold = (wvfm > dynamic_threshold) & (wvfm > height)

    # Find first bins over threshold (rising edge)
    first_bins_over = bins_over_dynamic_threshold.copy()
    first_bins_over[..., 1:] &= ~bins_over_dynamic_threshold[..., :-1]

    if use_rising_edge:
        return first_bins_over, hit_config

    # Peak finding
    elif use_local_maxima:
        '''
        # Use argrelextrema to find local peaks
        peak_bins = np.zeros_like(wvfm, dtype=bool)
        for idx in np.ndindex(wvfm.shape[:-1]):
            peaks = argrelextrema(wvfm[idx], np.greater)[0]
            peak_bins[idx + (peaks,)] = True
        peak_bins &= (wvfm > dynamic_threshold) & (wvfm > height)
        '''
        # check 5 bins after first_bins_over and add argmax
        peak_bins = np.zeros_like(wvfm, dtype=bool)
        first_bins_indices = np.where(first_bins_over)
        for idx in zip(*first_bins_indices):
            start_idx = idx[-1]
            #end_idx = min(start_idx + n_bins_rolled, wvfm.shape[-1])
            end_idx = min(start_idx + 5, wvfm.shape[-1])
            peak_bin = np.argmax(wvfm[idx[:-1] + (slice(start_idx, end_idx),)])
            peak_bins[idx[:-1] + (start_idx + peak_bin,)] = True
    else:
        # Derivative-based peak detection
        wvfm_d1 = np.gradient(wvfm, axis=-1)
        wvfm_d2 = np.gradient(wvfm_d1, axis=-1)

        peak_bins = (wvfm > dynamic_threshold) & (wvfm > height) & \
                    (wvfm_d1 < 0) & (wvfm_d2 < 0)

        # Keep only the first peak in consecutive runs
        peak_bins[..., 1:] &= ~peak_bins[..., :-1]

    return peak_bins, hit_config



def main(path, nfiles, is_data, summed, max_evts, run_hitfinder, overwrite_preprocessing, overwrite_hitfinder, save_truth, overwrite_truth, is_cont):

    # check if path has multiple files
    if nfiles > 1:
        print("Multiple files found, processing each file individually...")

        # get bookkeeping
        filenames = [p.split('/')[-1] for p in path]
        names = [fn.split('.hdf5')[0] for fn in filenames]
        dirname, channel_status_filename, geom_filename, calib_filename, maskfile = bookkeeping(names, nfiles, is_data, summed, max_evts)

    else:
        print("Single file found, processing...")

        # get bookkeeping
        filename = path[0].split('/')[-1]
        name = filename.split('.hdf5')[0]
        dirname, channel_status_filename, geom_filename, calib_filename, maskfile = bookkeeping(name, nfiles, is_data, summed, max_evts)


    if not os.path.exists(dirname) or overwrite_preprocessing:
        if overwrite_preprocessing:
            print("Directory exists: ", dirname, ", overwriting data...")
        else:
            print("Directory does not exist, processing...")
            os.makedirs(dirname)
            print("Directory created: ", dirname)

        # make config file json
        config = {'timestamp': str(pd.Timestamp.now()),
                  'paths': [p.split('/')[0] for p in path],
                  'nfiles': len(filenames) if nfiles > 1 else 1,
                  'filenames': filenames if nfiles > 1 else filename,
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

        # loop over files
        for i, p in enumerate(path):
            print("Processing file: ", p)

            # get data
            # n_mad_factor = 1.4826 # mad->stddev
            spes_evt, noise_evt = get_data(p, calib_filename, geom_filename, channel_status_filename, maskfile, max_evts)
            print("Data processed, shapes: ", spes_evt.shape, noise_evt.shape)

            # save data
            np.savez(dirname+'/spes_evt_'+str(i)+'.npz', spes_evt)
            np.savez(dirname+'/noise_evt_'+str(i)+'.npz', noise_evt)

    else:
        print("Directory exists...")

    # run hitfinder
    if run_hitfinder:

        # loop over files
        for i, p in enumerate(path):

            # load data
            print("Loading data for hitfinder...")
            spes_file = np.load(dirname+'/spes_evt_'+str(i)+'.npz')
            spes_evt = spes_file['arr_0']
            noise_file = np.load(dirname+'/noise_evt_'+str(i)+'.npz')
            noise_evt = noise_file['arr_0']

            print("Data loaded, shapes: ", spes_evt.shape, noise_evt.shape)

            # check if hitfinder has already been run
            if not os.path.exists(dirname+'/hits_evt_'+str(i)+'.npz') or overwrite_hitfinder:
                if os.path.exists(dirname+'/hits_evt_'+str(i)+'.npz'):
                    print(dirname+'/hits_evt_'+str(i)+'.npz exists, overwriting...')
                else:
                    print(dirname+'/hits_evt_'+str(i)+'.npz does not exist, processing...')
                print("Running hitfinder...")
                hits_evt, hits_config = interaction_finder(spes_evt, noise_evt)
                print("Hitfinder run, shape: ", hits_evt.shape)
                print("Total hits: ", np.sum(hits_evt != -1))

                # save config
                with open(dirname+'/hits_config_'+str(i)+'.json', 'w') as f:
                    json.dump(hits_config, f, indent=4)

                # save hits
                np.savez(dirname+'/hits_evt_'+str(i)+'.npz', hits_evt)

            else:
                print(dirname+'/hits_evt_'+str(i)+'.npz exists, exiting.') #loading hits...')
                #hits_file = np.load(dirname+'/hits_evt_'+str(i)+'.npz')
                #hits_evt = hits_file['arr_0']

    # get truth info
    if is_data and save_truth:
        print("No truth information for data...")

    if save_truth and not is_data:

        # loop over files
        for i, p in enumerate(path):

            # check if file exists
            true_hits_output = dirname+'/true_hits_'+str(i)+'.csv'
            if os.path.exists(true_hits_output):
                if overwrite_truth:
                    print("Overwriting truth information...")
                else:
                    print("Truth information already saved, exiting...")
                    continue

            # get truth info
            print("Getting truth info...")
            truth_information = get_truth(p, i, is_cont, 7500)

            # save as csv file
            cols = ['event_id',
                    'vertex_id',
                    'start_time',
                    'start_time_idx',
                    'tpc_num']
            #true_hits_output = dirname+'/true_hits_'+str(i)+'.csv'
            print("Creating output: ", true_hits_output)
            df = pd.DataFrame(truth_information, columns=cols)
            df.to_csv(true_hits_output, index=False)

    else:
        print("Truth information not saved!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some waveforms.')
    parser.add_argument('path', type=str, nargs='+', help='The name(s) of the file(s) to process')
    parser.add_argument('nfiles', type=int, help='The number of files to process')
    parser.add_argument('--is_data', action='store_true', help='Flag to indicate if the file is data')
    parser.add_argument('--summed', type=str, default=None, help='Summing method for channels')
    parser.add_argument('--max_evts', type=int, default=None, help='Maximum number of events to process')
    parser.add_argument('--run_hitfinder', action='store_true', help='Flag to indicate if hitfinder should be run')
    parser.add_argument('--opp', action='store_true', help='Flag to indicate if preprocessing should be overwritten')
    parser.add_argument('--ohf', action='store_true', help='Flag to indicate if hit finder output should be overwritten')
    parser.add_argument('--get_truth', action='store_true', help='Flag to indicate if truth info should be extracted')
    parser.add_argument('--owt', action='store_true', help='Flag to indicate if truth info should be overwritten')
    parser.add_argument('--is_cont', action='store_true', help='Flag to indicate if the vertices are contained in active volume')
    args = parser.parse_args()

    ## timing info
    number_of_cpus = os.cpu_count()
    clock_speed = 1.0 / time.get_clock_info('monotonic').resolution
    flops_per_cycle = 8 # for modern CPUs?
    print("Number of CPUs: ", number_of_cpus)
    print("Clock Speed: ", clock_speed)
    print("FLOPS per Cycle: ", flops_per_cycle)

    # start timer
    start_time = time.time()

    # execute main function for preprocessing data
    main(args.path, args.nfiles, args.is_data, args.summed, args.max_evts, args.run_hitfinder, args.opp, args.ohf, args.get_truth, args.owt, args.is_cont)

    # end timer
    end_time = time.time()
    seconds = end_time - start_time
    cpu_seconds = seconds * number_of_cpus
    cpu_flops = cpu_seconds * clock_speed * flops_per_cycle
    print(f"Seconds: {seconds:.4f}")
    print(f"CPU seconds: {cpu_seconds:.4f}")
    print(f"CPU FLOPS: {cpu_flops:.4e}")
