# lrs_sanity_check

For running on NERSC:
```
pip3 install h5py pandas scipy matplotlib
```
And see local hd5flow installation instractions:
< https://github.com/lbl-neutrino/h5flow >


```
python3 wvfm_processing.py <path>/<file.FLOW.hdf5> --summed <TPC,TrapType,Detector>
```

Additional options:
`--is_data` (otherwise MC)
`--run_hitfinder` (otherwise doesn't)
`--opp` (overwrites existing directory)
`--ohf` (overwrites existing hitfinding)

Run over processed directory using `wvfm_plotting{,_mc}.ipynb` for wvfm visualisation.

e.g. tested on NERSC 25-02-2025:
MC
```
python3 wvfm_processing.py /global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun5_1E19_RHC/MiniRun5_1E19_RHC.flow.beta2a/FLOW/0000000/MiniRun5_1E19_RHC.flow.0000000.FLOW.hdf5  --summed TPC --run_hitfinder --get_truth
```
Data
```
python3 wvfm_processing.py /global/cfs/cdirs/dune/www/data/2x2/nearline/flowed_light/data_bin004/mpd_run_hvramp_rctl_105_p350.FLOW.hdf5 --summed TPC --is_data --run_hitfinder
```

