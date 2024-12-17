# lrs_sanity_check

```
python wvfm_processing.py <path>/<file.FLOW.hdf5> --summed <TPC,TrapType,Detector>
```

Additional options:
`--is_data` (otherwise MC)
`--run_hitfinder` (otherwise doesn't)
`--opp` (overwrites existing directory)
`--ohf` (overwrites existing hitfinding)

Run over processed directory using `wvfm_plotting.ipynb` for wvfm visualisation.