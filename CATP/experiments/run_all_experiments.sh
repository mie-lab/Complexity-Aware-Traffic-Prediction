nohup python spatial_n_grids.py "0" > spatial_n_grids.txt &
sleep 5
nohup python temporal_horizon.py "1"> horiz.txt &
sleep 5
nohup python temporal_io_length.py "0" > io.txt &
