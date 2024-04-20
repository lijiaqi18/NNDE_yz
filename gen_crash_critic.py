import pandas as pd
import pickle
import os
import glob
from tqdm import tqdm
import logging
import cv2
import numpy as np
from vehicle import Vehicle
from geo_engine import GeoEngine
from ROIs_yz.ROIs_252 import ROIMatcher_252
from datetime import datetime

base_map_dir = 'data_yz/inference/rcu_252/basemap/252.png'
initail_region_dir = 'data_yz/inference/rcu_252/ROIs-map/vehicle-initial-region'
initial_pickle_dir = 'data_yz/inference/rcu_252/simulation_initialization/initial_clips'
# output_initial_vehicle_filename = 'data_yz/inference/rcu_252/simulation_initialization/gen_veh_states/initial_vehicle_dict_0403.pickle'

# 单位：crash/km

time_format = '%Y-%m-%d %H-%M-%S-%f'
start_time = '2024-01-24 08-24-31-991000'
end_time = '2024-01-24 12-26-01-828000'
time_region = datetime.strptime(end_time, time_format) - datetime.strptime(start_time, time_format)
time_region_h = time_region.total_seconds() / 3600

logging.basicConfig(filename='252_near_miss_rate_cal_0409.log', level=logging.INFO)

path_to_traj_data = initial_pickle_dir
subfolders = sorted(os.listdir(os.path.join(path_to_traj_data))) # 001
subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                    range(len(subfolders))] # 以rcu_252_0306为例，002~007


for i in range(len(subfolders)):
    for j in tqdm(range(len(subsubfolders[i])), desc='counting near-miss scenes'): # len(subsubfolders[i])
        # print(j)
        files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
        buff_name = subsubfolders[i][j]
        # print(f'start generating features of {buff_name}')
        pickle_sum = len(files_list)
        try:
            for k in tqdm(range(pickle_sum),desc='counting near-miss scenes of each frame'):
                vehicle_list = pickle.load(open(files_list[k], "rb")) # file_list[i]是一个pickle，即一帧
                for v in vehicle_list:
                    pass
                    
        except EOFError:
            logging.info(f'TIME_BUFF{buff_name} occurs EOFError, skipped')
            continue
        logging.info(f'TIME_BUFF{buff_name} calculated successfully')



logging.info(f'arrive rate calculated successfully')