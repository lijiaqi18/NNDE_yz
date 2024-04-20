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

base_map_dir = 'data_yz/inference/rcu_252/basemap/252.png'
initail_region_dir = 'data_yz/inference/rcu_252/ROIs-map/vehicle-initial-region'
initial_pickle_dir = 'data_yz/inference/rcu_252/simulation_initialization/initial_clips'
output_initial_vehicle_filename = 'data_yz/inference/rcu_252/simulation_initialization/gen_veh_states/initial_vehicle_dict_0403.pickle'

map_width = 1896
map_height = 2091

if initail_region_dir is not None:
    circle_1_in1_map = cv2.imread(os.path.join(initail_region_dir, '1_in1.png'), cv2.IMREAD_GRAYSCALE)
    circle_1_in2_map = cv2.imread(os.path.join(initail_region_dir, '1_in2.png'), cv2.IMREAD_GRAYSCALE)
    circle_1_in3_map = cv2.imread(os.path.join(initail_region_dir, '1_in3.png'), cv2.IMREAD_GRAYSCALE)

    circle_2_in1_map = cv2.imread(os.path.join(initail_region_dir, '2_in1.png'), cv2.IMREAD_GRAYSCALE)
    circle_2_in2_map = cv2.imread(os.path.join(initail_region_dir, '2_in2.png'), cv2.IMREAD_GRAYSCALE)
    circle_2_in3_map = cv2.imread(os.path.join(initail_region_dir, '2_in3.png'), cv2.IMREAD_GRAYSCALE)
    circle_2_in4_map = cv2.imread(os.path.join(initail_region_dir, '2_in4.png'), cv2.IMREAD_GRAYSCALE)

    circle_3_in1_map = cv2.imread(os.path.join(initail_region_dir, '3_in1.png'), cv2.IMREAD_GRAYSCALE)
    circle_3_in2_map = cv2.imread(os.path.join(initail_region_dir, '3_in2.png'), cv2.IMREAD_GRAYSCALE)
    circle_3_in3_map = cv2.imread(os.path.join(initail_region_dir, '3_in3.png'), cv2.IMREAD_GRAYSCALE)
    
    circle_1_in1_map = cv2.resize(circle_1_in1_map, (map_width, map_height))
    circle_1_in2_map = cv2.resize(circle_1_in2_map, (map_width, map_height))
    circle_1_in3_map = cv2.resize(circle_1_in3_map, (map_width, map_height))

    circle_2_in1_map = cv2.resize(circle_2_in1_map, (map_width, map_height))
    circle_2_in2_map = cv2.resize(circle_2_in2_map, (map_width, map_height))
    circle_2_in3_map = cv2.resize(circle_2_in3_map, (map_width, map_height))
    circle_2_in4_map = cv2.resize(circle_2_in4_map, (map_width, map_height))

    circle_3_in1_map = cv2.resize(circle_3_in1_map, (map_width, map_height))
    circle_3_in2_map = cv2.resize(circle_3_in2_map, (map_width, map_height))
    circle_3_in3_map = cv2.resize(circle_3_in3_map, (map_width, map_height))

def initial_region_position_matching(pxl_pt):
    initial_region_position = 'not_initial_region'
    y0, x0 = pxl_pt[0], pxl_pt[1]

    # circle_1 entrance
    if circle_1_in1_map[x0, y0] > 128.:
        initial_region_position = '1_in1'
        return initial_region_position
    if circle_1_in2_map[x0, y0] > 128.:
        initial_region_position = '1_in2'
        return initial_region_position
    if circle_1_in3_map[x0, y0] > 128.:
        initial_region_position = '1_in3'
        return initial_region_position
    
    # circle_2 entrance
    if circle_2_in1_map[x0, y0] > 128.:
        initial_region_position = '2_in1'
        return initial_region_position
    if circle_2_in2_map[x0, y0] > 128.:
        initial_region_position = '2_in2'
        return initial_region_position
    if circle_2_in3_map[x0, y0] > 128.:
        initial_region_position = '2_in3'
        return initial_region_position
    if circle_2_in4_map[x0, y0] > 128.:
        initial_region_position = '2_in4'
        return initial_region_position
    
    # circle_1 entrance
    if circle_3_in1_map[x0, y0] > 128.:
        initial_region_position = '3_in1'
        return initial_region_position
    if circle_3_in2_map[x0, y0] > 128.:
        initial_region_position = '3_in2'
        return initial_region_position
    if circle_3_in3_map[x0, y0] > 128.:
        initial_region_position = '3_in3'
        return initial_region_position
    
base_map = GeoEngine(base_map_dir, map_height, map_width)

initial_vehicle_set = {}
initial_vehicle_set['1_in1'] = []
initial_vehicle_set['1_in2'] = []
initial_vehicle_set['1_in3'] = []
initial_vehicle_set['2_in1'] = []
initial_vehicle_set['2_in2'] = []
initial_vehicle_set['2_in3'] = []
initial_vehicle_set['2_in4'] = []
initial_vehicle_set['3_in1'] = []
initial_vehicle_set['3_in2'] = []
initial_vehicle_set['3_in3'] = []

logging.basicConfig(filename='252_initial_gen_0403.log', level=logging.INFO)

path_to_traj_data = initial_pickle_dir
subfolders = sorted(os.listdir(os.path.join(path_to_traj_data))) # 001
subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                    range(len(subfolders))] # 以rcu_252_0306为例，002~007
# print(len(subfolders))
# print(subfolders)
# print(len(subsubfolders[0])) # 
# print(subsubfolders[0][1]) #

for i in range(len(subfolders)):
    for j in tqdm(range(len(subsubfolders[i])), desc='searching initial regions'): # len(subsubfolders[i])
        # print(j)
        files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
        buff_name = subsubfolders[i][j]
        # print(f'start generating features of {buff_name}')
        pickle_sum = len(files_list)
        try:
            for k in tqdm(range(pickle_sum),desc='getting initial region vehicles of subsubfolder'):
                vehicle_list = pickle.load(open(files_list[k], "rb")) # file_list[i]是一个pickle，即一帧
                for v in vehicle_list:
                    pxl_pt = base_map._world2pxl([v.location.x, v.location.y])
                    pxl_pt[1] = np.clip(pxl_pt[1], a_min=0, a_max=map_width-1) # road_map (map_width, map_height)
                    pxl_pt[0] = np.clip(pxl_pt[0], a_min=0, a_max=map_height-1)
                    if initial_region_position_matching(pxl_pt) == 'not_initial_region':
                        continue
                    if initial_region_position_matching(pxl_pt) == '1_in1':
                        initial_vehicle_set['1_in1'].append(v)
                    if initial_region_position_matching(pxl_pt) == '1_in2':
                        initial_vehicle_set['1_in2'].append(v)
                    if initial_region_position_matching(pxl_pt) == '1_in3':
                        initial_vehicle_set['1_in3'].append(v)
                    if initial_region_position_matching(pxl_pt) == '2_in1':
                        initial_vehicle_set['2_in1'].append(v)
                    if initial_region_position_matching(pxl_pt) == '2_in2':
                        initial_vehicle_set['2_in2'].append(v)
                    if initial_region_position_matching(pxl_pt) == '2_in3':
                        initial_vehicle_set['2_in3'].append(v)
                    if initial_region_position_matching(pxl_pt) == '2_in4':
                        initial_vehicle_set['2_in4'].append(v)
                    if initial_region_position_matching(pxl_pt) == '3_in1':
                        initial_vehicle_set['3_in1'].append(v)
                    if initial_region_position_matching(pxl_pt) == '3_in2':
                        initial_vehicle_set['3_in2'].append(v)
                    if initial_region_position_matching(pxl_pt) == '3_in3':
                        initial_vehicle_set['3_in3'].append(v)
        except EOFError:
            logging.info(f'TIME_BUFF{buff_name} occurs EOFError, skipped')
            continue
        logging.info(f'TIME_BUFF{buff_name} searched successfully')

with open(output_initial_vehicle_filename, 'wb') as f:
    pickle.dump(initial_vehicle_set, f)

logging.info(f'initial_vehicle_dict generated successfully')