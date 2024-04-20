import yaml
import shutil
import os
import torch
import argparse
import warnings
import glob
from tqdm import tqdm
import logging

from realistic_metric_yz.realistic_analysis import RealisticAnalysis

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, default=r'./configs/yz_252_preprocess.yml',
                    help='The path to the simulation config file. E.g., ./configs/AA_rdbt_inference.yml')

args = parser.parse_args()

if __name__ == '__main__':
    # Load config

    logging.basicConfig(filename='252_ds_feature_pltyx_new_PET_0409.log', level=logging.INFO)

    with open(args.config, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    yz_realistic_analyzer = RealisticAnalysis(configs=configs)
    background_map = yz_realistic_analyzer.background_map
    buff_video_path = configs['buff_video_save_path']

    # load data_yz
    path_to_traj_data = configs["path_to_traj_data"]
    subfolders = sorted(os.listdir(os.path.join(path_to_traj_data))) # 001
    subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                        range(len(subfolders))] # 以rcu_252_0306为例，002~007
    # print(len(subsubfolders)) #
    # print(subsubfolders) #
    
    for i in range(len(subfolders)):
        for j in tqdm(range(len(subsubfolders[i])), desc='generating subsubforder features'): # len(subsubfolders[i])
            files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
            buff_name = subsubfolders[i][j]
            print(f'start generating features of {buff_name}')
            try:
                time_buff = yz_realistic_analyzer._get_time_buff(files_list)
            except EOFError:
                logging.info(f'TIME_BUFF{buff_name} occurs EOFError, skipped')
                continue
            yz_realistic_analyzer.generate_realistic_metric(time_buff)
            print(f'new feature of {buff_name} generated successfully')
            # 每次都存储
            yz_realistic_analyzer.save_realistic_metric()
            print(f'252 features before {buff_name} newly saved successfully')


