import yaml
import shutil
import os
import torch
import argparse
import warnings
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from realistic_metric_yz.realistic_analysis import RealisticAnalysis

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, default=r'./configs/yz_252_preprocess.yml',
                    help='The path to the simulation config file. E.g., ./configs/AA_rdbt_inference.yml')

args = parser.parse_args()

if __name__ == '__main__':
    # Load config
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

    def process_subsubfoldrs(subsubfolder, ids):
        files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[ids], subsubfolder, '*.pickle')))
        buff_name = subsubfolder
        print(f'start generating features of {buff_name}')
        time_buff = yz_realistic_analyzer._get_time_buff(files_list)
        yz_realistic_analyzer.generate_realistic_metric(time_buff)
        print(f'new feature of {buff_name} generated successfully')
    
    for i in range(len(subfolders)):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # 提交任务到线程池，每个任务都需要文件路径、操作和值
            futures = [executor.submit(process_subsubfoldrs, subsubfolder, i) for subsubfolder in subsubfolders[i]]

    yz_realistic_analyzer.save_realistic_metric()


