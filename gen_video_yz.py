import yaml
import os
import argparse
import glob
from tqdm import tqdm

from realistic_visualization_yz.realistic_playback import RealWorldVisualize

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, default=r'./configs/yz_252_preprocess.yml',
                    help='The path to the simulation config file. E.g., ./configs/AA_rdbt_inference.yml')

args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Load config
    with open(args.config, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    yz_realistic_vis = RealWorldVisualize(configs=configs)
    background_map = yz_realistic_vis.background_map
    buff_video_path = configs['buff_video_save_path']

    # load data_yz
    path_to_traj_data = configs["path_to_traj_data"]
    subfolders = sorted(os.listdir(os.path.join(path_to_traj_data))) # 001
    subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                        range(len(subfolders))] # 以rcu_252_0306为例，002~007

# len(subfolders)
# len(subsubfolders)

    for i in range(len(subfolders)):
        for j in tqdm(range(20,40), desc='saving videos'):
            files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
            buff_name = subsubfolders[i][j]
            print(f'start processing time buff of {buff_name}')
            time_buff = yz_realistic_vis._get_time_buff(files_list)
            # save time buff as video
            # buff_name = subsubfolders[i][j]
            video_name = f'{buff_name}_withtraj_nobox_1820'
            yz_realistic_vis.save_time_buff_video(time_buff, background_map=background_map, file_name=video_name, save_path=buff_video_path)
            print(f'realistic playback of {buff_name} has been generated successfully')

    