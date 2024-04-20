import os
import glob
import pickle
import numpy as np
import cv2
from tqdm import tqdm

from simulation_modeling.trajectory_interpolator import TrajInterpolator
from sim_evaluation_metric.realistic_metric import RealisticMetrics

from basemap import Basemap

class RealWorldVisualize(object):
    def __init__(self, configs):

        self.dataset = configs["dataset"]
        self.history_length, self.pred_length, self.m_tokens = configs["history_length"], configs["pred_length"], configs["m_tokens"]
        self.rolling_step = configs["rolling_step"]
        self.sim_wall_time = configs["sim_wall_time"]
        self.sim_resol = configs["sim_resol"]
        self.use_neural_safety_mapping = configs["use_neural_safety_mapping"]
        self.background_map = Basemap(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])
        self.viz_real_buff_flag = configs["viz_real_buff_flag"]
        self.save_viz_flag = configs["save_viz_flag"]
        self.save_fps = configs["save_fps"]
        # self.device = configs["device"]

        self.traj_dirs, self.subfolder_data_proportion, self.subsubfolder_data_proportion = self._get_traj_dirs(path_to_traj_data=configs["path_to_traj_data"])

        self.gen_realistic_metric_flag = configs["gen_realistic_metric_flag"]  # Whether to generate metrics.
        self.gen_realistic_metric_dict = configs["gen_realistic_metric_dict"]  # What metrics to generate.
        self.realistic_metric_save_folder = configs["realistic_metric_save_folder"]
        # os.makedirs(self.realistic_metric_save_folder, exist_ok=True)

        self.interpolate_real_buff_flag = configs["interpolate_real_buff_flag"]
        if self.interpolate_real_buff_flag:
            # initialize the trajectory interpolator
            # The number of steps interpolate between predicted steps. For example, if resolution is 0.4s and steps is 3, then new resolution is 0.1s.
            self.real_buff_intep_steps = configs["real_buff_intep_steps"]
            self.traj_interpolator = TrajInterpolator()

    @staticmethod
    def _get_traj_dirs(path_to_traj_data):
        subfolders = sorted(os.listdir(os.path.join(path_to_traj_data)))
        subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                         range(len(subfolders))]

        traj_dirs = []
        each_subfolder_size = []
        each_subsubfolder_size = []
        for i in range(len(subfolders)):
            one_video = []
            each_subsubfolder_size_tmp = []
            for j in range(len(subsubfolders[i])):
                files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
                one_video.append(files_list)
                each_subsubfolder_size_tmp.append(len(files_list))
            traj_dirs.append(one_video)
            each_subfolder_size.append(sum([len(listElem) for listElem in one_video]))
            each_subsubfolder_size.append(each_subsubfolder_size_tmp)

        subfolder_data_proportion = [each_subfolder_size[i] / sum(each_subfolder_size) for i in range(len(each_subfolder_size))]
        subsubfolder_data_proportion = [[each_subsubfolder_size[i][j] / sum(each_subsubfolder_size[i]) for j in range(len(each_subsubfolder_size[i]))] for i in
                                        range(len(each_subsubfolder_size))]

        return traj_dirs, subfolder_data_proportion, subsubfolder_data_proportion

    def _get_time_buff(self, files_list):
        # TIME_BUFF = []
        TIME_BUFF = {}
        pickle_sum = len(files_list)
        # print("start idx: {0}".format(str(t0 - history_length)))
        for i in tqdm(range(pickle_sum),desc='getting time buff of subsubfolder'):
            vehicle_list = pickle.load(open(files_list[i], "rb"))
            # TIME_BUFF.append(vehicle_list)
            TIME_BUFF[files_list[i][-33:-7]] = vehicle_list

        return TIME_BUFF

    def visualize_time_buff(self, TIME_BUFF, tt=0):
        # visualization and write result video
        if self.viz_real_buff_flag:
            if self.interpolate_real_buff_flag:
                visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF, intep_steps=self.real_buff_intep_steps, original_resolution=self.sim_resol)
                freq = self.real_buff_intep_steps + 1
            else:
                visualize_TIME_BUFF = TIME_BUFF
                freq = 1

            if tt == 0:
                self._visualize_time_buff(visualize_TIME_BUFF, self.background_map)
            else:
                self._visualize_time_buff(visualize_TIME_BUFF[-self.rolling_step * freq:], self.background_map)

    def _visualize_time_buff(self, TIME_BUFF, background_map):
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            vis = background_map.render(vehicle_list, with_traj=True, linewidth=6)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            cv2.imshow('vis', img)  # rgb-->bgr
            cv2.waitKey(1)

    def save_time_buff_video(self, TIME_BUFF, background_map, file_name, save_path, color_vid_list=None, with_traj=True):
        if self.interpolate_real_buff_flag:
            visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF,
                                                                             intep_steps=self.real_buff_intep_steps,
                                                                             original_resolution=self.sim_resol)
        else:
            visualize_TIME_BUFF = TIME_BUFF

        os.makedirs(save_path, exist_ok=True)
        playback_video_writer = cv2.VideoWriter(save_path + r'/{0}.mp4'.format(file_name), cv2.VideoWriter_fourcc(*"mp4v"), self.save_fps, (background_map.w, background_map.h))

        for timestamp_step, vehicle_list_step in tqdm(visualize_TIME_BUFF.items(), desc='Processing saving videos'): # range(len(visualize_TIME_BUFF)):
            timestamp = timestamp_step
            vehicle_list = vehicle_list_step
            vis = background_map.render(timestamp, vehicle_list, with_traj=with_traj, linewidth=6, color_vid_list=color_vid_list)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (3242, int(3242 * background_map.h / background_map.w)))  # resize when needed
            playback_video_writer.write(img)
        
        playback_video_writer.release()