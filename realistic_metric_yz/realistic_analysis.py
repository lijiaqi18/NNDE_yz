import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import json
import pandas as pd
import time
from tqdm import tqdm

from simulation_modeling.traffic_simulator import TrafficSimulator
from simulation_modeling.crashcritic import CrashCritic
from simulation_modeling.trajectory_interpolator import TrajInterpolator
from simulation_modeling.vehicle_generator import AA_rdbt_TrafficGenerator, rounD_TrafficGenerator
from sim_evaluation_metric.realistic_metric_yz import RealisticMetricsYZ
from sim_evaluation_metric.realistic_metric import RealisticMetrics
from trajectory_pool import TrajectoryPool

from basemap import Basemap


class RealisticAnalysis(object):
    def __init__(self, configs):

        self.dataset = configs["dataset"]
        self.history_length, self.pred_length, self.m_tokens = configs["history_length"], configs["pred_length"], configs["m_tokens"]
        self.background_map = Basemap(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])
        # self.device = configs["device"]

        
        self.traj_dirs, self.subfolder_data_proportion, self.subsubfolder_data_proportion = self._get_traj_dirs(path_to_traj_data=configs["path_to_traj_data"])

        self.gen_realistic_metric_flag = configs["gen_realistic_metric_flag"]  # Whether to generate metrics.
        self.gen_realistic_metric_dict = configs["gen_realistic_metric_dict"]  # What metrics to generate.
        self.realistic_metric_save_folder = configs["realistic_metric_save_folder"]
        # os.makedirs(self.realistic_metric_save_folder, exist_ok=True)

        if self.gen_realistic_metric_flag:

            # ROIs
            circle_map_dir = os.path.join(configs["ROI_map_dir"], 'circle')
            entrance_map_dir = os.path.join(configs["ROI_map_dir"], 'entrance')
            exit_map_dir = os.path.join(configs["ROI_map_dir"], 'exit')
            # crosswalk_map_dir = os.path.join(configs["ROI_map_dir"], 'crosswalk')
            yielding_area_map_dir = os.path.join(configs["ROI_map_dir"], 'yielding-area')
            at_circle_lane_map_dir = os.path.join(configs["ROI_map_dir"], 'at-circle-lane')

            self.sim_resol = configs['sim_resol']
            # PET (post-encroachment time) analysis configs
            if self.dataset == 'yz_252':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

                self.RealWorldMetricsAnalyzer = RealisticMetricsYZ(drivable_map_dir=configs["drivable_map_dir"], 
                                                               sim_remove_vehicle_area_map=None,
                                                                circle_map_dir=circle_map_dir, 
                                                                entrance_map_dir=entrance_map_dir, 
                                                                exit_map_dir=exit_map_dir,
                                                                crosswalk_map_dir=None, 
                                                                yielding_area_map_dir=yielding_area_map_dir, 
                                                                at_circle_lane_map_dir=at_circle_lane_map_dir,
                                                                sim_resol=self.sim_resol,
                                                                dataset = self.dataset,
                                                                map_height=configs["map_height"], map_width=configs["map_width"],
                                                                PET_configs=PET_configs)

            elif self.dataset == 'yz_13':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

                self.RealWorldMetricsAnalyzer = RealisticMetricsYZ(drivable_map_dir=configs["drivable_map_dir"], 
                                                               sim_remove_vehicle_area_map=None,
                                                                circle_map_dir=circle_map_dir, 
                                                                entrance_map_dir=entrance_map_dir, 
                                                                exit_map_dir=exit_map_dir,
                                                                crosswalk_map_dir=None, 
                                                                yielding_area_map_dir=yielding_area_map_dir, 
                                                                at_circle_lane_map_dir=at_circle_lane_map_dir,
                                                                sim_resol=self.sim_resol,
                                                                dataset = self.dataset,
                                                                map_height=configs["map_height"], map_width=configs["map_width"],
                                                                PET_configs=PET_configs)
            
            

            elif self.dataset == 'AA_rdbt':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

                self.RealWorldMetricsAnalyzer = RealisticMetrics(drivable_map_dir=configs["drivable_map_dir"], 
                                                               sim_remove_vehicle_area_map=None,
                                                                circle_map_dir=circle_map_dir, 
                                                                entrance_map_dir=entrance_map_dir, 
                                                                exit_map_dir=exit_map_dir,
                                                                crosswalk_map_dir=None, 
                                                                yielding_area_map_dir=yielding_area_map_dir, 
                                                                at_circle_lane_map_dir=at_circle_lane_map_dir,
                                                                sim_resol=self.sim_resol,
                                                                map_height=configs["map_height"], map_width=configs["map_width"],
                                                                PET_configs=PET_configs)

            
            self.output_instant_speed_list = []  # This list is instant speed in the circle
            self.output_yielding_conflict_dist_and_v_dict_list = []
            self.output_distance_all_vehicle_pairs_list_three_circle = []
            self.output_PET_list = []  # Post-encroachment time results.

        self.interpolate_flag = configs["interpolate_flag"]
        if self.interpolate_flag:
            # initialize the trajectory interpolator
            # The number of steps interpolate between predicted steps. For example, if resolution is 0.4s and steps is 3, then new resolution is 0.1s.
            self.intep_steps = configs["intep_steps"]
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
        TIME_BUFF = []
        # TIME_BUFF = {}
        pickle_sum = len(files_list)
        # print("start idx: {0}".format(str(t0 - history_length)))
        for i in tqdm(range(pickle_sum),desc='getting time buff of subsubfolder'):
            vehicle_list = pickle.load(open(files_list[i], "rb")) # file_list[i]是一个pickle，即一帧
            TIME_BUFF.append(vehicle_list)
            # TIME_BUFF[files_list[i][-33:-7]] = vehicle_list

        return TIME_BUFF # TIME_BUFF来自一个子文件夹
    

    def generate_realistic_metric(self, one_TIME_BUFF):
        """
        The input is the simulation episode index and the simulated trajectories of this episode.
        This function will calculate the simulation metrics (e.g., vehicle speed, distance, etc.)
        """
        if self.interpolate_flag:
            # interpolate the trajectory to a finer resolution first for analysis
            evaluate_metric_TIME_BUFF, new_resol = self.traj_interpolator.interpolate_traj(one_TIME_BUFF, intep_steps=self.intep_steps, original_resolution=self.sim_resol)
            self.RealWorldMetricsAnalyzer.sim_resol = new_resol
        else:
            evaluate_metric_TIME_BUFF = one_TIME_BUFF
        self._gen_realistic_metric(one_TIME_BUFF=evaluate_metric_TIME_BUFF)

    def _gen_realistic_metric(self, one_TIME_BUFF):
        # Construct traj dataframe
        self.RealWorldMetricsAnalyzer.construct_traj_data(one_TIME_BUFF)

        # PET analysis
        if self.gen_realistic_metric_dict["PET"]:
            PET_list = self.RealWorldMetricsAnalyzer.PET_analysis()
            self.output_PET_list.append(PET_list)

        # In circle instant speed analysis
        if self.gen_realistic_metric_dict["instant_speed"]:
            instant_speed_list = self.RealWorldMetricsAnalyzer.in_circle_instant_speed_analysis()
            self.output_instant_speed_list.append(instant_speed_list)

        # yielding distance and speed analysis
        if self.gen_realistic_metric_dict["yielding_speed_and_distance"]:
            yielding_conflict_dist_and_v_dict = self.RealWorldMetricsAnalyzer.yielding_distance_and_speed_analysis()
            self.output_yielding_conflict_dist_and_v_dict_list.append(yielding_conflict_dist_and_v_dict)

        # all positions distance distribution analysis
        if self.gen_realistic_metric_dict["distance"]:
            distance_all_vehicle_pairs_list_three_circle = self.RealWorldMetricsAnalyzer.distance_analysis(mode='three_circle', only_in_roundabout_circle=False)
            self.output_distance_all_vehicle_pairs_list_three_circle.append(distance_all_vehicle_pairs_list_three_circle)

    def save_realistic_metric(self):
        if self.gen_realistic_metric_dict["PET"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_PET_list.json"), 'w') as f:
                json.dump(self.output_PET_list, f, indent=4)
            print('PET results are saved.')

        if self.gen_realistic_metric_dict["instant_speed"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_instant_speed_list.json"), 'w') as f:
                json.dump(self.output_instant_speed_list, f, indent=4)
            print('Instant speed results are saved.')

        if self.gen_realistic_metric_dict["yielding_speed_and_distance"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_yielding_conflict_dist_and_v_dict_list.json"), 'w') as f:
                json.dump(self.output_yielding_conflict_dist_and_v_dict_list, f, indent=4)
            print('Yielding distance and speed results are saved.')

        if self.gen_realistic_metric_dict["distance"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_distance_list_three_circle.json"), 'w') as f:
                json.dump(self.output_distance_all_vehicle_pairs_list_three_circle, f, indent=4)
            print('Distance results are saved.')


    def visualize_time_buff(self, TIME_BUFF, tt=0):
        # visualization and write result video
        if self.viz_flag:
            if self.interpolate_flag:
                visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF, intep_steps=self.intep_steps, original_resolution=self.sim_resol)
                freq = self.intep_steps + 1
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
        if self.interpolate_flag:
            visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF,
                                                                             intep_steps=self.intep_steps,
                                                                             original_resolution=self.sim_resol)
        else:
            visualize_TIME_BUFF = TIME_BUFF

        os.makedirs(save_path, exist_ok=True)
        collision_video_writer = cv2.VideoWriter(save_path + r'/{0}.mp4'.format(file_name), cv2.VideoWriter_fourcc(*'MP4V'), self.save_fps, (background_map.w, background_map.h))
        for i in range(len(visualize_TIME_BUFF)):
            vehicle_list = visualize_TIME_BUFF[i]
            vis = background_map.render(vehicle_list, with_traj=with_traj, linewidth=6, color_vid_list=color_vid_list)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            collision_video_writer.write(img)

    def save_trajectory(self, TIME_BUFF, save_path, sim_id, step_id):
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            frame_id = step_id + i
            output_file_path = os.path.join(save_path, str(sim_id) + '-' + str(frame_id).zfill(5) + '.pickle')
            pickle.dump(vehicle_list, open(output_file_path, "wb"))
