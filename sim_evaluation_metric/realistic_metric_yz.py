"""
This class is to calculate realistic metrics to validate the performance of the proposed simulator.
"""
import os
import pandas as pd
import copy
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from itertools import combinations
from scipy.spatial.distance import pdist
from tqdm import tqdm
from datetime import datetime


from road_matching import RoadMatcher
from ROIs_yz.ROIs_252 import ROIMatcher_252
from trajectory_pool import TrajectoryPool


class RealisticMetricsYZ(object):

    def __init__(self, drivable_map_dir=None, sim_remove_vehicle_area_map=None,
                 circle_map_dir=None, entrance_map_dir=None, exit_map_dir=None, crosswalk_map_dir=None, yielding_area_map_dir=None, at_circle_lane_map_dir=None,
                 sim_resol=0.4,
                 dataset = None,
                 map_height=936, map_width=1678,
                 PET_configs=None):

        self.road_matcher = RoadMatcher(map_file_dir=drivable_map_dir, map_height=map_height, map_width=map_width)

        self.ROI_matcher = ROIMatcher_252(drivable_map_dir=drivable_map_dir, sim_remove_vehicle_area_map_dir=sim_remove_vehicle_area_map, circle_map_dir=circle_map_dir,
                                      entrance_map_dir=entrance_map_dir, exit_map_dir=exit_map_dir, crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir,
                                      at_circle_lane_map_dir=at_circle_lane_map_dir,
                                      map_height=map_height, map_width=map_width)

        self.traj_pool = None
        self.traj_df = None
        self.TIME_BUFF = None
        self.sim_resol = sim_resol  # simulation resolution in [s]
        self.dataset = dataset
        self.PET_configs = PET_configs

    def time_buff_to_traj_pool(self, TIME_BUFF):
        traj_pool = TrajectoryPool(max_missing_age=float("inf"), road_matcher=self.road_matcher, ROI_matcher=self.ROI_matcher)
        # for i in range(len(TIME_BUFF)):
        #     traj_pool.update(TIME_BUFF[i], ROI_matching=True)
        # return traj_pool
        for vehicle_list in tqdm(TIME_BUFF, desc='converting time buff to traj pool'):
            traj_pool.update(vehicle_list, ROI_matching=True)
            # print(timestamp)
        return traj_pool

    def construct_traj_data(self, TIME_BUFF):

        self.TIME_BUFF = TIME_BUFF

        time_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        # Construct traj pool
        self.traj_pool = self.time_buff_to_traj_pool(TIME_BUFF)
        self.traj_df = pd.DataFrame(columns=['vid', 'x', 'y', 'heading', 'region_position', 'yielding_area', 'at_circle_lane', 't', 'timestamp', 'update', 'vehicle', 'dt', 'missing_days'])

        # Construct traj df
        for vid in tqdm(self.traj_pool.vehicle_id_list(), desc='constructing vehicle traj data'):
            self.traj_df = self.traj_df.append(pd.DataFrame.from_dict(self.traj_pool.pool[vid]), ignore_index=True)
        
        self.traj_df.to_csv(f'./z_feature_json_check/constructed_traj_252_ds_{time_str}.csv')

    def construct_traj_data_clear(self, TIME_BUFF):

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Construct traj pool
        self.TIME_BUFF = TIME_BUFF
        self.traj_pool = self.time_buff_to_traj_pool(TIME_BUFF)
        self.traj_df = pd.DataFrame(columns=['vid', 'x', 'y', 'heading', 'region_position', 'yielding_area', 'at_circle_lane', 't','timestamp', 'update', 'vehicle', 'dt', 'missing_days'])

        # Construct traj df
        for vid in tqdm(self.traj_pool.vehicle_id_list(), desc='constructing vehicle traj data'):
            vid_data = pd.DataFrame.from_dict(self.traj_pool.pool[vid])
            # vid_data.to_csv(f'./z_feature_json_check/0312_mini_clear_check/origin/vid_data_check_{vid}_{time_str}.csv')

            vid_data_unique = vid_data.groupby('t').first().reset_index()
            # vid_data_unique.to_csv(f'./z_feature_json_check/0312_mini_clear_check/groupby/vid_data_check_{vid}_{time_str}.csv')

            # 如果车辆exit与circle区域内位置一直不变，则不用添加到traj_df
            if (vid_data_unique['region_position'].isin(['circle_1_t', 'circle_2_t', 'circle_3_t', 'exit_1_t', 'exit_2_t', 'exit_3_t','exit_1_t_rightturn', 'offroad']).any() and 
                (vid_data_unique['x'].nunique() == 1) and 
                (vid_data_unique['y'].nunique() == 1)):
                continue
            
            # 如果车辆轨迹数据在一段时间后位置不变，删除之后不变的部分
            constant_xy_idx = None
            for i in range(len(vid_data_unique)-1, 0, -1):
                if (vid_data_unique.loc[i, 'x'] == vid_data_unique.loc[i-1, 'x']) and \
                (vid_data_unique.loc[i, 'y'] == vid_data_unique.loc[i-1, 'y']):
                    constant_xy_idx = i
                else:
                    # 一旦发现x或y值发生变化，停止搜索
                    break
            # 如果找到了不变的索引，删除该索引之后的所有行
            if constant_xy_idx is not None:
                vid_data_unique = vid_data_unique.iloc[:constant_xy_idx].copy()
            # vid_data_unique.to_csv(f'./z_feature_json_check/0312_mini_clear_check/constant_clear/vid_data_check_{vid}_{time_str}.csv')
            

            # 将处理后的数据追加到 traj_df
            self.traj_df = self.traj_df.append(vid_data_unique, ignore_index=True)

        self.traj_df.to_csv(f'./z_feature_json_check/constructed_traj_origin_check_mini_clear_{time_str}.csv')
        
            

    # def construct_traj_data(self, TIME_BUFF):
    #     self.TIME_BUFF = TIME_BUFF
    #     # Construct traj pool
    #     self.traj_pool = self.time_buff_to_traj_pool(TIME_BUFF)
    #     self.traj_df = pd.DataFrame(columns=['vid', 'x', 'y', 'heading', 'region_position', 'yielding_area', 'at_circle_lane', 't', 'timestamp', 'update', 'vehicle', 'dt', 'missing_days'])
    #     # Construct traj df
    #     for vid in tqdm(self.traj_pool.vehicle_id_list(), desc='constructing vehicle traj data'):
    #         self.traj_df = self.traj_df.append(pd.DataFrame.from_dict(self.traj_pool.pool[vid]), ignore_index=True)
    #         # 从 traj_pool 中获取 vid 对应的数据
    #         vid_data = pd.DataFrame.from_dict(self.traj_pool.pool[vid])
            
    #         # 对 vid_data 根据 't' 列进行分组，并选择每个 't' 的第一个记录
    #         vid_data_unique = vid_data.groupby('t').first().reset_index()
            
    #         # 将处理后的数据追加到 traj_df
    #         self.traj_df = self.traj_df.append(vid_data_unique, ignore_index=True)
        
    #     # 确保 traj_df 中的信息是唯一的
    #     # time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     # self.traj_df.to_csv(f'./z_feature_json_check/constructed_traj_unique_t_check_{time_str}.csv')
    #     self.traj_df = self.traj_df.groupby(['vid', 't']).first().reset_index()
    #     # self.traj_df.to_csv(f'./z_feature_json_check/constructed_traj_unique_t_vid_check_{time_str}.csv')


    def in_circle_instant_speed_analysis(self):
        instant_speed_list = []
        for vid in tqdm(self.traj_pool.vehicle_id_list(), desc='instant speed results processing'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]


            v_in_circle_traj = v_traj[v_traj['region_position'].isin(['circle_1_t', 'circle_2_t', 'circle_3_t'])]
            # print(v_in_circle_traj.columns)
            # v_in_circle_traj = v_in_circle_traj.groupby('t').first().reset_index()

            # import ipdb 
            # ipdb.set_trace()

            v_in_circle_traj.loc[:, "dt"] = v_in_circle_traj["t"].diff()
            v_in_circle_traj.loc[:, "dx"] = v_in_circle_traj["x"].diff()
            v_in_circle_traj.loc[:, "dy"] = v_in_circle_traj["y"].diff()
            v_in_circle_traj.loc[:, "travel_distance"] = (v_in_circle_traj["dx"] ** 2 + v_in_circle_traj["dy"] ** 2) ** 0.5
            # print(v_in_circle_traj.loc[:, "dt"])
            # assert (v_in_circle_traj.dt.dropna() > 0).all()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not (v_in_circle_traj.dt.dropna() > 0).all():
                # time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                abnormal_data = v_in_circle_traj[v_in_circle_traj.dt <= 0]
                v_in_circle_traj.to_csv(f'./z_abnormal_data_collection/v_in_crcle_traj_{time_str}.csv', index=False)
                v_in_circle_traj["t"].to_csv(f'./z_abnormal_data_collection/v_in_circle_traj_t_data_{time_str}.csv', index=False)
                abnormal_data.to_csv(f'./z_abnormal_data_collection/v_in_crcle_traj_abnormal_data_{time_str}.csv', index=False)

            # v_in_circle_traj.to_csv(f'./z_feature_json_check/v_in_circle_traj_check_{time_str}.csv')
            v_in_circle_traj.loc[:, "instant_speed"] = v_in_circle_traj['travel_distance'] / (v_in_circle_traj['dt'] * self.sim_resol)  # [m/s]
            instant_speed_tmp = v_in_circle_traj['instant_speed'].dropna().tolist()
            # v_in_circle_traj.to_csv(f'./z_feature_json_check/v_in_circle_traj_check_{time_str}.csv')
            # break
            instant_speed_list += instant_speed_tmp

        return instant_speed_list

    def yielding_distance_and_speed_analysis(self, yielding_speed_thres=2.2352):
        """The Euclidean distance and the speed of the closest vehicle in the circle with the ego-vehicle

        Parameters
        ----------
        yielding_speed_thres

        Returns
        -------

        """
        yielding_conflict_dist_and_v_dict = {"yield_dist_and_v_list": [], "not_yield_dist_and_v_list": []}  # [[dist, v], ...], unit: [m, m/s].

        yielding_conflicting_third_mapping = {'yielding_1_t': 'circle_3_t', 'yielding_2_t': 'circle_1_t',
                                                 'yielding_3_t': 'circle_2_t'}

        for vid in tqdm(self.traj_pool.vehicle_id_list(), desc='yielding results propcessing'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            v_in_yielding_area = v_traj[v_traj['yielding_area'].isin(['yielding_1_t', 'yielding_2_t', 'yielding_3_t'])]
            v_yielding_location_list = v_in_yielding_area.yielding_area.unique().tolist()

            if v_in_yielding_area.shape[0] <= 1 or len(v_yielding_location_list) != 1:  # At least 1 time step in the yielding area
                continue

            v_yielding_location = v_yielding_location_list[0]
            conflict_circle_quadrant = yielding_conflicting_third_mapping[v_yielding_location]

            # v_in_yielding_area = v_in_yielding_area.groupby('t').first().reset_index()
            
            v_in_yielding_area.loc[:, "dt"] = v_in_yielding_area["t"].diff()
            # assert (v_in_yielding_area.dt.dropna() > 0).all()
            if not (v_in_yielding_area.dt.dropna() > 0).all():
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                abnormal_data = v_in_yielding_area[v_in_yielding_area.dt <= 0]
                v_in_yielding_area.to_csv(f'./z_abnormal_data_collection/v_in_yielding_area_{time_str}.csv', index=False)
                v_in_yielding_area["t"].to_csv(f'./z_abnormal_data_collection/v_in_yielding_area_t_data_{time_str}.csv', index=False)
                abnormal_data.to_csv(f'./z_abnormal_data_collection/v_in_yielding_area_abnormal_data_{time_str}.csv', index=False)

            v_in_yielding_area.loc[:, "dx"] = v_in_yielding_area["x"].diff()
            v_in_yielding_area.loc[:, "dy"] = v_in_yielding_area["y"].diff()
            v_in_yielding_area.loc[:, "travel_distance"] = (v_in_yielding_area["dx"] ** 2 + v_in_yielding_area["dy"] ** 2) ** 0.5
            v_in_yielding_area.loc[:, 'speed'] = v_in_yielding_area['travel_distance']/(v_in_yielding_area['dt'] * self.sim_resol)

            for t in v_in_yielding_area['t'].tolist():
                ego_state_this_step = v_in_yielding_area[v_in_yielding_area['t'] == t]
                ego_speed, ego_x, ego_y = ego_state_this_step.speed.item(), ego_state_this_step.x.item(), ego_state_this_step.y.item()
                if not pd.notna(ego_speed):
                    continue

                other_v_in_conflict_quadrant = self.traj_df[(self.traj_df['t'] == t) & (self.traj_df['vid'] != vid) & (self.traj_df['region_position'] == conflict_circle_quadrant)]
                if other_v_in_conflict_quadrant.shape[0] == 0:  # No other vehicles in the conflict quadrant
                    continue
                other_v_in_conflict_quadrant['Euclidean_dist'] = ((other_v_in_conflict_quadrant['x'] - ego_x) ** 2 + (other_v_in_conflict_quadrant['y'] - ego_y) ** 2) ** 0.5
                closest_other_v = other_v_in_conflict_quadrant.loc[other_v_in_conflict_quadrant.Euclidean_dist.idxmin()]

                # Calculate closest other vehicle speed
                conflict_v_id = closest_other_v.vid
                conflict_v_prev_traj = self.traj_df[(self.traj_df['vid'] == conflict_v_id) & (self.traj_df['t'] <= t)]
                if conflict_v_prev_traj.shape[0] <= 1:
                    continue
                travel_dist = ((conflict_v_prev_traj.iloc[-1]['x'] - conflict_v_prev_traj.iloc[-2]['x']) ** 2 + (conflict_v_prev_traj.iloc[-1]['y'] - conflict_v_prev_traj.iloc[-2]['y']) ** 2) ** 0.5
                travel_time = conflict_v_prev_traj.iloc[-1]['t'] - conflict_v_prev_traj.iloc[-2]['t']
                if travel_time > 2:
                    continue
                conflict_v_speed = travel_dist / (travel_time * self.sim_resol)
                conflict_v_dist = closest_other_v.Euclidean_dist

                # Whether the subject vehicle (SV) is yielding at the current step
                yield_flag = ego_state_this_step['speed'].item() < yielding_speed_thres
                if yield_flag:
                    yielding_conflict_dist_and_v_dict['yield_dist_and_v_list'].append([conflict_v_dist, conflict_v_speed])
                else:
                    yielding_conflict_dist_and_v_dict['not_yield_dist_and_v_list'].append([conflict_v_dist, conflict_v_speed])

        return yielding_conflict_dist_and_v_dict

    def distance_analysis(self, mode='single_circle', only_in_roundabout_circle=False):
        """
        mode:
            - 'single_circle': calculate the Euclidean distance of vehicle center.
            - 'three_circle': approximate each vehicle using three circles and calculate the closest distance between vehicles' circle center.

        Calculate the Euclidean distance between any pair of vehicles
        """
        if mode not in ['single_circle', 'three_circle']:
            raise ValueError("{0} not supported for distance analysis, choose from [center_distance, three_circle].".format(mode))

        distance_list = []

        if mode == 'single_circle':
            for t_step in tqdm(self.traj_df.t.unique().tolist(), desc='single circle distance results propcessing'):
                traj_df_at_t_step = self.traj_df[self.traj_df['t'] == t_step]
                if only_in_roundabout_circle:
                    traj_df_at_t_step = traj_df_at_t_step[traj_df_at_t_step['region_position'].isin(['circle_1_t', 'circle_2_t', 'circle_3_t'])]
                pos_list = [[val[0], val[1]] for val in zip(traj_df_at_t_step.x.tolist(), traj_df_at_t_step.y.tolist())]
                distance_list_at_t_step = list(pdist(pos_list, metric='euclidean'))  # pairwise distance of all vehicles.
                distance_list += distance_list_at_t_step

        if mode == "three_circle":
            radius, center_point_distance = 1.0, 2.7  # the radius of each circle, the distance between the front and rear circles.
            for t_step in tqdm(self.traj_df.t.unique().tolist(), desc='threee circle distance results propcessing'):
                traj_df_at_t_step = self.traj_df[self.traj_df['t'] == t_step]
                if only_in_roundabout_circle:
                    traj_df_at_t_step = traj_df_at_t_step[traj_df_at_t_step['region_position'].isin(['circle_1_t', 'circle_2_t', 'circle_3_t'])]
                traj_df_at_t_step["center_circle_x"] = traj_df_at_t_step["x"]
                traj_df_at_t_step["center_circle_y"] = traj_df_at_t_step["y"]
                traj_df_at_t_step["front_circle_x"] = traj_df_at_t_step["x"] + (center_point_distance / 2) * np.cos(np.radians(traj_df_at_t_step["heading"]))
                traj_df_at_t_step["front_circle_y"] = traj_df_at_t_step["y"] + (center_point_distance / 2) * np.sin(np.radians(traj_df_at_t_step["heading"]))
                traj_df_at_t_step["rear_circle_x"] = traj_df_at_t_step["x"] - (center_point_distance / 2) * np.cos(np.radians(traj_df_at_t_step["heading"]))
                traj_df_at_t_step["rear_circle_y"] = traj_df_at_t_step["y"] - (center_point_distance / 2) * np.sin(np.radians(traj_df_at_t_step["heading"]))

                # Loop through all vehicles and calculate distance with other vehicles.
                for row_idx in range(traj_df_at_t_step.shape[0] - 1):
                    dis_v = []

                    v_info = traj_df_at_t_step.iloc[row_idx]
                    other_vs = traj_df_at_t_step.iloc[row_idx+1:]

                    for ego_v_circle_pos in ['center_circle', 'front_circle', 'rear_circle']:
                        x_name, y_name = '_'.join([ego_v_circle_pos, 'x']), '_'.join([ego_v_circle_pos, 'y'])
                        v_pos = np.array([[v_info[x_name], v_info[y_name]]])
                        for other_v_circle_pos in ['center_circle', 'front_circle', 'rear_circle']:
                            x_name, y_name = '_'.join([other_v_circle_pos, 'x']), '_'.join([other_v_circle_pos, 'y'])
                            pos_array = np.array([[val[0], val[1]] for val in zip(other_vs[x_name].tolist(), other_vs[y_name].tolist())])
                            if pos_array.shape[0] == 0:
                                continue
                            dis = np.linalg.norm(v_pos - pos_array, axis=1)
                            dis_v.append(list(dis))

                    dis_v = np.array(dis_v)
                    distance_the_vehicle = list(dis_v.min(axis=0))  # The minimum distance with each other vehicle
                    distance_list += distance_the_vehicle

        return distance_list

    def PET_analysis(self):

        # occupancy_res = [ndarray, ndarray,...] each ndarray in it is the occupancy results of each position at the time step i.
        # each ndarray is height_n * width_n, where each cell is the vehicle id that occupied the position at that time step.
        occupancy_res = []
        for t in tqdm(range(len(self.TIME_BUFF)), desc='PET occupancy processing'):
            occupancy_at_t_step = np.zeros((self.PET_configs['height_n'], self.PET_configs['width_n']), dtype=object)
            occupancy_at_t_step[:, :] = 'empty'  # initialize the occupancy ndarray where no vehicles any position.

            vehicle_list_at_t_step = self.TIME_BUFF[t]
            for v in vehicle_list_at_t_step:
                # Check all 5 points of the vehicle (center, each corner point of the bounding box)
                v_center_pos = [v.location.x, v.location.y]
                v_pt1_pos, v_pt2_pos, v_pt3_pos, v_pt4_pos = v.realworld_4_vertices
                v_all_pos = [list(v_center_pos), list(v_pt1_pos), list(v_pt2_pos), list(v_pt3_pos), list(v_pt4_pos)]
                v_all_pos = self.road_matcher._world2pxl(v_all_pos)  # Transform to pxl coordinates
                for pos in v_all_pos:
                    # Check whether the position is within the grid area
                    if not ((pos[1] >= self.PET_configs['height_end']) or (pos[1] <= self.PET_configs['height_start']) \
                            or (pos[0] <= self.PET_configs['width_start']) or (pos[0] >= self.PET_configs['width_end'])):
                        height_idx, width_idx = int(divmod((pos[1] - self.PET_configs['height_start']), self.PET_configs['height_res'])[0]), \
                                                int(divmod((pos[0] - self.PET_configs['width_start']), self.PET_configs['width_res'])[0])

                        if occupancy_at_t_step[height_idx, width_idx] == 'empty':
                            occupancy_at_t_step[height_idx, width_idx] = v.id
                        else:
                            # If there are multiple vehicles occupying a same location (the grid might be too sparse), randomly determine which vehicle is occupying.
                            if np.random.uniform() > 0.5:
                                occupancy_at_t_step[height_idx, width_idx] = v.id
            occupancy_at_t_step[:self.PET_configs['left_top_height_n'],:self.PET_configs['left_top_width_n']] = 'empty'
            occupancy_at_t_step[:self.PET_configs['right_top_height_n'],- self.PET_configs['right_top_width_n']:] = 'empty'
            occupancy_res.append(occupancy_at_t_step)
        occupancy_res = np.stack(occupancy_res)  # time * height_n * width_n

        # Loop over all positions to calculate the PET on it along the time
        PET_list = []
        for height_idx in tqdm(range(occupancy_res.shape[1]), desc='PET results processing'):
            for width_idx in range(occupancy_res.shape[2]):
                occupancy_at_this_position = occupancy_res[:, height_idx, width_idx]
                PET_at_this_position = []
                last_occupied_t, last_occupied_vid = None, 'empty'
                for t_idx in range(occupancy_res.shape[0]):
                    current_occupied_t, current_occupied_vid = t_idx, occupancy_at_this_position[t_idx]
                    if current_occupied_vid == 'empty':  # No vehicle on this position at this moment, nothing needs to do.
                        continue
                    if last_occupied_vid != 'empty' and last_occupied_vid != current_occupied_vid:  # There is a new vehicle occupy this position.
                        PET = (current_occupied_t - last_occupied_t) * self.sim_resol  # The PET
                        PET_at_this_position.append(PET)
                        last_occupied_t, last_occupied_vid = current_occupied_t, current_occupied_vid
                    if last_occupied_vid == 'empty' or last_occupied_vid == current_occupied_vid:  # There is a vehicle first occupy this position.
                        last_occupied_t, last_occupied_vid = current_occupied_t, current_occupied_vid
                PET_list += PET_at_this_position

        return PET_list

