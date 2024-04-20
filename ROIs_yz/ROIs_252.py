import cv2
import os


class ROIMatcher_252(object):
    """
    This class includes different ROI maps for rcu_252, each point can be checked whether
    within the interested ROI.
    """

    def __init__(self, drivable_map_dir=None, sim_remove_vehicle_area_map_dir=None, circle_map_dir=None, entrance_map_dir=None,
                 exit_map_dir=None, crosswalk_map_dir=None, yielding_area_map_dir=None, at_circle_lane_map_dir=None,
                 map_height=1024, map_width=1024):
        self.drivable_map = None
        self.sim_remove_vehicle_area_map = None

        if drivable_map_dir is not None:
            self.drivable_map = cv2.imread(drivable_map_dir, cv2.IMREAD_GRAYSCALE)
            self.drivable_map = cv2.resize(self.drivable_map, (map_width, map_height))

        if sim_remove_vehicle_area_map_dir is not None:
            self.sim_remove_vehicle_area_map = cv2.imread(sim_remove_vehicle_area_map_dir, cv2.IMREAD_GRAYSCALE)
            self.sim_remove_vehicle_area_map = cv2.resize(self.sim_remove_vehicle_area_map, (map_width, map_height))

        if circle_map_dir is not None:
            self.circle_1_t_map = cv2.imread(os.path.join(circle_map_dir, 'circle_1_t-map.png'), cv2.IMREAD_GRAYSCALE)
            self.circle_2_t_map = cv2.imread(os.path.join(circle_map_dir, 'circle_2_t-map.png'), cv2.IMREAD_GRAYSCALE)
            self.circle_3_t_map = cv2.imread(os.path.join(circle_map_dir, 'circle_3_t-map.png'), cv2.IMREAD_GRAYSCALE)

            self.circle_1_t_map = cv2.resize(self.circle_1_t_map, (map_width, map_height))
            self.circle_2_t_map = cv2.resize(self.circle_2_t_map, (map_width, map_height))
            self.circle_3_t_map = cv2.resize(self.circle_3_t_map, (map_width, map_height))

        if entrance_map_dir is not None:
            # 1_t entrance
            self.entrance_1_t_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_1_t_1-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_1_t_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_1_t_2-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_1_t_3 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_1_t_3-map.png'), cv2.IMREAD_GRAYSCALE)

            self.entrance_1_t_1 = cv2.resize(self.entrance_1_t_1, (map_width, map_height))
            self.entrance_1_t_2 = cv2.resize(self.entrance_1_t_2, (map_width, map_height))
            self.entrance_1_t_3 = cv2.resize(self.entrance_1_t_3, (map_width, map_height))

            # 2_t entrance
            self.entrance_2_t_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_2_t_1-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_2_t_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_2_t_2-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_2_t_3 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_2_t_3-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_2_t_rightturn = cv2.imread(os.path.join(entrance_map_dir, 'entrance_2_t_rightturn-map.png'), cv2.IMREAD_GRAYSCALE)

            self.entrance_2_t_1 = cv2.resize(self.entrance_2_t_1, (map_width, map_height))
            self.entrance_2_t_2 = cv2.resize(self.entrance_2_t_2, (map_width, map_height))
            self.entrance_2_t_3 = cv2.resize(self.entrance_2_t_3, (map_width, map_height))
            self.entrance_2_t_rightturn = cv2.resize(self.entrance_2_t_rightturn, (map_width, map_height))

            # 3_t entrance
            self.entrance_3_t_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_3_t_1-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_3_t_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_3_t_2-map.png'), cv2.IMREAD_GRAYSCALE)
            self.entrance_3_t_3 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_3_t_3-map.png'), cv2.IMREAD_GRAYSCALE)

            self.entrance_3_t_1 = cv2.resize(self.entrance_3_t_1, (map_width, map_height))
            self.entrance_3_t_2 = cv2.resize(self.entrance_3_t_2, (map_width, map_height))
            self.entrance_3_t_3 = cv2.resize(self.entrance_3_t_3, (map_width, map_height))

        if exit_map_dir is not None:
            # 1_t exit
            self.exit_1_t_1 = cv2.imread(os.path.join(exit_map_dir, 'exit_1_t_1-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_1_t_2 = cv2.imread(os.path.join(exit_map_dir, 'exit_1_t_2-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_1_t_3 = cv2.imread(os.path.join(exit_map_dir, 'exit_1_t_3-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_1_t_rightturn = cv2.imread(os.path.join(exit_map_dir, 'exit_1_t_rightturn-map.png'), cv2.IMREAD_GRAYSCALE)

            self.exit_1_t_1 = cv2.resize(self.exit_1_t_1, (map_width, map_height))
            self.exit_1_t_2 = cv2.resize(self.exit_1_t_2, (map_width, map_height))
            self.exit_1_t_3 = cv2.resize(self.exit_1_t_3, (map_width, map_height))
            self.exit_1_t_rightturn = cv2.resize(self.exit_1_t_rightturn, (map_width, map_height))

            # 2_t exit
            self.exit_2_t_1 = cv2.imread(os.path.join(exit_map_dir, 'exit_2_t_1-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_2_t_2 = cv2.imread(os.path.join(exit_map_dir, 'exit_2_t_2-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_2_t_3 = cv2.imread(os.path.join(exit_map_dir, 'exit_2_t_3-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_2_t_4 = cv2.imread(os.path.join(exit_map_dir, 'exit_2_t_4-map.png'), cv2.IMREAD_GRAYSCALE)

            self.exit_2_t_1 = cv2.resize(self.exit_2_t_1, (map_width, map_height))
            self.exit_2_t_2 = cv2.resize(self.exit_2_t_2, (map_width, map_height))
            self.exit_2_t_3 = cv2.resize(self.exit_2_t_3, (map_width, map_height))
            self.exit_2_t_4 = cv2.resize(self.exit_2_t_4, (map_width, map_height))

            # 3_t exit
            self.exit_3_t_1 = cv2.imread(os.path.join(exit_map_dir, 'exit_3_t_1-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_3_t_2 = cv2.imread(os.path.join(exit_map_dir, 'exit_3_t_2-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_3_t_3 = cv2.imread(os.path.join(exit_map_dir, 'exit_3_t_3-map.png'), cv2.IMREAD_GRAYSCALE)
            self.exit_3_t_4 = cv2.imread(os.path.join(exit_map_dir, 'exit_3_t_4-map.png'), cv2.IMREAD_GRAYSCALE)


            self.exit_3_t_1 = cv2.resize(self.exit_3_t_1, (map_width, map_height))
            self.exit_3_t_2 = cv2.resize(self.exit_3_t_2, (map_width, map_height))
            self.exit_3_t_3 = cv2.resize(self.exit_3_t_3, (map_width, map_height))
            self.exit_3_t_4 = cv2.resize(self.exit_3_t_4, (map_width, map_height))


        if crosswalk_map_dir is not None:
            self.crosswalk = cv2.imread(os.path.join(crosswalk_map_dir, 'crosswalk-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.crosswalk = cv2.resize(self.crosswalk, (map_width, map_height))

        if yielding_area_map_dir is not None:
            self.yielding_1_t = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_1_t-map.png'), cv2.IMREAD_GRAYSCALE)
            self.yielding_1_t = cv2.resize(self.yielding_1_t, (map_width, map_height))
            self.yielding_2_t = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_2_t-map.png'), cv2.IMREAD_GRAYSCALE)
            self.yielding_2_t = cv2.resize(self.yielding_2_t, (map_width, map_height))
            self.yielding_3_t = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_3_t-map.png'), cv2.IMREAD_GRAYSCALE)
            self.yielding_3_t = cv2.resize(self.yielding_3_t, (map_width, map_height))

        if at_circle_lane_map_dir is not None:
            self.circle_inner_lane = cv2.imread(os.path.join(at_circle_lane_map_dir, 'circle_inner_lane-map.png'), cv2.IMREAD_GRAYSCALE)
            self.circle_inner_lane = cv2.resize(self.circle_inner_lane, (map_width, map_height))
            self.circle_middle_lane = cv2.imread(os.path.join(at_circle_lane_map_dir, 'circle_middle_lane-map.png'), cv2.IMREAD_GRAYSCALE)
            self.circle_middle_lane = cv2.resize(self.circle_inner_lane, (map_width, map_height))
            self.circle_outer_lane = cv2.imread(os.path.join(at_circle_lane_map_dir, 'circle_outer_lane-map.png'), cv2.IMREAD_GRAYSCALE)
            self.circle_outer_lane = cv2.resize(self.circle_outer_lane, (map_width, map_height))

    def region_position_matching(self, pxl_pt):
        region_position = 'offroad'
        y0, x0 = pxl_pt[0], pxl_pt[1]

        # circle in different quadrant
        if self.circle_1_t_map[x0, y0] > 128.:
            region_position = 'circle_1_t'
            return region_position
        if self.circle_2_t_map[x0, y0] > 128.:
            region_position = 'circle_2_t'
            return region_position
        if self.circle_3_t_map[x0, y0] > 128.:
            region_position = 'circle_3_t'
            return region_position

        # Entrance. facing circle, left lane is 1, right lane is 2.
        # 1_t entrance
        if self.entrance_1_t_1[x0, y0] > 128.:
            region_position = 'entrance_1_t_1'
            return region_position
        if self.entrance_1_t_2[x0, y0] > 128.:
            region_position = 'entrance_1_t_2'
            return region_position
        if self.entrance_1_t_3[x0, y0] > 128.:
            region_position = 'entrance_1_t_3'
            return region_position
        # 2_t entrance
        if self.entrance_2_t_1[x0, y0] > 128.:
            region_position = 'entrance_2_t_1'
            return region_position
        if self.entrance_2_t_2[x0, y0] > 128.:
            region_position = 'entrance_2_t_2'
            return region_position
        if self.entrance_2_t_3[x0, y0] > 128.:
            region_position = 'entrance_2_t_3'
            return region_position
        if self.entrance_2_t_rightturn[x0, y0] > 128.:
            region_position = 'entrance_2_t_rightturn'
            return region_position
        # 3_t entrance
        if self.entrance_3_t_1[x0, y0] > 128.:
            region_position = 'entrance_3_t_1'
            return region_position
        if self.entrance_3_t_2[x0, y0] > 128.:
            region_position = 'entrance_3_t_2'
            return region_position
        if self.entrance_3_t_3[x0, y0] > 128.:
            region_position = 'entrance_3_t_3'
            return region_position

        # Exit
        if self.exit_1_t_1[x0, y0] > 128. or self.exit_1_t_2[x0, y0] > 128. or self.exit_1_t_3[x0, y0] > 128.:
            region_position = 'exit_1_t'
            return region_position
        if self.exit_1_t_rightturn is not None:
            if self.exit_1_t_rightturn[x0, y0] > 128.:
                region_position = 'exit_1_t_rightturn'
                return region_position
        if self.exit_2_t_1[x0, y0] > 128. or self.exit_2_t_2[x0, y0] > 128. or self.exit_2_t_3[x0, y0] > 128. or self.exit_2_t_4[x0, y0] > 128.:
            region_position = 'exit_2_t'
            return region_position
        if self.exit_3_t_1[x0, y0] > 128. or self.exit_3_t_2[x0, y0] > 128. or self.exit_3_t_3[x0, y0] > 128. or self.exit_3_t_4[x0, y0] > 128.:
            region_position = 'exit_3_t'
            return region_position
        

        # crosswalk
        # if self.crosswalk[y0, x0] > 128.:
        #     region_position = 'crosswalk'
        #     return region_position

        return region_position

    def yielding_area_matching(self, pxl_pt):
        yielding_area = 'Not_in_yielding_area'
        y0, x0 = pxl_pt[0], pxl_pt[1]
        if self.yielding_1_t[x0, y0] > 128.:
            yielding_area = 'yielding_1_t'
        if self.yielding_2_t[x0, y0] > 128.:
            yielding_area = 'yielding_2_t'
        if self.yielding_3_t[x0, y0] > 128.:
            yielding_area = 'yielding_3_t'
        return yielding_area

    def at_circle_lane_matching(self, pxl_pt):
        at_circle_lane = 'Not_in_circle'
        y0, x0 = pxl_pt[0], pxl_pt[1]
        if self.circle_inner_lane[x0, y0] > 128.:
            at_circle_lane = 'inner'
        if self.circle_middle_lane[x0, y0] > 128.:
            at_circle_lane = 'middle'
        if self.circle_outer_lane[x0, y0] > 128.:
            at_circle_lane = 'outer'
        return at_circle_lane

    def entrance_lane_matching(self, pxl_pt):
        """
        This function is to find whether a give position is in certain entrance lane.
        Used for safty check when initializing vehicles.
        """
        # Entrance. facing circle, left lane is 1, right lane is 2.
        # north entrance
        region_position = 'not_at_entrance'
        y0, x0 = pxl_pt[0], pxl_pt[1]

        # 1_t entrance
        if self.entrance_1_t_1[x0, y0] > 128.:
            region_position = 'entrance_1_t_1'
            return region_position
        if self.entrance_1_t_2[x0, y0] > 128.:
            region_position = 'entrance_1_t_2'
            return region_position
        if self.entrance_1_t_3[x0, y0] > 128.:
            region_position = 'entrance_1_t_3'
            return region_position
        # 2_t entrance
        if self.entrance_2_t_1[x0, y0] > 128.:
            region_position = 'entrance_2_t_1' # initial_2_in2
            return region_position
        if self.entrance_2_t_2[x0, y0] > 128.:
            region_position = 'entrance_2_t_2' # initial_2_in3
            return region_position
        if self.entrance_2_t_3[x0, y0] > 128.:
            region_position = 'entrance_2_t_3' # initial_2_in4
            return region_position
        if self.entrance_2_t_rightturn[x0, y0] > 128.:
            region_position = 'entrance_2_t_rightturn' # initial_2_in1
            return region_position
        # 3_t entrance
        if self.entrance_3_t_1[x0, y0] > 128.:
            region_position = 'entrance_3_t_1'
            return region_position
        if self.entrance_3_t_2[x0, y0] > 128.:
            region_position = 'entrance_3_t_2'
            return region_position
        if self.entrance_3_t_3[x0, y0] > 128.:
            region_position = 'entrance_3_t_3'
            return region_position

        return region_position
