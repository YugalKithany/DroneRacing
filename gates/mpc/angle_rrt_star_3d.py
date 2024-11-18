import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import splprep, splev

show_animation_2d = True
show_animation_3d = True
debug_visualization = True
plot_waypoints = True
waypoint_dist = 1

class RRTStar:
    class Node:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None
            self.cost = 0.0
            self.children = []

    def __init__(self, start, goal, obstacle_list, gate_list, rand_area,
                 expand_dis=0.5, path_resolution=0.05, goal_sample_rate=1, max_iter=1000,
                 connect_circle_dist=8.335):
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.gate_list = gate_list
        self.node_list = []
        self.connect_circle_dist = connect_circle_dist
        self.animation_objects = []
        self.current_gate_index = -1
        self.next_gate_index = 0
        self.drone_size = [0.35, 0.35, 0.15]

        self.fig = None
        self.ax1 = None
        self.ax2 = None

        self.compute_sampling_area()

    def get_gate_entry_exit_points(self, gate):
        x, y, z, size, roll, pitch, yaw, entry_direction = gate
        entry_offset = 0.25
        
        # Convert yaw to radians
        yaw_rad = math.radians(yaw)
        
        # Calculate entry and exit points based on yaw
        if entry_direction == '+':
            entry_point = (
                x - entry_offset * math.cos(yaw_rad),
                y - entry_offset * math.sin(yaw_rad),
                z
            )
            exit_point = (
                x + entry_offset * math.cos(yaw_rad),
                y + entry_offset * math.sin(yaw_rad),
                z
            )
        else:
            entry_point = (
                x + entry_offset * math.cos(yaw_rad),
                y + entry_offset * math.sin(yaw_rad),
                z
            )
            exit_point = (
                x - entry_offset * math.cos(yaw_rad),
                y - entry_offset * math.sin(yaw_rad),
                z
            )
        
        return entry_point, exit_point

    def compute_sampling_area(self):
        gate_x = [gate[0] for gate in self.gate_list]
        gate_y = [gate[1] for gate in self.gate_list]

        all_x = [self.start.x, self.end.x] + gate_x
        all_y = [self.start.y, self.end.y] + gate_y

        margin = 0.335
        self.sampling_area = {
            'min_x': min(all_x) - margin,
            'max_x': max(all_x) + margin,
            'min_y': min(all_y) - margin,
            'max_y': max(all_y) + margin
        }

    def planning(self, animation=True):
        self.node_list = [self.start]

        if animation and self.fig is None:
            self.setup_animation()

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if new_node and self.check_collision(new_node, self.obstacle_list) and self.check_gate_collision(new_node):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.update_animation()

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y, self.node_list[-1].z) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if final_node and self.check_collision(final_node, self.obstacle_list) and self.check_gate_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None
    
    def plan_next_segment(self, start_point, end_point):
        self.start = self.Node(*start_point)
        self.end = self.Node(*end_point)
        self.compute_sampling_area()

        max_attempts = 5
        path = None
        for attempt in range(max_attempts):
            path = self.planning(animation=False)
            if path is not None:
                break
            print(f"Attempt {attempt + 1} failed, retrying...")

        if path is None:
            print(f"Cannot find path from {start_point} to {end_point} after {max_attempts} attempts")
            return None

        return path[::-1]

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d, theta, phi = self.calc_distance_and_angles(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_z = [new_node.z]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        max_z_change = 0.1  # Maximum allowed z change per step

        for _ in range(n_expand):
            predict_x = not (self.start.x == self.end.x == new_node.x)
            predict_y = not (self.start.y == self.end.y == new_node.y)
            predict_z = not (self.start.z == self.end.z == new_node.z)

            new_x = new_node.x + self.path_resolution * math.sin(phi) * math.cos(theta) if predict_x else new_node.x
            new_y = new_node.y + self.path_resolution * math.sin(phi) * math.sin(theta) if predict_y else new_node.y
            
            # Calculate the desired z change
            desired_z_change = self.path_resolution * math.cos(phi) if predict_z else 0
            
            # Limit the z change to the maximum allowed
            z_change = max(min(desired_z_change, max_z_change), -max_z_change)
            new_z = new_node.z + z_change

            # Check for gate collision before adding the new point
            temp_node = self.Node(new_x, new_y, new_z)
            temp_node.parent = new_node  # Set parent for direction check
            if not self.check_gate_collision(temp_node):
                return None

            new_node.x = new_x
            new_node.y = new_y
            new_node.z = new_z
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_z.append(new_node.z)

            if not self.check_collision(new_node, self.obstacle_list):
                return None

        d, _, _ = self.calc_distance_and_angles(new_node, to_node)
        if d <= self.path_resolution:
            temp_node = self.Node(to_node.x, to_node.y, to_node.z)
            temp_node.parent = new_node  # Set parent for direction check
            if self.check_gate_collision(temp_node):
                new_node.path_x.append(to_node.x)
                new_node.path_y.append(to_node.y)
                new_node.path_z.append(to_node.z)
                new_node.x = to_node.x
                new_node.y = to_node.y
                new_node.z = to_node.z

        new_node.parent = from_node
        new_node.cost = from_node.cost + d

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])
        return path

    def smooth_path(self, path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
        if len(path) < 2:
            return path

        newpath = [[p[0], p[1], p[2]] for p in path]
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(3):  # x, y, z
                    aux = newpath[i][j]
                    newpath[i][j] += weight_data * (path[i][j] - newpath[i][j])
                    newpath[i][j] += weight_smooth * (newpath[i - 1][j] + newpath[i + 1][j] - 2.0 * newpath[i][j])
                    change += abs(aux - newpath[i][j])

        return newpath

    def optimize_path_for_racing(self, path):
        # Spline interpolation
        tck, u = splprep([np.array(path)[:, 0], np.array(path)[:, 1], np.array(path)[:, 2]], s=0)
        u_new = np.linspace(0, 1, num=len(path) * 5)
        smooth_path = np.column_stack(splev(u_new, tck))

        # Optimize path
        optimized_path = []
        for i in range(len(smooth_path)):
            point = smooth_path[i]
            if i > 0 and i < len(smooth_path) - 1:
                prev_point = smooth_path[i - 1]
                next_point = smooth_path[i + 1]
                direction = (next_point - prev_point) / np.linalg.norm(next_point - prev_point)
                optimized_point = point + direction * 0.067
                node = self.Node(*optimized_point)
                node.path_x = [optimized_point[0]]
                node.path_y = [optimized_point[1]]
                node.path_z = [optimized_point[2]]
                if self.check_collision(node, self.obstacle_list) and self.check_gate_collision(node):
                    optimized_path.append(optimized_point)
                else:
                    optimized_path.append(point)
            else:
                optimized_path.append(point)

        return optimized_path

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.end.x
        dy = y - self.end.y
        predict_z = not (self.start.z == self.end.z == z)
        dz = z - self.end.z if predict_z else 0
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            predict_z = self.start.z != self.end.z
            rnd = self.Node(
                random.uniform(self.sampling_area['min_x'], self.sampling_area['max_x']),
                random.uniform(self.sampling_area['min_y'], self.sampling_area['max_y']),
                random.uniform(self.min_rand, self.max_rand) if predict_z else self.start.z
            )
        else:
            rnd = self.Node(self.end.x, self.end.y, self.end.z)
        return rnd

    def setup_animation(self):
        if self.fig is not None:
            return

        plt.ion()
        if show_animation_2d and show_animation_3d:
            self.fig = plt.figure(figsize=(12, 5))
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122, projection='3d')
        elif show_animation_2d:
            self.fig, self.ax1 = plt.subplots(figsize=(6, 5))
            self.ax2 = None
        elif show_animation_3d:
            self.fig = plt.figure(figsize=(6, 5))
            self.ax2 = self.fig.add_subplot(111, projection='3d')
            self.ax1 = None

        self.setup_plot_2d()
        self.setup_plot_3d()

    def setup_plot_2d(self):
        if not show_animation_2d:
            return
        self.ax1.clear()
        self.ax1.set_xlim(self.sampling_area['min_x'] - 0.33, self.sampling_area['max_x'] + 0.33)
        self.ax1.set_ylim(self.sampling_area['min_y'] - 0.33, self.sampling_area['max_y'] + 0.33)
        self.ax1.grid(True)
        self.ax1.set_title('2D View (XY Plane)')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        for gate in self.gate_list:
            self.plot_gate_2d(self.ax1, gate)
        self.ax1.plot(self.start.x, self.start.y, "xr")
        self.ax1.plot(self.end.x, self.end.y, "xr")
        self.path_2d, = self.ax1.plot([], [], '-g')
        self.animation_objects.append(self.path_2d)
        self.plot_obstacles_2d(self.ax1)
        self.ax1.set_aspect('equal', 'box')

    def setup_plot_3d(self):
        if not show_animation_3d:
            return
        self.ax2.clear()
        self.ax2.set_xlim(self.sampling_area['min_x'] - 0.33, self.sampling_area['max_x'] + 0.33)
        self.ax2.set_ylim(self.sampling_area['min_y'] - 0.33, self.sampling_area['max_y'] + 0.33)
        self.ax2.set_zlim(0, 1.67)
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        self.ax2.set_title('3D View')
        for gate in self.gate_list:
            self.plot_gate_3d(self.ax2, gate)
        self.ax2.scatter(self.start.x, self.start.y, self.start.z, marker='x', color='r', s=100, label='Start')
        self.ax2.scatter(self.end.x, self.end.y, self.end.z, marker='x', color='r', s=100, label='Goal')
        self.path_3d, = self.ax2.plot([], [], [], '-g')
        self.animation_objects.append(self.path_3d)
        self.plot_obstacles_3d(self.ax2)

    def update_animation(self):
        x_list = [node.x for node in self.node_list]
        y_list = [node.y for node in self.node_list]
        z_list = [node.z for node in self.node_list]

        if show_animation_2d:
            self.path_2d.set_data(x_list, y_list)

        if show_animation_3d:
            self.path_3d.set_data(x_list, y_list)
            self.path_3d.set_3d_properties(z_list)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_gate_2d(self, ax, gate):
        x, y, z, size, roll, pitch, yaw, entry_direction = gate
        
        yaw_rad = math.radians(yaw)
        
        half_size = size / 2
        line_start = np.array([
            x - half_size * math.sin(yaw_rad),
            y + half_size * math.cos(yaw_rad)
        ])
        line_end = np.array([
            x + half_size * math.sin(yaw_rad),
            y - half_size * math.cos(yaw_rad)
        ])
        
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r-', linewidth=2)
        
        if entry_direction == '+':
            arrow_start = np.array([
                x - 0.25 * math.cos(yaw_rad),
                y - 0.25 * math.sin(yaw_rad)
            ])
            arrow_dir = np.array([
                0.2 * math.cos(yaw_rad),
                0.2 * math.sin(yaw_rad)
            ])
        else:
            arrow_start = np.array([
                x + 0.25 * math.cos(yaw_rad),
                y + 0.25 * math.sin(yaw_rad)
            ])
            arrow_dir = np.array([
                -0.2 * math.cos(yaw_rad),
                -0.2 * math.sin(yaw_rad)
            ])
        
        ax.arrow(arrow_start[0], arrow_start[1],
                arrow_dir[0], arrow_dir[1],
                head_width=0.05, head_length=0.05,
                fc='r', ec='r', alpha=0.7)
        
        num_arrows = 3
        arrow_spacing = size / (num_arrows + 1)
        
        perp_dir = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)])
        
        for i in range(num_arrows):
            offset = (i - (num_arrows-1)/2) * arrow_spacing
            start_point = arrow_start + perp_dir * offset
            
            ax.arrow(start_point[0], start_point[1],
                    arrow_dir[0] * 0.5, arrow_dir[1] * 0.5,
                    head_width=0.03, head_length=0.03,
                    fc='g', ec='g', alpha=0.5)

    def plot_gate_3d(self, ax, gate):
        x, y, z, size, roll, pitch, yaw, entry_direction = gate
        center = np.array([x, y, z])
        R = rotation_matrix(roll, pitch, yaw)
        
        theta = np.linspace(0, 2*np.pi, 100)
        circle_points = np.array([
            np.zeros_like(theta),
            size/2 * np.cos(theta),
            size/2 * np.sin(theta)
        ])
        
        circle_points_world = np.zeros_like(circle_points)
        for i in range(circle_points.shape[1]):
            local_point = circle_points[:, i]
            rotated_point = R @ local_point
            circle_points_world[:, i] = rotated_point + center
        
        ax.plot(circle_points_world[0, :], 
                circle_points_world[1, :], 
                circle_points_world[2, :], 
                'r-', linewidth=2)
        
        arrow_length = size * 0.3
        if entry_direction == '+':
            arrow_start_local = np.array([-arrow_length, 0, 0])
            arrow_dir_local = np.array([arrow_length*2, 0, 0])
        else:
            arrow_start_local = np.array([arrow_length, 0, 0])
            arrow_dir_local = np.array([-arrow_length*2, 0, 0])
        
        arrow_start_world = R @ arrow_start_local + center
        arrow_dir_world = R @ arrow_dir_local
        
        ax.quiver(arrow_start_world[0], 
                arrow_start_world[1], 
                arrow_start_world[2],
                arrow_dir_world[0], 
                arrow_dir_world[1], 
                arrow_dir_world[2],
                color='r', 
                length=1.0, 
                arrow_length_ratio=0.3)

    def plot_obstacles_2d(self, ax):
        for (ox, oy, oz, radius, height) in self.obstacle_list:
            circle = Circle((ox, oy), radius, fill=True, color='r', alpha=0.3)
            ax.add_artist(circle)

    def plot_obstacles_3d(self, ax):
        for (ox, oy, oz, radius, height) in self.obstacle_list:
            z = np.linspace(oz, oz + height, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + ox
            y_grid = radius * np.sin(theta_grid) + oy
            ax.plot_surface(x_grid, y_grid, z_grid, color='r', alpha=0.3)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 + (node.z - rnd_node.z) ** 2
                for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angles(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        theta = math.atan2(dy, dx)
        phi = math.atan2(math.sqrt(dx ** 2 + dy ** 2), dz)
        return d, theta, phi

    def check_collision(self, node, obstacleList):
        if node is None or not node.path_x:
            return False
        
        for (ox, oy, oz, radius, height) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            dz_list = [oz - z for z in node.path_z]
            d_list = [math.sqrt(dx ** 2 + dy ** 2) for (dx, dy) in zip(dx_list, dy_list)]

            if any(d <= radius + max(self.drone_size[:2]) / 2 and oz <= z <= oz + height 
                for d, z in zip(d_list, node.path_z)):
                return False

        # Check collision with gates
        for gate in self.gate_list:
            x, y, z, size, roll, pitch, yaw, entry_direction = gate
            center = np.array([x, y, z])
            R = rotation_matrix(roll, pitch, yaw)
            R_inv = R.T
            ring_thickness = 0.1  # 0.05?
            
            for x_node, y_node, z_node in zip(node.path_x, node.path_y, node.path_z):
                point = np.array([x_node, y_node, z_node])
                point_local = R_inv @ (point - center)
                
                if abs(point_local[0]) <= ring_thickness:
                    distance = math.sqrt(point_local[1]**2 + point_local[2]**2)
                    inner_radius = size/2 - ring_thickness
                    outer_radius = size/2 + ring_thickness
                    if inner_radius <= distance <= outer_radius:
                        return False

        return True

    def check_gate_collision(self, node):
        if node.parent is None:
            return True

        for gate in self.gate_list:
            x, y, z, size, roll, pitch, yaw, entry_direction = gate
            center = np.array([x, y, z])
            R = rotation_matrix(roll, pitch, yaw)
            R_inv = R.T
            
            p0 = np.array([node.parent.x, node.parent.y, node.parent.z])
            p1 = np.array([node.x, node.y, node.z])
            
            p0_local = R_inv @ (p0 - center)
            p1_local = R_inv @ (p1 - center)
            
            if p0_local[0] * p1_local[0] <= 0:
                t = -p0_local[0] / (p1_local[0] - p0_local[0]) if p1_local[0] != p0_local[0] else 0
                intersection = p0_local + t * (p1_local - p0_local)
                
                if math.sqrt(intersection[1]**2 + intersection[2]**2) <= size/2 + 0.1:
                    if entry_direction == '+' and p1_local[0] < p0_local[0]:
                        return False
                    if entry_direction == '-' and p1_local[0] > p0_local[0]:
                        return False

        return True
    
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.expand_dis * 5)
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 + (node.z - new_node.z) ** 2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list) and self.check_gate_collision(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)

        if min_cost == float("inf"):
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def calc_new_cost(self, from_node, to_node):
        d, _, _ = self.calc_distance_and_angles(from_node, to_node)
        return from_node.cost + d

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list) and self.check_gate_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                if near_node.parent:
                    if near_node in near_node.parent.children:
                        near_node.parent.children.remove(near_node)
                near_node.parent = new_node
                near_node.cost = edge_node.cost
                new_node.children.append(near_node)
                self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, parent_node):
        for child in parent_node.children:
            child.cost = self.calc_new_cost(parent_node, child)
            self.propagate_cost_to_leaves(child)

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def print_scenario_details(scenario, gate_list, gate_order, start, end, obstacle_list, debug_visualization=False):
    print(f"\nScenario: {scenario}")
    print("Gate List:")
    for i, gate in enumerate(gate_list):
        print(f"  Gate {i+1}: {gate}")
    print("Gate Order:", [str(g) if isinstance(g, int) else g for g in gate_order])
    print(f"Start: {start}")
    print(f"End: {end}")
    print("Obstacles:")
    for obstacle in obstacle_list:
        print(f"  {obstacle}")

    if debug_visualization:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True)

        for i, gate in enumerate(gate_list):
            x, y, z, size, roll, pitch, yaw, entry_direction = gate
            
            yaw_rad = math.radians(yaw)
            
            half_size = size / 2
            line_start = np.array([
                x - half_size * math.sin(yaw_rad),
                y + half_size * math.cos(yaw_rad)
            ])
            line_end = np.array([
                x + half_size * math.sin(yaw_rad),
                y - half_size * math.cos(yaw_rad)
            ])
            
            ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r-', linewidth=2)
            
            if entry_direction == '+':
                arrow_start = np.array([
                    x - 0.25 * math.cos(yaw_rad),
                    y - 0.25 * math.sin(yaw_rad)
                ])
                arrow_dir = np.array([
                    0.2 * math.cos(yaw_rad),
                    0.2 * math.sin(yaw_rad)
                ])
            else:
                arrow_start = np.array([
                    x + 0.25 * math.cos(yaw_rad),
                    y + 0.25 * math.sin(yaw_rad)
                ])
                arrow_dir = np.array([
                    -0.2 * math.cos(yaw_rad),
                    -0.2 * math.sin(yaw_rad)
                ])
            
            ax.arrow(arrow_start[0], arrow_start[1],
                    arrow_dir[0], arrow_dir[1],
                    head_width=0.05, head_length=0.05,
                    fc='r', ec='r', alpha=0.7)
            
            ax.text(x, y, f'{i+1}', fontsize=12, ha='center', va='center')

        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(end[0], end[1], 'bo', markersize=10, label='End')

        for obstacle in obstacle_list:
            ox, oy, oz, radius, height = obstacle
            circle = Circle((ox, oy), radius, fill=True, color='gray', alpha=0.5)
            ax.add_artist(circle)
            ax.text(ox, oy, f'H:{height}', fontsize=10, ha='center', va='center')

        for i in range(len(gate_order) - 1):
            if gate_order[i] == 'START':
                x1, y1 = start[:2]
            elif gate_order[i] == 'END':
                x1, y1 = end[:2]
            else:
                x1, y1 = gate_list[gate_order[i]][:2]

            if gate_order[i+1] == 'START':
                x2, y2 = start[:2]
            elif gate_order[i+1] == 'END':
                x2, y2 = end[:2]
            else:
                x2, y2 = gate_list[gate_order[i+1]][:2]

            ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.05, head_length=0.1, fc='g', ec='g', alpha=0.5)

        ax.legend()
        plt.title(f"{scenario} Scenario")
        plt.show()

def create_gate_list(scenario):
    if scenario == 'circle':
        return [
            (1.5, 0, 1, 0.395, 'X', '+'),  # Gate 1
            (3, 1.5, 0.5, 0.395, 'Y', '+'),  # Gate 2
            (1.5, 3, 1, 0.395, 'X', '-'),  # Gate 3
            (0, 1.5, 0.5, 0.395, 'Y', '-'),  # Gate 4
        ]
    elif scenario in ['circle-constant-z', 'circle-3-loop']:
        return [
            (1.5, 0, 0.8, 0.395, 'X', '+'),  # Gate 1
            (3, 1.5, 0.8, 0.395, 'Y', '+'),  # Gate 2
            (1.5, 3, 0.8, 0.395, 'X', '-'),  # Gate 3
            (0, 1.5, 0.8, 0.395, 'Y', '-'),  # Gate 4
        ]
    elif scenario in ['ellipse']:
        return [
            (1.5, 0, 1, 0.395, 'X', '+'),  # Gate 1
            (3, 0.5, 1, 0.395, 'Y', '+'),  # Gate 2
            (1.5, 1, 1, 0.395, 'X', '-'),  # Gate 3
            (0, 0.5, 1, 0.395, 'Y', '-'),  # Gate 4
        ]
    elif scenario == '2-circles':
        return [
            (0.75, 0, 1, 0.395, 'X', '+'),  # Gate 1
            (1.5, 0.75, 1, 0.395, 'Y', '+'),  # Gate 2
            (0.75, 1.5, 1, 0.395, 'X', '-'),  # Gate 3
            (0, 0.75, 1, 0.395, 'Y', '-'),  # Gate 4
            (2.25, 1.5, 1, 0.395, 'X', '+'),  # Gate 5
            (3, 2.25, 1, 0.395, 'Y', '+'),  # Gate 6
            (2.25, 3, 1, 0.395, 'X', '-'),  # Gate 7
            (1.5, 2.25, 1, 0.395, 'Y', '-'),  # Gate 8
        ]
    elif scenario == 'figure-eight':
        return [
            (0.75, 0, 1, 0.395, 'X', '+'),  # Gate 1
            (1.5, 0.75, 1, 0.395, 'Y', '+'),  # Gate 2
            (0.75, 1.5, 1, 0.395, 'X', '-'),  # Gate 3
            (0, 0.75, 1, 0.395, 'Y', '-'),  # Gate 4
            (2.25, 1.5, 1, 0.395, 'X', '+'),  # Gate 5
            (3, 0.75, 1, 0.395, 'Y', '-'),  # Gate 6
        ]
    elif scenario == 'u-shape':
        return [
            (0, 1.5, 1, 0.395, 'Y', '+'),  # Gate 1
            (1.5, 0, 1, 0.395, 'X', '+'),  # Gate 2
            (3, 1.5, 1, 0.395, 'Y', '-'),  # Gate 3
        ]
    elif scenario == 'u-shape-2':
        return [
            (1.5, -1.5, 1, 0.395, 'X', '+'),  # Gate 1
        ]
    elif scenario == 's-shape':
        return [
            (0.5, 0.5, 1, 0.395, 'Y', '+'),  # Gate 1
            (1.5, 1, 1, 0.395, 'Y', '-'),  # Gate 2
            (0.5, 1.5, 1, 0.395, 'Y', '+'),  # Gate 3
            (1.5, 2, 1, 0.395, 'Y', '-'),  # Gate 4
        ]
    elif scenario == 'circle-new':
        return [
            # (x, y, z, size, roll, pitch, yaw, entry_direction)
            (1.5, 0, 0.8, 0.395, 0, 0, 0, '+'),      # Gate 1 - facing +X
            # (3, 1.5, 0.8, 0.395, 0, 0, 90, '+'),     # Gate 2 - facing -Y
            (3.3, 1.4, 0.8, 0.395, 0, 0, 60, '+'),     # Gate 2 - facing -Y
            (1.5, 3, 0.8, 0.395, 0, 0, 180, '+'),    # Gate 3 - facing -X
            (0, 1.5, 0.8, 0.395, 0, 0, -90, '+'),    # Gate 4 - facing +Y
        ]
    elif scenario == 'pid2':
        return [ 
            # (x, y, z, size, roll, pitch, yaw, entry_direction)
            (10.388, 80.774, -43.580 ,  5,  0,0,0,      '+'),
            (18.110, 76.260, -43.580 ,  5,  0,0,0,      '+'), 
            (25.434, 66.287, -43.580 ,  5,  0,0,0,      '+'),
            (30.066, 56.550, -43.580 ,  5,  0,0,0,      '+'),
            (32.801, 45.931, -43.580 ,  5,  0,0,0,      '+'),
            (30.503, 38.200, -43.580 ,  5,  0,0,0,      '+'), 
            (3.264, 37.569, -43.580 ,   5,  0,0,0,      '+'),
            (-17.863, 45.418, -46.580 , 5,  0,0,0,      '+'),
            (-15.494, 63.187, -52.080 , 5,  0,0,0,      '+')

            # (-6.321, 78.212, -55.780 , 5,  0,0,0,      '+'),
            # ( 5.144, 82.385, -55.780 , 5,  0,0,0,      '+'),
            # (14.559, 84.432, -55.180 , 5,  0,0,0,      '+'),
            # (22.859, 82.832, -32.080 , 5,  0,0,0,      '+'),
            # (38.259, 78.132, -31.380 , 5,  0,0,0,      '+'),
            # (51.059, 52.132, -25.880 , 5,  0,0,0,      '+'),
            # (44.959, 38.932, -25.880 , 5,  0,0,0,      '+'),
            # (25.959, 26.332, -17.880 , 5,  0,0,0,      '+'),
            # (11.659, 26.332, -13.780 , 5,  0,0,0,      '+'),
            # (-10.141, 22.632, -6.380 , 5,  0,0,0,      '+')

            # (-23.641, 10.132, 2.120  , 5,  0,0,-180,      '+')
        ]

    else:
        raise ValueError("Invalid scenario")

def create_gate_order(scenario):
    if scenario in ['circle', 'circle-constant-z', 'ellipse', 'circle-new']:
        return ['START', 0, 1, 2, 3, 0, 'END']
    elif scenario == 'circle-3-loop':
        return ['START', 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 'END']
    elif scenario == '2-circles':
        return ['START', 0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 'END']
    elif scenario == 'figure-eight':
        return ['START', 0, 1, 4, 5, 1, 2, 3, 0, 'END']
    elif scenario == 'u-shape':
        return ['START', 0, 1, 2, 'END']
    elif scenario == 'u-shape-2':
        return ['START', 0, 'END']
    elif scenario == 's-shape':
        return ['START', 0, 1, 2, 3, 'END']
    elif scenario == 'pid2':
        # return ['START', 0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 'END']
        return ['START', 0, 1, 2, 3,4,5,6,7,8, 'END']

    else:
        raise ValueError("Invalid scenario")

def create_obstacle_list(scenario):
    if scenario in ['circle', 'circle-new', 'circle-3-loop', 'circle-constant-z', 'u-shape', '']:
        return [(1.5, 1.5, 0, 1.3, 3.0)]  # (x, y, z, radius, height)
    elif scenario == '2-circles':
        return [
            (0.75, 0.75, 0, 0.55, 3.0),  # Obstacle in the first circle
            (2.25, 2.25, 0, 0.55, 3.0)   # Obstacle in the second circle
        ]
    elif scenario == 'figure-eight':
        return [
            (0.75, 0.75, 0, 0.55, 3.0),  # Obstacle in the first circle
            (2.25, 0.75, 0, 0.55, 3.0)   # Obstacle in the second circle
        ]
    elif scenario == 'u-shape-2':
        return [
            (1.5, 0, 0, 1.3, 3.0)
        ]
    elif scenario == 'ellipse':
        return [
            (1.5, 0.5, 0, 0.3, 3.0),
            (1, 0.5, 0, 0.3, 3.0),
            (2, 0.5, 0, 0.3, 3.0),
            (2.5, 0.5, 0, 0.3, 3.0),
            (0.5, 0.5, 0, 0.3, 3.0),
        ]
    elif scenario in ['s-shape', 'pid2']:
        return []
    else:
        raise ValueError("Invalid scenario")

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def rotation_matrix(roll, pitch, yaw):
    """Return rotation matrix given roll, pitch, yaw in degrees"""
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    
    # Roll (X-axis rotation)
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Apply rotations: yaw, then pitch, then roll
    R = R_z @ R_y @ R_x
    return R

def transform_point(point, origin, R):
    """Transform point using rotation matrix R and origin"""
    point = np.array(point)
    origin = np.array(origin)
    translated = point - origin
    rotated = R @ translated
    
    return origin + rotated

def main_func(waypoint_dist=5, scenario='circle', debug_visualization=False, plan_type='Full'):
    print(f"Start path planning for {scenario} scenario with {plan_type} planning")

    gate_list = create_gate_list(scenario)
    gate_order = create_gate_order(scenario)
    obstacle_list = create_obstacle_list(scenario)

    # start = [0, 0, 0.8]  # Start point
    # end = [3, 0, 0.8]    # End point
    start = [6.788, 81.6774, -43.380]  # Start point
    end = [-15.494, 63.187, -52.080]    # End point
    # end = [-15.494, 63.187, -52.080]    # End point
    



    if scenario not in ['circle-constant-z', 'circle-3-loop', 'circle', '2-circles']:
        start = [0, 0, 1]
        end = [3, 0, 1]

    start = [6.788, 81.6774, -43.380]  # Start point
    end = [-23.641, 10.132, 2.120]    # End point

    print_scenario_details(scenario, gate_list, gate_order, start, end, obstacle_list, debug_visualization)

    rrt_star = RRTStar(
        start=start,
        goal=end,
        rand_area=[-0.5, 3.5],
        obstacle_list=obstacle_list,
        gate_list=gate_list,
        expand_dis=3,
        path_resolution=0.2,
        goal_sample_rate=20,
        max_iter=2000,
        connect_circle_dist=15
    )

    if plan_type == 'Full':
        waypoints = []
        for point in gate_order:
            if point == 'START':
                waypoints.append(start)
            elif point == 'END':
                waypoints.append(end)
            else:
                gate = gate_list[point]
                entry_point, exit_point = rrt_star.get_gate_entry_exit_points(gate)
                waypoints.append(entry_point)
                waypoints.append(gate[:3])
                waypoints.append(exit_point)

        full_path = []
        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            rrt_star.start = rrt_star.Node(*start_point)
            rrt_star.end = rrt_star.Node(*end_point)
            rrt_star.current_gate_index = i
            rrt_star.next_gate_index = (i + 1) % (len(waypoints) - 1)
            rrt_star.compute_sampling_area()

            path = None
            for attempt in range(5):
                path = rrt_star.planning(animation=False)
                if path is not None:
                    break
                print(f"Attempt {attempt + 1} failed, retrying...")

            if path is None:
                print(f"Cannot find path from {start_point} to {end_point}")
                return

            if i > 0:
                path = path[1:]
            full_path.extend(path[::-1])

    else:  # 'Fly' planning
        gate = gate_list[gate_order[1]]
        entry_point, _ = rrt_star.get_gate_entry_exit_points(gate)
        path = rrt_star.plan_next_segment(start, entry_point)
        if path is None:
            return None
        smoothed_path = rrt_star.smooth_path(path)
        racing_path = rrt_star.optimize_path_for_racing(smoothed_path)
        final_path = racing_path
        final_path[-1] = entry_point

    if plan_type == 'Full':
        print("Found initial path!")
        smoothed_path = rrt_star.smooth_path(full_path)
        print("Path smoothed")
        racing_path = rrt_star.optimize_path_for_racing(smoothed_path)
        print("Path optimized for racing")
        final_path = racing_path
        final_path[-1] = end

    path_array = np.array(final_path)
    total_length = np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))
    num_points = int(total_length / waypoint_dist)
    
    normalized_path = []
    remaining_dist = 0
    current_segment = 0
    current_point = path_array[0]
    
    for _ in range(num_points):
        while current_segment < len(path_array) - 1:
            segment = path_array[current_segment + 1] - path_array[current_segment]
            segment_length = np.linalg.norm(segment)
            if remaining_dist <= segment_length:
                current_point = path_array[current_segment] + (segment / segment_length) * remaining_dist
                normalized_path.append(current_point)
                remaining_dist += waypoint_dist
                break
            remaining_dist -= segment_length
            current_segment += 1
    
    if np.linalg.norm(normalized_path[-1] - path_array[-1]) > waypoint_dist/2:
        normalized_path.append(path_array[-1])

    dt = 0.3
    final_waypoints = []
    normalized_path = np.array(normalized_path)
    
    velocities = []
    for i in range(len(normalized_path)):
        if i < len(normalized_path) - 1:
            velocity = (normalized_path[i + 1] - normalized_path[i]) / dt
        else:
            velocity = velocities[-1] if velocities else np.zeros(3)
        velocities.append(velocity)

    smoothed_velocities = []
    current_vel = np.zeros(3)
    vel_smoothing = 0.1
    for vel in velocities:
        current_vel += vel_smoothing * (vel - current_vel)
        smoothed_velocities.append(current_vel.copy())

    yaw_angles = []
    for vel in smoothed_velocities:
        yaw = math.atan2(vel[1], vel[0])
        yaw_angles.append(yaw)

    smoothed_yaw = []
    yaw_smoothing = 0.1
    current_yaw = yaw_angles[0]
    for target_yaw in yaw_angles:
        diff = normalize_angle(target_yaw - current_yaw)
        current_yaw += yaw_smoothing * diff
        current_yaw = normalize_angle(current_yaw)
        smoothed_yaw.append(current_yaw)

    orientations = []
    for i in range(len(normalized_path)):
        vel = smoothed_velocities[i]
        speed = np.linalg.norm(vel[:2])
        
        pitch = -speed / 5.0
        
        roll = 0.0
        if i > 0 and speed > 0.1:
            yaw_rate = normalize_angle(smoothed_yaw[i] - smoothed_yaw[i-1]) / dt
            roll = yaw_rate * speed / 9.81
        
        orientations.append((roll, pitch, smoothed_yaw[i]))

    for i in range(len(normalized_path)):
        pos = normalized_path[i]
        vel = smoothed_velocities[i]
        roll, pitch, yaw = orientations[i]
        
        if i < len(normalized_path) - 1:
            next_roll, next_pitch, next_yaw = orientations[i + 1]
            droll = normalize_angle(next_roll - roll) / dt
            dpitch = normalize_angle(next_pitch - pitch) / dt
            dyaw = normalize_angle(next_yaw - yaw) / dt
        else:
            # For last point, use 0s
            droll = dpitch = dyaw = 0.0
        
        waypoint = [
            dt * float(i), # timestamp
            pos[0],        # x
            pos[1],        # y
            pos[2],        # z
            roll,          # roll
            pitch,         # pitch
            yaw,           # yaw
            droll,         # droll
            dpitch,        # dpitch
            dyaw,          # dyaw
            vel[0],        # dx
            vel[1],        # dy
            vel[2]         # dz
        ]
        final_waypoints.append(waypoint)

    if show_animation_2d or show_animation_3d:
        x_list = [wp[1] for wp in final_waypoints]
        y_list = [wp[2] for wp in final_waypoints]
        z_list = [wp[3] for wp in final_waypoints]

        if rrt_star.fig is None:
            rrt_star.setup_animation()

        if show_animation_2d and rrt_star.ax1 is not None:
            rrt_star.ax1.cla()
            rrt_star.setup_plot_2d()
            rrt_star.ax1.plot(x_list, y_list, '-b')
            rrt_star.ax1.plot(x_list[0], y_list[0], 'go', markersize=10, label='Start')
            rrt_star.ax1.plot(x_list[-1], y_list[-1], 'bo', markersize=10, label='End')
            
            arrow_indices = np.linspace(0, len(x_list) - 1, 20, dtype=int)
            for i in arrow_indices:
                x, y = x_list[i], y_list[i]
                dx, dy = final_waypoints[i][10], final_waypoints[i][11]
                rrt_star.ax1.arrow(x, y, dx*0.2, dy*0.2, 
                                head_width=0.05, head_length=0.1, 
                                fc='r', ec='r', alpha=0.5)
            
            if plot_waypoints:
                rrt_star.ax1.plot(x_list, y_list, 'ro', markersize=2, label='Waypoints')
            
            rrt_star.ax1.legend()

        if show_animation_3d and rrt_star.ax2 is not None:
            rrt_star.ax2.cla()
            rrt_star.setup_plot_3d()
            rrt_star.ax2.plot(x_list, y_list, z_list, '-b')
            rrt_star.ax2.scatter(x_list[0], y_list[0], z_list[0], color='g', s=100, label='Start')
            rrt_star.ax2.scatter(x_list[-1], y_list[-1], z_list[-1], color='b', s=100, label='End')
            
            arrow_indices = np.linspace(0, len(x_list) - 1, 20, dtype=int)
            for i in arrow_indices:
                if i + 1 < len(x_list):
                    rrt_star.ax2.quiver(x_list[i], y_list[i], z_list[i],
                                    final_waypoints[i][10]*0.2, # dx
                                    final_waypoints[i][11]*0.2, # dy
                                    final_waypoints[i][12]*0.2, # dz
                                    color='r', alpha=0.5, arrow_length_ratio=0.15)
            
            if plot_waypoints:
                rrt_star.ax2.scatter(x_list, y_list, z_list, color='r', s=10, label='Waypoints')
            
            rrt_star.ax2.legend()

        plt.tight_layout()
        plt.ioff()
        plt.show()

    path_length = np.sum(np.linalg.norm(np.diff(normalized_path, axis=0), axis=1))
    print(f"\nFinal path length: {path_length:.2f} units")
    print(f"Number of waypoints: {len(final_waypoints)}")
    
    return final_waypoints

if __name__ == '__main__':
    scenario = 'pid2'
    result = main_func(scenario=scenario, debug_visualization=debug_visualization, waypoint_dist=1)
    if True or (not show_animation_2d and not show_animation_3d):
        print("Waypoints:")
        print("[Timestamp, x, y, z, roll, pitch, yaw, droll, dpitch, dyaw, dx, dy, dz]")
        for waypoint in result:
            print("[" + ", ".join(f"{value:.3f}" for value in waypoint) + "]")
            
          
# if __name__ == '__main__':
#     scenarios = ['circle-constant-z', 'figure-eight', 'ellipse']
#     for scenario in scenarios:
#         result = main_func(waypoint_dist=waypoint_dist, scenario=scenario, debug_visualization=debug_visualization)
#         print(f"Completed {scenario} scenario")
#         if not show_animation_2d and not show_animation_3d:
#             print("Waypoints:")
#             for waypoint in result:
#                 print("[" + ", ".join(f"{value:.3f}" for value in waypoint) + "]")
#         print("\n")