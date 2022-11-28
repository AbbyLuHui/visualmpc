import numpy as np
import math
import random
import os
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Cone, ImageFrame
from ..params import DEFAULT_PARAMS
from PIL import Image

random.seed(42)

class Square(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self,
        num_rows=5,
        num_cols=5,
        room_size=2,
        max_episode_steps=None,
        exp_index=0,
        **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25
        self.img_index = 0

        super().__init__(
            max_episode_steps = max_episode_steps or num_rows * num_cols * 24,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size
            

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall',
                    floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = rows[i][j]

                if (j < self.num_rows - 1) and ((i == 0) or (i == self.num_rows - 1)):
                    neighbor_h = rows[i][j+1]
                    self.connect_rooms(room, neighbor_h, min_z = room.min_z, max_z = room.max_z)
                if (i < self.num_rows - 1) and ((j == 0) or (j == self.num_cols - 1)):
                    neighbor_v = rows[i+1][j]
                    self.connect_rooms(room, neighbor_v, min_x = room.min_x, max_x = room.max_x)
        
        # Generate maps 
        maze = [[0 for i in range(self.num_cols)] for j in range(self.num_rows)]

        for i in range(1, self.num_rows-1):
            for j in range(1, self.num_cols-1):
                maze[i][j] = 4

        rand = random.random()
        if rand < 0.05:
            self.connect_rooms(rows[2][0], rows[2][1], min_z = rows[2][0].min_z, max_z = rows[2][0].max_z)
            maze[2][1] = 0
        
        elif rand > 0.05 and rand < 0.10:
            self.connect_rooms(rows[2][0], rows[2][1], min_z = rows[2][0].min_z, max_z = rows[2][0].max_z)
            self.connect_rooms(rows[2][1], rows[2][2], min_z = rows[2][1].min_z, max_z = rows[2][1].max_z)
            maze[2][1] = 0
            maze[2][2] = 0

        elif rand > 0.10 and rand < 0.15:
            self.connect_rooms(rows[2][2], rows[2][3], min_z = rows[2][2].min_z, max_z = rows[2][2].max_z)
            self.connect_rooms(rows[2][3], rows[2][4], min_z = rows[2][3].min_z, max_z = rows[2][3].max_z)
            maze[2][3] = 0
            maze[2][2] = 0
       
        elif rand > 0.15 and rand < 0.20:
            self.connect_rooms(rows[2][3], rows[2][4], min_z = rows[2][3].min_z, max_z = rows[2][3].max_z)
            maze[2][3] = 0
            
        elif rand > 0.20 and rand < 0.25:
            self.connect_rooms(rows[0][2], rows[1][2], min_x = rows[0][2].min_x, max_x = rows[0][2].max_x)
            maze[1][2] = 0

        elif rand > 0.25 and rand < 0.30:
            self.connect_rooms(rows[0][2], rows[1][2], min_x = rows[0][2].min_x, max_x = rows[0][2].max_x)
            self.connect_rooms(rows[1][2], rows[2][2], min_x = rows[1][2].min_x, max_x = rows[1][2].max_x)
            maze[1][2] = 0
            maze[2][2] = 0

        
        elif rand > 0.30 and rand < 0.35:
            self.connect_rooms(rows[2][2], rows[3][2], min_x = rows[2][2].min_x, max_x = rows[2][2].max_x)
            self.connect_rooms(rows[3][2], rows[4][2], min_x = rows[3][2].min_x, max_x = rows[3][2].max_x)
            maze[3][2] = 0
            maze[2][2] = 0

        elif rand > 0.35 and rand < 0.40:
            self.connect_rooms(rows[3][2], rows[4][2], min_x = rows[3][2].min_x, max_x = rows[3][2].max_x)
            maze[3][2] = 0

        elif rand > 0.40 and rand < 0.55:
            for j in range(4):
                self.connect_rooms(rows[2][j], rows[2][j+1], min_z = rows[2][j].min_z, max_z = rows[2][j].max_z)
                maze[2][j+1] = 0

        elif rand > 0.55 and rand < 0.70:
            for i in range(4):
                self.connect_rooms(rows[i][2], rows[i+1][2], min_x = rows[i][2].min_x, max_x = rows[i][2].max_x)
                maze[i+1][2] = 0

        elif rand > 0.70 and rand < 0.85:
            for i in range(4):
                self.connect_rooms(rows[i][2], rows[i+1][2], min_x = rows[i][2].min_x, max_x = rows[i][2].max_x)
                maze[i+1][2] = 0
            for j in range(4):
                self.connect_rooms(rows[2][j], rows[2][j+1], min_z = rows[2][j].min_z, max_z = rows[2][j].max_z)
                maze[2][j+1] = 0
        
        else:
            pass

       
        #room_idx = [[i, 0] for i in range(1, self.num_rows)] + [[self.num_rows-1, i] for i in range(1, self.num_cols)] + [[i, self.num_cols-1] for i in range(self.num_rows-2, 0, -1)]  + [[0,i] for i in range(self.num_cols-1, 0, -1)]
        room_idx = [[0, i] for i in range(1, self.num_rows)] + [[i, self.num_rows-1] for i in range(1, self.num_cols)] + [[self.num_cols-1, i] for i in range(self.num_cols-2, 0, -1)] + [[i, 0] for i in range(self.num_cols-1, 0, -1)]

        goal_cnt = 1
        obstacle_cnt = np.random.choice(np.arange(0, 4), p=[0.0, 0.35, 0.35, 0.3])
        pos = random.sample(range(len(room_idx)), obstacle_cnt)
        #random.shuffle(pos)

        #first_move = 1 if pos[0] == min(pos) else 2

        goal_x, goal_y = room_idx[pos[0]]
        #self.box = self.place_entity(Box(color='red'), room=rows[goal_y][goal_x])
        colors = ['green', 'blue', 'purple', 'yellow', 'grey']
        for i in range(obstacle_cnt):
            obs_x, obs_y = room_idx[pos[i]]
            maze[obs_x][obs_y] = 3
            
            color_idx = random.randint(0, len(colors)-1)
            rand = random.random()
            if rand < 0.5:
                self.place_entity(Box(color=colors[color_idx], is_hidden=True), room=rows[obs_x][obs_y])
            elif rand < 0.6:
                self.place_entity(Cone(is_hidden=True), room=rows[obs_x][obs_y])
            else:
                self.place_entity(Ball(color=colors[color_idx], is_hidden=True), room=rows[obs_x][obs_y])
        
        reachable = self.get_reachable_locs(maze)

        if len(reachable) < 2:
            return [], 0.0, True, 0
        goal_pos = random.sample(range(len(reachable)), 1)
        goal_x, goal_y = reachable[goal_pos[0]]
        
        start = (0, 0)
        end = (goal_x, goal_y)
        path = self.search(maze, start, end)
        path_count = self.multiple_path_check(maze, start, end)
        path, actions = self.merge_path(path)
        maze[0][0] = 1
        maze[goal_x][goal_y] = 2

        rand = random.random()
        dir = -math.pi / 2 if rand < 0.5 else 0.0

        self.place_agent(pos=np.array([1.0,0,1.0]), path=actions, dir=dir)
        if self.root_dir and not os.path.isdir(os.path.join(self.root_dir, 'actions')):
            os.makedirs(os.path.join(self.root_dir, 'actions'))
        if self.root_dir and not os.path.isdir(os.path.join(self.root_dir, 'maps')):
            os.makedirs(os.path.join(self.root_dir, 'maps'))
        if self.root_dir and not os.path.isdir(os.path.join(self.root_dir, 'agent_fov')):
            os.makedirs(os.path.join(self.root_dir, 'agent_fov'))
        if self.root_dir and not os.path.isdir(os.path.join(self.root_dir, 'maps_gt_obs')):
            os.makedirs(os.path.join(self.root_dir, 'maps_gt_obs'))

        #self.exp_index = len(os.listdir('./logs/actions/'))
        if self.root_dir:
            np.save(self.root_dir + '/maps_gt_obs/map' + str(self.exp_index).zfill(5) + '.npy', np.array(maze))

        
        for i in range(obstacle_cnt):
            obs_x, obs_y = room_idx[pos[i]]
            maze[obs_x][obs_y] = 0

        if self.root_dir:
            np.save(self.root_dir + '/maps/map' + str(self.exp_index).zfill(5) + '.npy', np.array(maze))

        maze[0][0] = 0
        maze[goal_x][goal_y] = 0
        path_no_obs = self.search(maze, start, end)
        path_no_obs, actions_no_obs = self.merge_path(path_no_obs)

        no_detour = False
        if (len(path) == len(path_no_obs)) and (path == path_no_obs):
            no_detour = True
        return actions, dir, no_detour, path_count
        


    def step(self, action):
        #1 - up, 2 - down, 3 - left, 4 - right
        exp_dir = self.root_dir + '/agent_fov/{}'.format(str(self.exp_index).zfill(5))
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        obs, reward, done, info = super().step(action)

        if action == self.actions.move_forward:
            im = Image.fromarray(obs)
            im.save(exp_dir + '/{}.png'.format(str(self.img_index).zfill(2)))
            self.img_index += 1

        #if self.near(self.box):
        #    reward += self._reward()
        #    done = True

        return obs, reward, done, info

    def get_reachable_locs(self, maze):
        locs = []
        stack = [(0,0)]
        visited = set()
        while stack:
            i, j = stack.pop(0)
            for x, y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                x_new = x + i
                y_new = y + j
                
                if x_new < 0 or x_new >= len(maze) or y_new < 0 or y_new >=len(maze[0]):
                    continue
                if maze[x_new][y_new] != 0 or (x_new, y_new) in visited:
                    continue 
                visited.add((x_new, y_new))
                stack.append((x_new, y_new))
                if x_new != 0 or y_new != 0:
                    locs.append([x_new, y_new])
        return locs

    def merge_path(self, path):
        if not path:
            return []
        key_points = []
        actions = []
        prev = -1
        for i in range(0, len(path) - 1):
            step = prev
            cur_x, cur_y = path[i]
            next_x, next_y = path[i+1]
            if cur_x == next_x + 1:
                step = 1
            elif cur_x == next_x - 1:
                step = 2
            elif cur_y == next_y + 1:
                step = 3
            elif cur_y == next_y - 1:
                step = 4
            else:
                print("incorrect path")
            actions.append(step)

            if step != prev:
                key_points.append((cur_x, cur_y))
            prev = step
        key_points.append(path[-1])
        actions.append(0)
        return key_points, actions

    def multiple_path_check(self, maze, start, end):
        start_node = Node(None, start)
        end_node = Node(None, end)
        visited = set()
        queue = [start_node]
        path_count = 0
        while queue:
            current_node = queue.pop(0)
            visited.add(current_node.position)

            if current_node == end_node:
                path_count += 1
                continue

            for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:

                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                        len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                    continue

                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                if node_position in visited:
                    continue

                new_node = Node(current_node, node_position)
                queue.append(new_node)
        return path_count

                

    def search(self, maze, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""
        start_node = Node(None, start)
        end_node = Node(None, end)
        visited = set()
        queue = [start_node]
        while queue:
            current_node = queue.pop(0)
            visited.add(current_node.position)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]
            
            for new_position in [(0, 1), (1, 0), (0, -1), (-1, 0)]:

                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                    continue

                if maze[node_position[0]][node_position[1]] != 0:
                    continue 
                
                if node_position in visited:
                    continue

                new_node = Node(current_node, node_position)
                queue.append(new_node)
        return []

class Node():
    """A node class for Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

    def __eq__(self, other):
        return self.position == other.position
