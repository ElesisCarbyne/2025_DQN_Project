import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from mapgeneration2 import MapGen

class TransportationEnv:
    def __init__(self, height, width):
        self.map_height = height
        self.map_width = width
        self.mth_factor = 2.5 # 맨하탄 거리에 적용할 배수
        self.tp_map = None # 지형 맵
        self.first = True # 환경이 처음 생성 및 초기화 되는 것인지의 여부
        self.src_coordinate = None # 출발지 좌표 및 초기 에이전트 좌표
        self.dest_coordinate = None # 목적지 좌표
        self.cur_agent_pos = None # 현재 에이전트 위치
        self.agent_step_upper_limit = None # 에이전트 스텝 수의 상한선
        self.agent_step_remains = None # 남은 에이전트 스텝 수
        
        self.action_space = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        self.rewards = {
            'lose':-30, # 경로 탐색 실패
            'invalid':-10, # 유효하지 않은 행동
            'valid':-1, # 유효한 행동이나 목적지에는 도달하지 못함
            "win":10 # 경로 탐색 성공
        }

    def mth_dist(self):
        return sum(map(lambda i, j: abs(i - j), self.src_coordinate, self.dest_coordinate))
    
    def reset(self, fixed=False):
        if self.first == True or fixed == False: # 새로운 환경 생성 및 초기화
            self.first = False
            map_gen = MapGen(self.map_height, self.map_width)
            self.tp_map, self.src_coordinate, self.dest_coordinate = map_gen.generate_map()
            self.cur_agent_pos = self.src_coordinate
            self.agent_step_remains = self.agent_step_upper_limit = int(self.mth_dist() * self.mth_factor)
        elif self.first == False and fixed == True: # 이전에 생성한 환경 유지 및 초기화
            self.tp_map[2][self.cur_agent_pos] = 1
            self.tp_map[2][self.src_coordinate] = 4
            self.cur_agent_pos = self.src_coordinate
            self.agent_step_remains = self.agent_step_upper_limit
        
        return self.tp_map

    def step(self, action):
        movement = self.action_map[self.action_space[action]]
        new_agent_pos = tuple(map(lambda i, j: i + j, self.cur_agent_pos, movement))
        self.agent_step_remains -= 1

        # 에이전트의 행동이 유효한 경우
        # self.tp_map[0] : 지형 맵
        # self.tp_map[1] : 출발지/목적지 위치 맵
        # self.tp_map[2] : 에이전트 위치 맵
        if((0 <= new_agent_pos[0] < self.map_height) and (0 <= new_agent_pos[1] < self.map_width) and self.tp_map[0][new_agent_pos] != 0):
            self.tp_map[2][self.cur_agent_pos] = 1
            self.cur_agent_pos = new_agent_pos
            self.tp_map[2][self.cur_agent_pos] = 4

            # 에이전트가 목적지에 도달한 경우
            if(self.tp_map[1][self.cur_agent_pos] == 3):
                reward = self.rewards['win']
                terminated = True
            # 에이전트 스텝 수가 상한선에 도달한 경우
            elif self.agent_step_remains == 0:
                reward = self.rewards['lose']
                terminated = True
            # 에이전트가 목적지에 도달하지 못한 경우
            else:
                reward = self.rewards['valid']
                terminated = False
        # 에이전트의 행동이 유효하지 않은 경우
        else:
            # 에이전트 스텝 수가 상한선에 도달한 경우
            if self.agent_step_remains == 0:
                reward = self.rewards['lose']
                terminated = True
            else:
                reward = self.rewards['invalid']
                terminated = False

        return (self.tp_map, reward, terminated)

    def env_info(self):
        return self.tp_map, self.src_coordinate

    def normal_view_render(self, ax=None, pause_time=1.0):
        maps = self.tp_map[0] * self.tp_map[1] * self.tp_map[2] # 3개의 map을 하나로 결합
        
        colors = ["black", "white", "green", "blue", "red"]
        cmap = mcolors.ListedColormap(colors) # colors에 정의된 색상 리스트로 컬러맵 생성
        
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 12.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(maps, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        
        ax.set_xticks(np.arange(-.48, maps.shape[0], 1), minor=False)
        ax.set_yticks(np.arange(-.48, maps.shape[1], 1), minor=False)
        ax.set_xlim(left=-.48)
        ax.set_ylim(ymin=-.48)
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.7)
        ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(f'{maps.shape[0]}x{maps.shape[1]} Transportation Map', fontsize=14)
        plt.tight_layout()

        plt.pause(pause_time)
        plt.show()

if __name__ == "__main__":
    plt.ion()
    
    env = TransportationEnv(height=32, width=32)
    state = env.reset()

    fig, ax = plt.subplots(figsize=(6, 6))
    env.normal_view_render(ax, pause_time=1.0)

    for _ in range(10):
        action = random.randint(0, 3)
        print(f"Taking action: {action}")

        next_state, reward, terminated = env.step(action)
        print(f"Reward: {reward}, Terminated: {terminated}")

        env.normal_view_render(ax, pause_time=1.0)

        if terminated:
            if reward == -30:
                print("Lose")
                break
            else:
                print("Win")
                break

    plt.ioff()
    plt.close()