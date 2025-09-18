import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random, math, pickle, os
from lib.map_generator_final import MapGen

class PathFindEnv:
    def __init__(self, height, width):
        self.map_height = height # 교통 맵 높이
        self.map_width = width # 교통 맵 너비
        self.tp_map = None # 교통 맵
        self.first = True # 환경이 처음 생성 및 초기화 되는 것인지의 여부
        self.src_pos = None # 출발지 좌표 및 초기 에이전트 위치 좌표
        self.dest_pos = None # 목적지 좌표
        self.cur_agent_pos = None # 현재 에이전트 위치 좌표
        self.s2d_full_dist = None # 출발지-목적지 간 전체 거리
        self.ep_step_upper_limit = 500 # 에이전트 스텝 상한선
        self.collision_cnt = None # 에피소드 내 충돌 횟수
        
        self.action_space = {
            0:(-1, 0), # up
            1:(1, 0), # down
            2:(0, -1), # left
            3:(0, 1) # right
        }
        self.rewards = {
            "lose":-200.0, # 경로 탐색 실패
            "step":-1.0, # 에이전트 행동에 대한 페널티
            "win":100.0 # 경로 탐색 성공
        }

    def get_dist(self, name, src_pos, dest_pos):
        """ 맨해튼 거리/유클리디안 거리 계산 """
        
        match name:
            case "l1": # 맨해튼 거리
                return sum(map(lambda i, j: abs(i - j), src_pos, dest_pos))
            case "l2": # 유클리디안 거리
                return int(math.sqrt(sum(map(lambda i, j: math.pow((i - j), 2), src_pos, dest_pos))))

    def get_dist_based_reward(self, prev_agent_pos, new_agent_pos, scale_factor=0.5):
        """ 거리 기반 보상 계산 """
        
        prev_remain_dist = self.get_dist("l1", prev_agent_pos, self.dest_pos)
        new_remain_dist = self.get_dist("l1", new_agent_pos, self.dest_pos)

        # 이전 남은 거리와 현재 남은 거리간 차이에 기반한 보상 계산
        reward = (prev_remain_dist - new_remain_dist) * scale_factor
        
        return reward

    def reset(self, fixed=False):
        """ 환경 생성 및 초기화 """
        
        # 새로운 환경 생성 및 초기화
        if not fixed:
            if self.first:
                self.first = False
            map_gen = MapGen(self.map_height, self.map_width)
            self.tp_map, self.src_pos, self.dest_pos = map_gen.generate_map()
            self.cur_agent_pos = self.src_pos
            self.s2d_full_dist = self.get_dist("l1", self.src_pos, self.dest_pos)
            self.collision_cnt = 0
        # 이전에 생성한 환경 유지 및 재초기화
        else:
            if self.first:
                return self.reset(fixed=False)
            self.tp_map[2][self.cur_agent_pos] = 0.0
            self.tp_map[2][self.src_pos] = 1.0
            self.cur_agent_pos = self.src_pos
            self.collision_cnt = 0
            
        return self.tp_map

    def step(self, action, cur_ep_step):
        """ 환경을 한 스텝 진행한다[충돌을 허용한다] """
        
        total_reward = self.rewards["step"]
        terminated = False
        passed = False
        debug = None
        
        movement = self.action_space[action]
        prev_agent_pos = self.cur_agent_pos # 선택한 행동에 따른 위치 갱신전 현재 에이전트 좌표
        new_agent_pos = tuple(map(lambda i, j: i + j, prev_agent_pos, movement)) # 선택한 행동에 따른 위치 갱신후 가상의 에이전트 좌표
        
        # 에이전트가 맵 밖으로 벗어나지 않음 and 충돌이 발생하지 않음
        # self.tp_map[0] : 도로 맵
        if ((0 <= new_agent_pos[0] < self.map_height) and (0 <= new_agent_pos[1] < self.map_width)) and (self.tp_map[0][new_agent_pos] == 0.0):
            # self.tp_map[2] : 에이전트 위치 맵
            # 에이전트를 실제로 이동시킨다
            self.tp_map[2][self.cur_agent_pos] = 0.0
            self.cur_agent_pos = new_agent_pos
            self.tp_map[2][self.cur_agent_pos] = 1.0

            total_reward += self.get_dist_based_reward(prev_agent_pos, new_agent_pos)

            # 에이전트가 목적지에 도달한 경우
            # self.tp_map[1] : 목적지 위치 맵
            if self.tp_map[1][self.cur_agent_pos] == 1.0:
                total_reward += self.rewards["win"]
                terminated = True
                passed = True
        # 에이전트가 맵 밖으로 벗어남 or 충돌이 발생함
        else:
            self.collision_cnt += 1
        # 에피소드 스텝이 상한선에 도달한 경우
        if (not passed) and (cur_ep_step == self.ep_step_upper_limit):
            total_reward += self.rewards["lose"]
            terminated = True

        debug = (self.s2d_full_dist, self.collision_cnt,) # "(self.s2d_full_dist,)"에서 콤마가 없으면 오류가 발생한다
        
        return self.tp_map, total_reward, terminated, passed, debug

    def restore(self, env_info_file):
        """ 학습한 환경 복원 """
        
        # 환경 정보 파일 절대 경로 생성
        env_info_file_abs_path = os.path.join(os.getcwd(), "objs", env_info_file)
        
        # 학습했던 환경 복원
        with open(env_info_file_abs_path, "rb") as f:
            env_info = pickle.load(f)
        
        self.tp_map, self.src_pos, self.dest_pos = env_info
        self.tp_map[2][self.dest_pos] = 0.0
        self.tp_map[2][self.src_pos] = 1.0
        self.cur_agent_pos = self.src_pos
        self.s2d_full_dist = self.get_dist("l1", self.src_pos, self.dest_pos)
        self.collision_cnt = 0

        return self.tp_map, self.src_pos, self.dest_pos

    def get_env_info(self):
        """ 현재 환경 정보 반환 """
        
        return self.tp_map, self.src_pos, self.dest_pos

    def normal_view_render(self, ax=None):
        """ 현재 환경 상태 그래프로 출력 """
        
        maps = self.tp_map[0] + (self.tp_map[1] * 2) + (self.tp_map[2] * 3) # 3개의 map을 하나로 결합
        
        colors = ["white", "black", "green", "red"]
        cmap = mcolors.ListedColormap(colors) # colors에 정의된 색상 리스트로 컬러맵 생성
        
        # 정수 값(0, 1, 2, 3)과 컬러맵의 색상 인덱스(0, 1, 2, 3)를 정확히 매핑하기 위해, 각 색상이 적용될 값의 경계(bounds)를 설정한다
        # Ex) -0.5 ~ 0.5 사이에 해당하는 값을 가진 픽셀은 첫 번째 색상(검정)이 칠해지도록 한다
        bounds = [-0.5, 0.5, 1.5, 2.5, 5.5]
        # BoundaryNorm: 주어진 경계(bounds)를 기준으로 입력된 데이터 값을 정규화하여 컬러맵 인덱스로 변환한다
        norm = mcolors.BoundaryNorm(bounds, cmap.N) # cmap.N은 컬러맵의 색상 개수(여기서는 4)
    
        # origin 옵션을 upper로 설정하면 그래프 y축값이 (64 x 64 맵 기준) -0.5가 아니라 -63.5부터 시작한다
        ax.imshow(maps, # 표시할 2D 배열 데이터
                cmap=cmap, # cmap: 위에서 정의한 커스텀 컬러맵
                norm=norm, # norm: 위에서 정의한 정규화 객체(그리드 월드 셀 값과 색상 매핑 역할)
                origin='lower', # 'lower': 배열 인덱스 (0, 0)이 그림의 좌측 하단에 오도록 설정('upper'는 좌측 상단)
                interpolation='nearest') # 'nearest': 셀 색상이 번지지 않고 각 셀이 명확한 사각형으로 표시됨
        
        # 셀 경계에 맞춰 그리드 선을 표시
        ax.set_xticks(np.arange(-.48, maps.shape[0], 1), minor=False) # x축에서 눈금은 부(minor) 눈금이 아닌 주(major) 눈금을 표시한다
        ax.set_yticks(np.arange(-.48, maps.shape[1], 1), minor=False) # y축에서 눈금은 부(minor) 눈금이 아닌 주(major) 눈금을 표시한다
        ax.set_xlim(left=-.48)
        ax.set_ylim(ymin=-.48)
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.7) # 부(minor) 눈금 위치에 맞춰 그리드 선을 그린다
        ax.tick_params(which='major', # tick_params에 명시한 매개변수 내용들이 적용될 눈금 그룹을 설정
                    bottom=False, # 하단 (주 또는 부) 눈금 표시를 생략한다
                    left=False, # 좌측 (주 또는 부) 눈금 표시를 생략한다
                    labelbottom=False, # 하단 레이블 표시를 생략한다
                    labelleft=False) # 좌측 레이블 표시를 생략한다
        ax.set_title(f'{maps.shape[0]}x{maps.shape[1]} Transportation Map', fontsize=14)
        plt.tight_layout() # 그림 요소들이 겹치지 않도록 조정
        plt.show()

if __name__ == "__main__":
    plt.ion() # plot을 연속적으로 출력하도록 설정
    
    env = PathFindEnv(height=24, width=24)
    cur_state = env.reset(fixed=False)
    cur_ep_step = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    env.normal_view_render(ax)

    for _ in range(10):
        cur_ep_step += 1
        action = random.randint(0, 3)
        print(f"Taking action: {action}")

        next_state, reward, terminated, passed, _ = env.step(action, cur_ep_step)
        print(f"Reward: {reward}")

        env.normal_view_render(ax)

        if terminated:
            print("win") if passed else print("Lose")
            break

    plt.ioff()
    plt.close() # plot을 닫는다