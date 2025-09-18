import numpy as np
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class MapGen:
    def __init__(self, height=24, width=24):
        self.height = height # 생성할 배열의 높이(픽셀 수); default = 24, available = 24, 32
        self.width = width # 생성할 배열의 너비(픽셀 수); default = 24, available = 24, 32
        self.basis = 8 # 기준 단위
        self.src_dest_avail_scope = height // self.basis # 24: 3, 32: 4

    def get_indices(self, dim_size):
        """ 2차원 N x N의 도로 맵에서 도로가 될 행 또는 열을 추출한다 """
        
        partition_cnt = random.randint(2, self.height // self.basis)
        partition_size = dim_size // partition_cnt
        
        selected_indices = []
        start = 0
        end = start + partition_size - 1
        
        for i in range(partition_cnt):
            selected = random.randint(start + 1, end - 2) # 각 파티션에서 도로가 시작될 인덱스를 무작위로 추출
            selected_indices.extend([selected - 1, selected, selected + 1])
            
            # 다음 파티션의 범위로 이동
            start = end + 1
            end = (dim_size - 1) if i == (partition_cnt - 2) else (start + partition_size - 1)
        
        return selected_indices

    def create_load_map(self):
        """ 2차원 N x N의 도로 맵을 생성한다 """
        
        ## 배경 맵 생성 ##
        load_map = np.ones((self.height, self.width))

        ## 도로 맵 생성 ##
        partition_cnt = self.height // self.basis
        row_idx_for_load = []
        base_row_idx = []
        base_col_idx = []

        # 수평 도로 생성
        for i in range(partition_cnt - 1):
            base_idx = self.basis * (i + 1)
            row_idx_for_load.extend([base_idx - 2, base_idx - 1, base_idx])
        base_row_idx.extend(row_idx_for_load[::3] + [self.height])
        
        load_map[row_idx_for_load, :] = 0.0

        # 수직 도로 생성
        start = 0
        for i in range(partition_cnt):
            col_idx_for_load = self.get_indices(self.width)
            base_col_idx.append(col_idx_for_load[::3])
            load_map[start:base_row_idx[i], col_idx_for_load] = 0.0
            start = base_row_idx[i] + 2
        base_row_idx = base_row_idx[:-1]
        base_col_idx = [base_col_idx[0], base_col_idx[-1]]

        # 대각 도로 생성
        match random.randint(0, 2):
            case 0:
                # 좌측 상단에서 우측 하단으로 향하는 대각선
                for i in range(self.height):
                    if (i + 2) < self.width:
                        load_map[i, [i, i+1, i+2]] = 0.0
            case 1:
                # 우측 상단에서 좌측 하단으로 향하는 대각선
                for i in range(self.height):
                    j = self.width - 1 - i
                    if (j - 2) >= 0:
                        load_map[i, [j, j-1, j-2]] = 0.0
                        
        return load_map, (base_row_idx, base_col_idx)

    def create_dest_pos_map(self, row_col_candidates):
        """ 도로 영역이 그려진 2D 그리드 월드 내에 출발지와 목적지를 설정하고 그린다 """
        
        dest_pos_map = np.zeros((self.height, self.width)) # load_map과 동일한 형태의 출발지와 도착지를 나타낼 맵 생성
        dims = dest_pos_map.shape # dims[0]: 행의 개수 = height, dims[1]: 열의 개수 = width
        base_row_idx, base_col_idx = row_col_candidates
        
        # 0 = 수평 도로 영역에서 출발지와 목적지를 선정하는 경우, 1 = 수직 도로 영역에서 출발지와 목적지를 선정하는 경우
        if random.randint(0, 1):
            candidate = random.sample(base_row_idx, 2) # 출발지와 목적지가 될 후보군을 선택
        
            src_x, src_y = candidate[0] + 1, random.randint(0, self.src_dest_avail_scope - 1) # 출발지 좌표(x, y) 선정
            dest_x, dest_y = candidate[1] + 1, random.randint(dims[1] - self.src_dest_avail_scope, dims[1] - 1)
        # 수직 도로 영역에서 출발지와 목적지를 선정하는 경우
        else:
            top_x, top_y = random.randint(0, self.src_dest_avail_scope - 1), random.sample(base_col_idx[0], 1)[0] + 1 # 최상단 수직 도로 영역에서 좌표(x, y) 선정
            down_x, down_y = random.randint(dims[0] - self.src_dest_avail_scope, dims[0] - 1), random.sample(base_col_idx[1], 1)[0] + 1 # 최하단 수직 도로 영역에서 좌표(x, y) 선정
            
            # 최종 출발지와 목적지 설정
            if random.randint(0, 1) == 0:
                # 최상단 수직 도로 영역에서 출발지를 선정하는 경우
                src_x, src_y = top_x, top_y # 출발지 설정
                dest_x, dest_y = down_x, down_y # 목적지 설정
            else:
                # 최하단 수직 도로 영역에서 출발지를 선정하는 경우
                src_x, src_y = down_x, down_y # 출발지 설정
                dest_x, dest_y = top_x, top_y # 목적지 설정

        # 최종 좌표 설정
        # 행 인덱스를 x좌표로, 열 인덱스를 y좌표로 한다
        final_src_pos = (src_x, src_y)
        final_dest_pos = (dest_x, dest_y)
        
        # 목적지 위치 맵 생성
        dest_pos_map[final_dest_pos] = 1.0 # 목적지 표시; 초록색
        
        return dest_pos_map, final_src_pos, final_dest_pos

    def create_agent_pos_map(self, src_pos):
        """ 에이전트의 이동을 나타내는 맵을 생성한다 """
        
        agent_pos_map = np.zeros((self.height, self.width)) # 에이전트 이동을 나타내는 맵
        
        # 에이전트 시작 위치 표시
        agent_pos_map[src_pos] = 1.0
        
        return agent_pos_map

    def generate_map(self):
        # 도로 맵 생성
        load_map, row_col_candidates = self.create_load_map()
        
        # 목적지 위치 맵 생성
        dest_pos_map, src_pos, dest_pos = self.create_dest_pos_map(row_col_candidates)
        
        # 에이전트 위치 맵 생성
        agent_pos_map = self.create_agent_pos_map(src_pos)
        
        # 최종 맵 생성(3채널)
        tp_map = np.concatenate((np.expand_dims(load_map, axis=0), np.expand_dims(dest_pos_map, axis=0), np.expand_dims(agent_pos_map, axis=0)), axis=0)
        
        return tp_map, src_pos, dest_pos

if __name__ == "__main__":
    generator = MapGen(height=24, width=24)
    tp_map, src_pos, dest_pos = generator.generate_map()
    
    print(f"transportation map shape: {tp_map.shape}")
    print(f"initial agent coordinate: {src_pos}")
    print(f"destination coordinate: {dest_pos}")