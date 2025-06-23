import numpy as np
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

class MapGen:
  def __init__(self, height=32, width=32):
    self.height = height
    self.width = width
    self.load_cnt_std = 32
    self.height_avail_scope = int(3 / 16 * height)
    self.width_avail_scope = int(3 / 16 * width)

  def perlin_noise(self):
    # 아래의 변수들은 수정하지 않을 것
    height = self.height # 생성할 numpy 배열의 높이(픽셀 수); default = 32, available = 32, 64, 128
    width = self.width # 생성할 numpy 배열의 너비(픽셀 수); default = 32, available = 32, 64, 128
    scale = 35.0
    octaves = 8

    noise_gen = PerlinNoise(octaves=octaves) # PerlinNoise 객체 생성

    noise_array = np.zeros((height, width)) # perlin noise 맵 배열 생성

    # 각 픽셀에 대해 펄린 노이즈 값 계산 및 배열 채우기
    for y in range(height):
        for x in range(width):
            noise_value = noise_gen([x / scale, y / scale])
            noise_array[y, x] = noise_value

    discret_noise_array = (noise_array >= 0.01).astype(int) # 생성된 펄린 노이즈 맵 이산화

    return discret_noise_array

  def get_indices(self, dim_size):
    load_cnt = dim_size // self.load_cnt_std + 2 # 생성할 최소 도로의 수
    num_partition = random.randint(load_cnt, dim_size // 10) # 지형의 높이 또는 너비를 몇 개의 파티션으로 분할할 것인지 설정(각 파티션 마다 1개의 도로가 생성된다)
    partition_size = dim_size // num_partition # 파티션 크기 설정

    selected_indices = []
    start = 0
    end = start + partition_size - 1

    for i in range(num_partition):
        selected = random.randint(start, end) # 각 파티션에서 도로가 시작될 인덱스를 무작위로 추출
        selected_indices.append(selected)
        selected_indices.append((selected - 1) if selected == end else (selected + 1))
        
        # 다음 파티션의 범위로 이동
        start = end + 1
        end = (dim_size - 1) if i == (num_partition - 2) else (start + partition_size - 1)

    return sorted(selected_indices) # 오름차순으로 정렬

  def draw_load(self, area_map, selected_indices_rows_cols):
    rows_indices, cols_indices = selected_indices_rows_cols

    # 그리드 월드에 도로 영역 표시
    area_map[rows_indices, :] = 1
    area_map[:, cols_indices] = 1

    return area_map

  def set_src_and_dest(self, dims, selected_indices_rows_cols):
    src_dest_map = np.ones(dims) # load_map과 동일한 형태의 출발지와 도착지를 나타낼 맵 생성
    rows_indices, cols_indices = selected_indices_rows_cols

    if random.randint(0, 1) == 0: # 0 = row, 1 = col
        ## 행으로 된 도로 영역에서 출발지와 목적지를 설정하는 경우 ##
        candidate = random.sample(rows_indices[::2], 2)

        src_x, src_y = candidate[0], random.randint(0, self.width_avail_scope - 1) # 출발지 좌표(x, y) 설정
        dest_x, dest_y = candidate[1], random.randint(dims[1] - 1 - self.width_avail_scope, dims[1] - 2) # 목적지 좌표(x, y) 설정
    else:
        ## 열로 된 도로 영역에서 출발지와 목적지를 설정하는 경우 ##
        candidate = random.sample(cols_indices[::2], 2)
        
        src_x, src_y = random.randint(0, self.height_avail_scope - 1), candidate[0] # 출발지 좌표(x, y) 설정
        dest_x, dest_y = random.randint(dims[0] - 1 - self.height_avail_scope, dims[0] - 2), candidate[1] # 목적지 좌표(x, y) 설정

    # 최종 출발지 및 목적지 좌표 설정(행 번호(인덱스)를 x좌표로, 열 번호(인덱스)를 y좌표로 한다)
    final_src_coordinate = (src_x + random.randint(0, 1), src_y + random.randint(0, 1))
    final_dest_coordinate = (dest_x + random.randint(0, 1), dest_y + random.randint(0, 1))

    # 최종 출발지/목적지 위치 맵 생성
    src_dest_map[final_src_coordinate] = 2 # 출발지 표시
    src_dest_map[final_dest_coordinate] = 3 # 목적지 표시

    return src_dest_map, final_src_coordinate, final_dest_coordinate

  def set_agent_init_position(self, dims, src_coordinate):
    position_map = np.ones(dims) # 에이전트 위치 맵

    # 에이전트 초기 위치 표시
    position_map[src_coordinate] = 4

    return position_map

  def generate_map(self):
    # 이산화 펄린 노이즈 맵 생성
    discret_noise_array = self.perlin_noise()

    # 이산화 펄린 노이즈 맵 기반 도로 맵 생성
    dims = discret_noise_array.shape
    selected_indices_rows_cols = (self.get_indices(dims[0]), self.get_indices(dims[1]))
    load_map = self.draw_load(discret_noise_array, selected_indices_rows_cols)

    # 출발지-목적지 맵 생성
    src_dest_map, src_coordinate, dest_coordinate = self.set_src_and_dest(dims, selected_indices_rows_cols)

    # 에이전트 위치 맵 생성
    agent_pos_map = self.set_agent_init_position(dims, src_coordinate)

    # 최종 맵 생성(3채널)
    final_map = np.append(np.expand_dims(load_map, axis=0), np.expand_dims(src_dest_map, axis=0), axis=0)
    final_map = np.append(final_map, np.expand_dims(agent_pos_map, axis=0), axis=0)

    return final_map, src_coordinate, dest_coordinate

if __name__ == "__main__":
    generator = MapGen(height=32, width=32)
    tp_map, src_coordinate, dest_coordinate = generator.generate_map()
    
    print(f"transportation map shape: {tp_map.shape}")
    print(f"initial agent position: {src_coordinate}")
    print(f"source coordinate: {src_coordinate}")
    print(f"destination coordinate: {dest_coordinate}")