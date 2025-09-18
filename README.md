# 프로젝트: 경로 탐색 에이전트
본 프로젝트는 단순화한 도로 지형 맵에서 경로 탐색을 수행하는 에이전트 구현을 목표로 합니다.

# 필요 라이브러리 설치
- $ pip install -r requirements.txt

# 프로젝트 구조
```
checkpoints/ # 모델 체크포인트 모음
└── ... # ".pth" 파일
lib/
├── agent_utils_final.py # 에이전트 아키텍처 모듈
├── dqn_utils_final.py # DQN 알고리즘 모듈
├── env_manager_final.py # 환경 관리 모듈
└── map_generator_final.py # 맵 생성 모듈
obj/ # 환경 정보(맵 객체, 출발지 좌표, 목적지 좌표로 구성) 파일 모음
└── ... # ".env" 파일
```

# 사용 알고리즘 및 기법
- DQN 알고리즘
- Experience Replay
- Target Network
- Epsilon decay
- AMP(Automatic Mixed Precision)
- Gradient clipping

# 환경 상세
본 프로젝트에서는 3개의 맵으로 구성된 환경을 자체 구축하여 사용하였습니다.
## 사용 맵
- 1. 지형 맵: **벽**(1.0으로 표시, 에이전트가 갈 수 없는 곳)과 **도로**(0.0으로 표시, 에이전트가 갈 수 있는 곳)으로 구성
  2. 목적지 위치 맵: **목적지 위치**(1.0으로 표시, 에이전트가 도달해야 하는 곳)를 나타냄(나머지는 0.0으로 표시)
  3. 에이전트 위치 맵: **에이전트 위치**(1.0으로 표시, 에이전트의 현재 위치한 곳)를 나타냄(나머지는 0.0으로 표시)
- 맵 크기(모든 맵 공통): 24 X 24
- 최종 맵 형태: 3 X 24 X 24
