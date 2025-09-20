# 프로젝트: 경로 탐색 에이전트
본 프로젝트는 DQN 기반 경로 탐색 에이전트 구현을 목표로 하는 프로젝트입니다.
이를 통해 경로 탐색에서 강화학습 알고리즘의 유효성을 실증합니다.

# 필요 라이브러리 설치
```bash
pip install -r requirements.txt
```

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

test_example.ipynb # 에이전트 추론 예시
train_example.ipynb # 에이전트 학습 예시
```

# 기술 스택
<div align=left>
<img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=flat-square&logo=Python&logoColor=white"/>
<img alt="PyTorch" src ="https://img.shields.io/badge/PyTorch-EE4C2C.svg?&style=flat-square&logo=PyTorch&logoColor=white"/>
<img alt="Jupyter" src ="https://img.shields.io/badge/Jupyter-F37626.svg?&style=flat-square&logo=Jupyter&logoColor=white"/>
</div>

# 사용 알고리즘 및 기법
- **DQN**
- **Experience Replay**
- **Target Network**
- **Epsilon decay**
- **AMP(Automatic Mixed Precision)**
- **Gradient clipping**
- **L2 Regularization**

# 학습 목표
전체 지역을 구성하는 부분 지역 최적화 경로 탐색 에이전트를 구현합니다.
### [상세 설명]
환경을 구성하는 맵의 크기가 커질수록 계산량 급증으로 인해 모델 학습 속도가 저하되는 것에 착안하여, 전체 지역을 부분 지역으로 분할하고, 해당 부분 지역 내에서의 경로 탐색에 최적화된 에이전트 구현을 목표로 합니다.

# 환경 상세
본 프로젝트에서는 3개의 맵으로 구성된 환경을 자체 구축하여 사용하였습니다.
### [환경 크기 및 형태]
- **맵 크기(모든 맵 공통) :** **24 X 24**
- **최종 맵 형태 :** **3 X 24 X 24**
### [사용 맵]
1. **🗺 지형 맵 :** **벽**(1.0으로 표시, 에이전트가 갈 수 없는 곳)과 **도로**(0.0으로 표시, 에이전트가 갈 수 있는 곳)으로 구성
2. **🏁 목적지 위치 맵 :** **목적지 위치**(1.0으로 표시, 에이전트가 도달해야 하는 곳)를 나타냄(나머지는 0.0으로 표시)
3. **🚩 에이전트 위치 맵 :** **에이전트 위치**(1.0으로 표시, 에이전트의 현재 위치한 곳)를 나타냄(나머지는 0.0으로 표시)
