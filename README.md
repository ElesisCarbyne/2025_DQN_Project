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

envs/ # 환경 객체(맵 객체, 출발지 좌표, 목적지 좌표) 모음
└── ... # ".obj" 파일

lib/
├── agent_utils_final.py # 에이전트 아키텍처 모듈
├── dqn_utils_final.py # DQN 알고리즘 모듈
├── env_manager_final.py # 환경 관리 모듈
└── map_generator_final.py # 맵 생성 모듈

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

# 환경 상세
본 프로젝트에서는 3개의 맵으로 구성된 환경을 자체 구축하여 사용하였습니다.
### [환경 크기 및 형태]
- **맵 크기(모든 맵 공통) :** **24 X 24**
- **최종 맵 형태 :** **3 X 24 X 24**
### [사용 맵]
1. **🗺 지형 맵 :** **벽**(1.0으로 표시, 에이전트가 갈 수 없는 곳)과 **도로**(0.0으로 표시, 에이전트가 갈 수 있는 곳)으로 구성
2. **🏁 목적지 위치 맵 :** **목적지 위치**(1.0으로 표시, 에이전트가 도달해야 하는 곳)를 나타냄(나머지는 0.0으로 표시)
3. **🚩 에이전트 위치 맵 :** **에이전트 위치**(1.0으로 표시, 에이전트의 현재 위치한 곳)를 나타냄(나머지는 0.0으로 표시)
### [맵 예시]
<table>
  <tr>
    <td><img width="446" height="570" alt="Image" src="https://github.com/user-attachments/assets/e02e3f72-492c-41d1-8926-0d03b357adbe"/></td>
    <td><img width="446" height="570" alt="Image" src="https://github.com/user-attachments/assets/9713e96f-efe6-4190-9b6e-30b048345147"/></td>
  </tr>
</table>
<table>
  <tr>
    <td><img width="446" height="570" alt="Image" src="https://github.com/user-attachments/assets/82abe106-59db-41f4-bc04-0dec79785c4e"/></td>
    <td><img width="446" height="570" alt="Image" src="https://github.com/user-attachments/assets/af67db55-d359-4c1a-8876-00a909c27397"/></td>
  </tr>
</table>

# 학습 목표
- 환경을 구성하는 맵의 크기가 커질수록 계산량 급증으로 인한 모델 학습 속도 저하  
- 맵 내 일부 지형애 변화가 발생한 경우 이를 모델 출력에 반영하기 위해서는 반드시 갱신된 전체 맵에 대한 재학습 필요

위 두가지 문제점에서 착안하여 **전체 지역을 부분 지역으로 분할하고, 이들 부분 지역에 대한 부분 지역 경로 탐색 최적화 에이전트의 학습**이 최종 학습 목표입니다.

# 학습 결과
