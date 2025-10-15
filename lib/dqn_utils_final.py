import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, random, pickle
from datetime import date
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from lib.agent_utils_final import Agent

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)
        return

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def empty(self):
        self.buffer.clear()

class DQN:
    def __init__(self, train=True, input_size=24, batch_size=256, gamma=0.99, lr=1e-4, eps_upper=1.0, eps_lower=0.05, eps_rate=10000, buffer_size=50000, update_freq=5000, max_norm=3.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size # 모델 입력 크기
        self.q_net = None # 주 신경망
        
        if train:
            self.batch_size = batch_size # 배치 크기
            self.gamma = gamma # 할인율
            self.lr = lr # 학습률
            
            # 입실론 탐욕 정책 및 입실론 쇠퇴(decay) 관련 변수
            self.eps_upper = eps_upper
            self.eps_lower = eps_lower
            self.eps_rate = eps_rate
            
            # 경험 재현 기법 구현을 위한 변수
            self.replay_memory = ReplayMemory(buffer_size=buffer_size)
            
            # 목표 신경망(target network) 관련 변수
            self.update_freq = update_freq # Target 네트워크 갱신 주기(누적 스텝 단위)
            self.target_net = None # 목표 신경망

            # 손실 함수, 옵티마이저, 그 외 학습 안정성 관련 변수
            self.loss_fn = nn.SmoothL1Loss()
            self.optimizer = None
            self.scaler = torch.GradScaler(self.device) # AMP 스케일러
            self.max_norm = max_norm # gradient clipping 하이퍼파리미터

    def clear_screen(self):
        """ 터미널 출력 지우기 """
        
        # Windows
        if os.name == "nt":
            os.system("cls")
        # Linux or macOS
        else:
            os.system("clear")

        return

    def display(self, env, verbose, path=None):
        """ 환경의 현재 상태를 시각화한다 """
        
        if verbose:
            fig, ax = plt.subplots(figsize=(6, 6))
            env.normal_view_render(ax, path)

        return
        
    def eps_decay(self, ep_cnt_in_env):
        """ 입실론 쇠퇴(decay) 구현 """
        
        return self.eps_lower + (self.eps_upper - self.eps_lower) * np.exp(-1. * ep_cnt_in_env / self.eps_rate) # 입실론 계산

    def select_action(self, cur_state, eps_threshold):
        """ 입실론 탐욕 정책(Epsilon-Greedy Policy)에 따라 행동을 선택 """

        # 탐험(Exploration)
        if random.random() < eps_threshold:
            return random.randint(0, 3)
        # 활용(Exploitation)
        else:
            self.q_net.eval()
            with torch.no_grad():
                q_values = self.q_net(torch.from_numpy(np.expand_dims(cur_state, axis=0)).float().to(self.device))
            return q_values.argmax(dim=1).item()

    def optimize_model(self, cu_ep_step):
        """ Replay Buffer에서 batch_size 만큼 무작위 샘플링한 데이터셋으로 모델 학습을 수행한다 """
        
        # Replay Buffer 내 저장된 데이터(transition)의 수가 batch_size 보다 크면 학습을 수행한다
        if len(self.replay_memory) < self.batch_size:
            return 0.0
        
        self.q_net.train() # 학습 모드
        
        # Replay Buffer에서 batch_size 만큼의 데이터를 무작위로 추출한다
        transition_set = self.replay_memory.sample(self.batch_size)
        
        # transition_set에서 개별 데이터 추출 및 batch 구성
        # transition: (cur_state, action, next_state, reward, terminated)
        cur_state_batch = torch.from_numpy(np.stack([transition[0] for transition in transition_set], axis=0)).to(self.device).float() # cur_state
        action_batch = torch.tensor([transition[1] for transition in transition_set], device=self.device, dtype=torch.long).reshape((-1, 1)) # action
        next_state_batch = torch.from_numpy(np.stack([transition[2] for transition in transition_set], axis=0)).to(self.device).float() # next_state
        reward_batch = torch.tensor([transition[3] for transition in transition_set], device=self.device, dtype=torch.float) # reward
        terminated_batch = torch.tensor([transition[4] for transition in transition_set], device=self.device, dtype=torch.bool) # terminated
        
        # AMP(Automatic Mixed Precision) 구현
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            # 예측 Q-value 계산
            pred_q_values = self.q_net(cur_state_batch).gather(dim=1, index=action_batch)
        
            # 타겟 Q-value 계산
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch).max(dim=1)[0]
            next_q_values[terminated_batch] = 0.0
            target_q_values = reward_batch + (self.gamma * next_q_values)
            
            # Loss 계산
            # Q-러닝 공식: Q(s, a) = Q(s, a) + α(r + γ * max_a' Q_target(s', a') - Q(s, a))
            loss = self.loss_fn(pred_q_values, target_q_values.unsqueeze(dim=1).detach())
        
        # 역전파
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # target network 갱신
        if cu_ep_step % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return round(loss.item(), 3)
    
    def train(self, env, checkpoint=None, max_env_creation_cnt=30, max_ep_win_rate_in_env=90, recent=7000, jupyter=True, verbose=True):
        """ DQN 에이전트의 학습을 수행한다 """
        
        env_creation_cnt = 0 # 환경 생성 횟수

        # 이전 환경 정보 불러오기
        if checkpoint is not None:
            _, env_config = self.load(checkpoint)
            max_env_creation_cnt, env_creation_cnt, max_ep_win_rate_in_env = env_config
        
        # 에피소드 승리 횟수가 max_ep_win_in_env에 이를 때 때까지 환경 생성 및 학습을 수행한다
        while env_creation_cnt != max_env_creation_cnt:
            # 주 신경망 생성
            self.q_net = Agent(input_size=self.input_size).to(self.device)

            # 목표 신경망 생성
            self.target_net = Agent(input_size=self.input_size).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict()) # 주 신경망의 파라미터를 목표 신경망에 불러오기
            self.target_net.eval() # 목표 신경망은 학습을 수행하지 않는다

            # 옵티마이저 설정
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
            
            # 학습을 위한 변수들
            cu_ep_step = 0 # 현재 환경에서 수행된 모든 에피소드들의 스텝 누적합(누적 스텝)
            cur_ep_step = 0 # 현재 에피소드 스텝 경과
            ep_cnt_in_env = 0 # 현재 환경에서의 에피소드 수행 횟수
            ep_win_in_env = 0 # 현재 환경에서의 에피소드 승리 횟수
            ep_result_in_env = deque([], maxlen=recent) # 현재 환경에서 수행된 최근 recent개 에피소드들의 결과 기록
            ep_win_rate_in_env = 0.0 # 현재 환경에서의 에피소드 성공률

            # Replay Memory 초기화
            self.replay_memory.empty()

            # 디버깅 로그
            ep_step_in_env = deque([], maxlen=20) # 현재 환경에서 수행된 모든 에피소드들의 최종 스텝 기록
            win_ep_step_in_env = deque([], maxlen=20) # 현재 환경에서 승리한 에피소들의 최종 스텝 기록
            total_ep_reward = 0.0 # 에피소드 내 전체 보상 합
            avg_total_ep_reward_in_env = [] # 현재 환경에서 수행된 모든 에피소드 각각에 대한 전체 보상의 평균 기록
            total_ep_loss = 0.0 # 에피소드 내 전체 손실 합
            avg_total_ep_loss_in_env = [] # 현재 환경에서 수행된 모든 에피소드 각각에 대한 전체 손실의 평균 기록
            ep_collision_cnt_in_env = deque([], maxlen=20) # 현재 환경에서 수행된 모든 에피소드 각각의 충돌 횟수 기록

            # 입실론 초기화
            eps_threshold = self.eps_upper

            # 새로운 환경 생성 및 초기화
            cur_state = env.reset(fixed=False) + np.random.rand(3, env.map_height, env.map_width) / 10.0
            env_creation_cnt += 1
            print(f"Create new Env : ({env_creation_cnt}/{max_env_creation_cnt}), eps_upper/eps_lower : {self.eps_upper}/{self.eps_lower}")
            
            # 현재 환경에서 에피소드 수행
            while True:
                cu_ep_step += 1
                cur_ep_step += 1
                action = self.select_action(cur_state, eps_threshold)
                next_state, reward, terminated, passed, debug = env.step(action, cur_ep_step) # debug: (self.s2d_full_dist, self.collision_cnt,)
                next_state = next_state + np.random.rand(3, env.map_height, env.map_width) / 10.0
                total_ep_reward += reward
                
                # 에피소드 수행 결과 성공인 경우
                if passed:
                    ep_win_in_env += 1
                    win_ep_step_in_env.append(cur_ep_step)

                self.replay_memory.append((cur_state, action, next_state, reward, terminated))
                total_ep_loss += self.optimize_model(cu_ep_step) # 에이전트 학습 및 타겟 네트워크 갱신
                
                # 에피소드 종료
                if terminated:
                    clear_output(wait=True) if jupyter else self.clear_screen()
                    ep_cnt_in_env += 1
                    ep_result_in_env.append(passed)
                    ep_win_rate_in_env = sum(ep_result_in_env) / recent * 100
                    ep_step_in_env.append(cur_ep_step)
                    avg_total_ep_reward_in_env.append(round(total_ep_reward / cur_ep_step, 3))
                    total_ep_reward = 0.0 # 다음 에피소드를 위해 초기화
                    avg_total_ep_loss_in_env.append(round(total_ep_loss / cur_ep_step, 3))
                    total_ep_loss = 0.0 # 다음 에피소드를 위해 초기화
                    ep_collision_cnt_in_env.append(debug[1])
                        
                    print(f"Env [{env_creation_cnt}/{max_env_creation_cnt}]: Episode {ep_cnt_in_env} Done!")
                    print(f"Distance between source to destination: {debug[0]}")
                    print(f"[Win rate]: {ep_win_rate_in_env:.3f} // Current eps_threshold: {eps_threshold:.4f} // Current ep_win_in_env = {ep_win_in_env}")
                    print(f"Tracking win episode total step: {list(win_ep_step_in_env)}")
                    print(f"Tracking episode step(currrent 20 episode): {list(ep_step_in_env)}")
                    print(f"Moving average of episode step: {sum(ep_step_in_env) / 20}")
                    print(f"Tracking average of total episode loss: {avg_total_ep_loss_in_env[0 if not ep_cnt_in_env // 20 else ep_cnt_in_env - 20:]}")
                    print(f"Tracking average of total episode reward: {avg_total_ep_reward_in_env[0 if not ep_cnt_in_env // 20 else ep_cnt_in_env - 20:]}")
                    print(f"Tracking collisoin count per episode: {list(ep_collision_cnt_in_env)}")
                    print(f"Moving average of collisoin count per episode: {sum(ep_collision_cnt_in_env) / 20:.2f}")

                    # 모니터링을 위한 그래프 출력
                    if verbose:
                        fig = plt.figure(figsize=(14, 4))
                        ax1 = fig.add_subplot(121)
                        ax1.plot(avg_total_ep_reward_in_env)
                        ax1.set_title("Average of total episode reward")
                        ax1.set_xticks(np.arange(0, ep_cnt_in_env // 1000 * 1000 + 2000, 1000))
                        ax1.set_xlabel("Episode")
                        ax1.set_ylabel("Average reward")
                        
                        ax2 = fig.add_subplot(122)
                        ax2.plot(avg_total_ep_loss_in_env)
                        ax2.set_title("Average of total episode loss")
                        ax2.set_xticks(np.arange(0, ep_cnt_in_env // 1000 * 1000 + 2000, 1000))
                        ax2.set_xlabel("Episode")
                        ax2.set_ylabel("Average loss")
    
                        plt.tight_layout()
                        plt.show()

                    # 현재 환경에서의 에피소드 성공률이 목표 성공률에 도달한 경우
                    if ep_win_rate_in_env >= max_ep_win_rate_in_env:
                        # 학습 검증에 성공한 경우
                        if self.validation(env):
                            self.save(max_env_creation_cnt, env_creation_cnt, max_ep_win_rate_in_env, ep_win_in_env, ep_cnt_in_env, env.get_env_info()) # 에이전트 체크포인트 저장 및 학습이 완료된 환경 정보 저장
                            break
                        # 학습 검증에 실패한 경우 and 현재 환경에서의 에피소드 성공률이 (max_ep_win_rate_in_env + 5)%를 넘어서는 경우
                        elif ep_win_rate_in_env >= (max_ep_win_rate_in_env + 5):
                            # 현재 N번째 환경을 다시 생성하여 처음부터 다시 학습을 수행한다
                            env_creation_cnt -= 1
                            break
                        # 학습 검증에 실패한 경우
                        else:
                            # 학습 검증에 성공할 때 까지 에피소드를 반복 수행한다
                            continue
                    # 현재 환경에서의 에피소드 성공률이 목표 성공률에 도달하지 못한 경우
                    else:
                        # 목표 에피소드 성공률에 도달할 때까지 현재 환경에서의 학습을 반복한다
                        cur_ep_step = 0
                        eps_threshold = self.eps_decay(ep_cnt_in_env)
                        # 현재 환경을 초기화하고 에피소드를 다시 수행한다
                        cur_state = env.reset(fixed=True) + np.random.rand(3, env.map_height, env.map_width) / 10.0 # "np.random.rand(3, env.map_height, env.map_width) / 100"은 학습 안정성을 위해 추가하는 잡음이다
                        continue
        
                cur_state = next_state # 다음 상태를 현재 상태로 설정
        
        print("===== Train Done =====")
        return

    def validation(self, env):
        """ 에이전트 학습 검증 """
        
        self.q_net.eval() # 평가 모드
        cur_ep_step = 0
        cur_state = env.reset(fixed=True)

        # 학습 검증
        while True:
            cur_ep_step += 1
            with torch.no_grad():
                q_values = self.q_net(torch.from_numpy(np.expand_dims(cur_state, axis=0)).float().to(self.device))
                action = q_values.argmax(dim=1).item()
            next_state, _, terminated, passed, _ = env.step(action, cur_ep_step)
            
            # 에피소드 종료
            if terminated:
                return passed

            cur_state = next_state # 다음 상태를 현재 상태로 설정
    
    def inference(self, env, init_state, checkpoint, verbose=False, path=None):
        """ 학습된 에이전트로 특정 환경에서 에피소드를 수행하고 그 결과를 리플레이 형태로 반환한다 """

        checkpoint_abs_path, _ = self.load(checkpoint)

        self.q_net = Agent(input_size=self.input_size).to(self.device) # 주 신경망 생성
        self.q_net.load_state_dict(torch.load(checkpoint_abs_path, weights_only=True)) # 에이전트 체크포인트 불러오기
        self.q_net.eval() # 평가 모드
        
        cur_ep_step = 0
        cur_state = init_state.copy()
        ep_replay = np.expand_dims(cur_state, axis=0) # 에피소드 내 모든 상태 기록

        # 현재 상태 출력
        self.display(env, verbose, path+f"-{len(ep_replay)}.png" if path is not None else path)

        # 에피소드 수행
        while True:
            cur_ep_step += 1
            with torch.no_grad():
                q_values = self.q_net(torch.from_numpy(np.expand_dims(cur_state, axis=0)).float().to(self.device))
                action = q_values.argmax(dim=1).item()
            next_state, _, terminated, passed, _ = env.step(action, cur_ep_step)

            # 에피소드 진행 과정 기록
            ep_replay = np.concatenate((ep_replay, np.expand_dims(next_state, axis=0)), axis=0)

            # 현재 상태 출력
            self.display(env, verbose, path+f"-{len(ep_replay)}.png" if path is not None else path)

            # 에피소드 종료
            if terminated:
                print(f"<<< Win >>>") if passed else print(f"<<< Lose >>>")
                break
            
            cur_state = next_state # 다음 상태를 현재 상태로 설정

        print("===== Inference Done =====")
        return ep_replay

    def save(self, max_env_creation_cnt, env_creation_cnt, max_ep_win_rate_in_env, ep_win_in_env, ep_cnt_in_env, env_info=None):
        """ 에이전트 체크포인트와 환경 저장 """

        # 저장 일시
        dates = str(date.today())
        
        # 체크포인트 절대 경로 생성
        checkpoint_abs_path = os.path.join(os.getcwd(), "checkpoints", f"checkpoint8_{max_env_creation_cnt}_{env_creation_cnt:02}_{max_ep_win_rate_in_env}_{ep_win_in_env}_{ep_cnt_in_env}_{dates}.pth")

        # 에이전트 체크포인트 저장
        torch.save(self.q_net.state_dict(), checkpoint_abs_path)

        # 환경 정보를 저장하는 경우
        if env_info is not None:
            # 환경 정보 파일 절대 경로 생성
            env_info_abs_path = os.path.join(os.getcwd(), "envs", f"env8_{max_env_creation_cnt}_{env_creation_cnt}_{dates}.obj")
    
            # 환경 정보 파일 저장
            with open(env_info_abs_path, "wb") as f:
                pickle.dump(env_info, f)

        print(f"Save complete!!!")
        return
        
    def load(self, checkpoint):
        """ 체크포인트 및 환경 불러오기 """
        
        # max_env_creation_cnt, env_creation_cnt, max_ep_win_rate_in_env 환경 설정 불러오기
        env_config = [int(item) for item in checkpoint.split('_')[1:4]]

        # 체크포인트 절대 경로 생성
        checkpoint_abs_path = os.path.join(os.getcwd(), "checkpoints", checkpoint)

        return checkpoint_abs_path, env_config