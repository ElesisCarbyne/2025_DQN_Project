import os, random
from lib.env_manager_final import PathFindEnv
from lib.dqn_utils_final import DQN

def load_env():
    # 복원 환경 무작위 선택
    check_env_pathes = list(zip(os.listdir(os.path.join(os.getcwd(), "checkpoints")), os.listdir(os.path.join(os.getcwd(), "envs"))))
    checkpoint_path, env_path = random.sample(check_env_pathes, k=1)[0]
    
    env = PathFindEnv(height=24, width=24)
    # env_info: (tp_map, src_pos, dest_pos)
    env_info = env.restore(env_path) # 환경 복원
    
    return env_info, checkpoint_path, env

# 주어진 환경 기반 추론
def call_inference(env, init_state, checkpoint_path, verbose=False):
    dqn = DQN(train=False, input_size=24)
    ep_replay = dqn.inference(env=env, init_state=init_state, checkpoint=checkpoint_path, verbose=verbose)

    return ep_replay