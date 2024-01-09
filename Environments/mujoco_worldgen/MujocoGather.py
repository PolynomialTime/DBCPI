from mujoco_worldgen.util.envs import EnvViewer
from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen.util.envs.flexible_load import load_env
from itertools import product
import copy
import numpy as np

class MujocoGather():
    def __init__(self, seed=None, modify_reward=False, render=False) -> None:
        self.env_name = 'examples/particle_gather_multiagent.py'
        foo_kw = {}
        env, args_remaining = load_env(self.env_name,
                                   core_dir=worldgen_path(), envs_dir='examples', xmls_dir='xmls',
                                   return_args_remaining=True, **foo_kw)
        self.env = env
        self.num_agents = 2
        self.observation_space = self.num_agents*2 + 10 + 5  # agent_position + food_position + food_health
        self.ori_idx = None
        self.alter_idx = None
        self.joint_act_nums = 81
        self.ori_alter_acts = None
        self.modify_reward=modify_reward
        self.render = render
        self.init_act_idx()
        self.init_ori_alter_acts()
        if seed:
            env.seed(seed)
            print(env.seed())
        self.env_viewer = EnvViewer(env)

    def step(self, action, render=False):
        obs, rew, done, info = self.env_viewer.env.step(action)
        if self.render:
            self.env_viewer.render()
        agent_pos = copy.copy(obs['qpos'])
        agent_pos_2d = agent_pos[[True,True,False,True,True,False]]  # Only want 2d information
        food_pos = copy.copy(obs['food_pos'])
        food_pos_2d = []
        for food in food_pos:
            food_pos_2d.append(food[0])
            food_pos_2d.append(food[1])
        food_pos_2d = np.array(food_pos_2d)
        food_health = copy.copy(obs['food_health'])
        food_health = food_health.reshape(5)
        format_obs = np.concatenate((agent_pos_2d, food_pos_2d, food_health))
        if self.modify_reward:
            if self.is_state_danger(format_obs):
                rew += np.array([-0.5 for _ in range(self.num_agents)])
        if render:
            self.env_viewer.render()
        format_obs = format_obs / 10  # state to 0~1
        return format_obs, rew, done, info
    
    def reset(self):
        self.env_viewer.env_reset()
        obs, rew, done, info = self.step([0,0,0,0])
        
        return obs

    def joint_acts(self):
        return product([-0.4, 0, 0.4],[-0.4, 0, 0.4],[-0.4, 0, 0.4],[-0.4, 0, 0.4]) ## 2d action for 2 agents


    def init_act_idx(self):
        self.ori_idx = [np.arange(81).reshape(9,9).tolist(),
                        np.arange(81).reshape(9,9).T.tolist()]
        self.alter_idx = [[np.arange(81).reshape(9,9).tolist() for _ in range(9)],
                        [np.arange(81).reshape(9,9).T.tolist() for _ in range(9)]]
        acts_0 = np.arange(81).reshape(9,9).tolist()
        acts_1 = np.arange(81).reshape(9,9).T.tolist()
        for i in range(len(acts_0)):
            self.alter_idx[0][i].remove(acts_0[i])
            self.alter_idx[1][i].remove(acts_1[i])
    def get_ori_idx(self, player):
        return self.ori_idx[player]

    def get_alter_idx(self, player):
        return self.alter_idx[player]

    def init_ori_alter_acts(self):  # size should be 9 * 8=72
        all_poss = [i for i in product((-0.4,0,0.4),repeat=2)]
        result = []
        for i in range(9):
            for j in range(9):
                if i != j:
                    result.append(list(all_poss[i]+all_poss[j]))
        self.ori_alter_acts = result

    def is_state_danger(self, obs, threshold=2):
        center_pos = np.mean(obs[:self.num_agents*2])
        center_floor = np.array([5,5])
        dist_centers = np.linalg.norm(center_pos - center_floor,axis=-1)
        if dist_centers > threshold:
            return True
        else:
            return False
