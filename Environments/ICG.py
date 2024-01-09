import numpy as np
import gym
from gym import spaces
import pickle as pkl
import copy
from itertools import product

class IterChikenGame(gym.Env):
    def __init__(self, n_agents=2, mod_rew=False,normalize_obs=True) -> None:
        super().__init__()
        self.punishment_reward = 1.5
        self.all_states = [0]
        self.env_name = 'ICG'
        self.num_agents = n_agents
        self.default_reward = 0
        #self.collect_reward = 1
        self.action_space = 2  # 2 movements
        self.joint_act_nums = np.power(self.action_space,self.num_agents)
        # observation space should be agent location + package location + agent energy
        self.observation_space = 1 
        self.s = 0
        self.mod_rew=mod_rew
        self.ori_idx = None
        self.alter_idx = None
        self.init_act_idx()
        self.all_states = self.get_all_states()
        self.all_jacts = [i for i in self.joint_acts()]
        self.possible_states = len(self.all_states)
        self.transition_matrix = self.initialize_trans_prob_dict_sas()
        self.bad_states = self.init_bad_states()
        self.check_transition_prob()
        self.reward_table = self.init_reward_tables()
        self.balance_states = self.init_balance_states()
    def step(self,act):
        if act == [0,0]:
            reward = [0,0]
        elif act == [0,1]:
            reward = [7,2]
        elif act == [1,0]:
            reward = [2,7]
        elif act == [1,1]:
            reward = [6,6]
        return 0,reward,False,None