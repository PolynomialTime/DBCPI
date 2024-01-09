import numpy as np
import gym
from gym import spaces
import pickle as pkl
import copy
from itertools import product

class Survive(gym.Env):
    """
    Num food
    Num wood
    Health for each agent
    """
    FOOD_COLLECT = 2
    WOOD_COLLECT = 2
    def __init__(self, n_agents=3, max_food = 5, max_wood = 5, max_health=3,health_thres=1,explore_target = 100, mod_rew=False,normalize_obs=True) -> None:
        super().__init__()
        self.all_states = None
        self.env_name = 'Survive'
        self.num_agents = n_agents
        self.max_food = max_food
        self.max_wood = max_wood
        self.max_health = max_health
        self.health_thres = health_thres
        self.action_space = 4  # 4 movements
        self.joint_act_nums = np.power(4,self.num_agents)
        # observation space should be agent location + package location + agent energy
        self.current_explore = 0
        self.explore_target = explore_target
        self.observation_space = self.num_agents+2
        self.s = np.zeros(self.observation_space)  # Store the current location of agents
        self.health = np.zeros(self.num_agents)
        self.mod_rew=mod_rew
        self.normalize_obs=normalize_obs
        self.ori_idx = None
        self.alter_idx = None
        self.init_act_idx()
        self.all_states = self.get_all_states()

    def step(self, act):
        ## For each agent, 0 for collecting food, 1 for collecting wood, 2 for resting, 3 for exploring
        done = False
        food_level = self.s[0]
        wood_level = self.s[1]
        reward = np.zeros(self.num_agents)
        explore_step = 0
        for agent in range(self.num_agents):
            action = act[agent]
            if self.health[agent]<=0:  # Force to rest
                self.health[agent] = 3
            else:
                if action == 0:
                    food_level = min(self.max_food, food_level+self.FOOD_COLLECT) 
                elif action == 1:
                    wood_level = min(self.max_wood, wood_level+self.WOOD_COLLECT) 
                elif action == 2:
                    self.health[agent] = 3
                elif action == 3:
                    self.current_explore += 1
                    explore_step+1
        self.s = np.concatenate(([food_level,wood_level],self.health))
        obs =copy.copy(self.s)
        if self.mod_rew:
            if self.is_state_danger(obs):
                for agent in range(self.num_agents):
                    reward[agent] -= 0.5
            for agent in range(self.num_agents):
                reward[agent] += explore_step
        if self.current_explore >= 100:
            done = True
        
        return obs, reward, done, {}

    def reset(self):
        self.current_explore = 0
        self.health = np.repeat(self.max_health,self.num_agents)
        wood_level = self.max_wood
        food_level = self.max_food
        self.s = np.concatenate(([food_level,wood_level],self.health))
        return self.s


    def is_state_danger(self, state):
        return any(self.health < self.health_thres)

    def joint_acts(self):
        return product([0, 1, 2, 3], repeat=self.num_agents)  # Top, Right, Down, Left

    def get_ori_idx(self, player):
        return self.ori_idx[player]

    def get_alter_idx(self, player):
        return self.alter_idx[player]

    def init_act_idx(self):
        # each agent has 2 actions 0,1
        # Initialize original actions idx lists and alternative actions idx lists.
        # Store as multi-dimensional lists
        # Can be called by index that represents agents
        all_joint_acts = [list(i) for i in self.joint_acts()
                          ]  # get all joint actions as a list to read index
        ## Initialize original and alternative action index lists
        self.ori_idx = [[[], [],[], []] for _ in range(self.num_agents)
                        ]  # 2 actions * n agents
        # 2 alter actions * 3 actions * n agents

        ## Initialize ori_idx list
        for i in range(len(all_joint_acts)):
            joint_act = all_joint_acts[
                i]  # joint_act should be a list with n elements represents actions
            for j in range(len(joint_act)):  # agent j
                act = joint_act[j]  # action of agent j
                self.ori_idx[j][act].append(
                    i)  # in joint_action i, agent j perform action act
        ## Initialize alter_idx list
        self.alter_idx = [[
            copy.deepcopy(self.ori_idx[i]),
            copy.deepcopy(self.ori_idx[i]),
            copy.deepcopy(self.ori_idx[i]),
            copy.deepcopy(self.ori_idx[i])
        ] for i in range(self.num_agents)]
        for i in range(len(all_joint_acts)):
            joint_act = all_joint_acts[
                i]  # joint_act should be a list with n elements represents actions
            for j in range(len(joint_act)):  # agent j
                act = joint_act[j]  # action of agent j
                for grp in self.alter_idx[j][act]:
                    if i in grp:
                        grp.remove(i)
        ## Remove empty list
        for agent_l in self.alter_idx:
            for ori_act_l in agent_l:
                for alter_act_l in ori_act_l:
                    if len(alter_act_l) == 0:
                        ori_act_l.remove(alter_act_l)

        return 1

    def eval_reset(self, agent_init_pos):
        return self.reset()

    def get_all_states(self):
        if self.all_states != None:
            return self.all_states
        else:
            all_states = []
            for food in range(self.max_food+1):
                
                for wood in range(self.max_wood+1):
                    
                    for health in product([_ for _ in range(self.max_health+1)],repeat=self.num_agents):
                        state = []
                        state.append(food)
                        state.append(wood)
                        health = list(health)
                        state+= health
                        all_states.append(state)
            return all_states

if __name__ == '__main__':
    env = Survive()
    env.reset()
    with open('survive_all_states.npy','wb') as f:
        np.save(f,env.all_states)
    
    obs, reward, done, info = None, None, None, None
    for step in range(100):
        act = np.random.randint(0,4,env.num_agents)
        obs, reward, done, info = env.step(act)
    