import numpy as np
import gym
from gym import spaces
import pickle as pkl
import copy
from itertools import product

class PowerGrid(gym.Env):
    """
    5*5 grid. 4 power stations at 4 corners.
    positions from (0,0) to (4,4) as default.
    """
    def __init__(self, n_agents=4, n_packages=4,size=4,init_energy=10,energy_threshold=4,mod_rew=False,normalize_obs=True) -> None:
        super().__init__()
        self.size=size
        self.env_name = 'PowerGrid'
        self.num_agents = n_agents
        self.num_packages = n_packages
        self.action_space = 4  # 4 movements
        self.joint_act_nums = np.power(4,self.num_agents)
        # observation space should be agent location + package location + agent energy
        self.observation_space = 2*self.num_agents + 2*self.num_packages + self.num_agents
        self.s = [np.array([0,0]) for _ in range(self.num_agents)]  # Store the current location of agents
        self.package_s = [[0,0] for _ in range(self.num_packages)]  # Store the current location of packages
        self.ene = np.zeros(self.num_agents)
        self.max_energy=init_energy
        self.mod_rew=mod_rew
        self.energy_threshold = energy_threshold
        self.normalize_obs=normalize_obs
        self.ori_idx = None
        self.alter_idx = None
        self.init_act_idx()

    def step(self, act):
        actions = []
        for a in act:
            if a == 0:
                actions.append(np.array((0,1)))  # Top
            elif a == 1:
                actions.append(np.array((1,0)))  # Right
            elif a==2:
                actions.append(np.array((0,-1)))  # Down
            elif a == 3:
                actions.append(np.array((-1,0)))  # Left
        
        done=False
        reward = [-0.1 for _ in range(self.num_agents)]
        visit_sequence = np.arange(self.num_agents)
        package_visited = []
        np.random.shuffle(visit_sequence)
        for agent_idx in visit_sequence:
            if self.ene[agent_idx] > 0:
                self.ene[agent_idx] -= 1
                agent_loc = self.s[agent_idx]
                agent_new_loc = agent_loc + actions[agent_idx]
                for i in range(len(agent_new_loc)):  # agent should not get out of the map
                    if agent_new_loc[i] < 0:
                        agent_new_loc[i] = 0
                        reward[agent_idx] -= 0.1  # Negative reward for crashing on wall
                    elif agent_new_loc[i] > self.size:
                        agent_new_loc[i] = self.size
                        reward[agent_idx] -= 0.1  # Negative reward for crashing on wall
                if list(agent_new_loc) in self.package_s: # Agent pick up package
                    package_idx = self.package_s.index(list(agent_new_loc))
                    if package_idx not in package_visited:
                        package_visited.append(package_idx)
                        reward[agent_idx] += 1
                if list(agent_new_loc) in self.power_station_locs():  # Charge agent
                    self.ene[agent_idx] = self.max_energy
                self.s[agent_idx] = agent_new_loc
            else:
                pass
        if all(self.ene<=0):
            done = True
        if self.normalize_obs:
            obs = np.concatenate((np.array(self.s).flatten()/self.size, np.array(self.package_s).flatten()/self.size, self.ene/self.max_energy))
        else:
            obs = np.concatenate((np.array(self.s).flatten(), np.array(self.package_s).flatten(), self.ene))
        if self.mod_rew:
            if any(self.ene<self.energy_threshold):
                reward=[reward[i]-0.5 for i in range(self.num_agents)]
        for package_idx in package_visited:  # Generate new packages
            self.package_s[package_idx] = self.generate_new_package()

        return obs, reward, done, {}

    def reset(self):
        for i in range(self.num_packages):
            self.package_s[i] = self.generate_new_package()
        for i in range(self.num_agents):
            self.s[i] = np.random.randint(0,self.size+1,2)
            self.ene[i] = self.max_energy
        obs = np.concatenate((np.array(self.s).flatten(), np.array(self.package_s).flatten(), self.ene))
        return obs
    
    def power_station_locs(self):
        return [[0,0], [0,self.size], [self.size, self.size], [self.size,0]]

    def generate_new_package(self):
        flag = True
        all_locs = self.power_station_locs()
        while(flag):
            loc = list(np.random.randint(0,self.size+1,2))
            if loc not in all_locs:  # Package should not be at power station
                if loc not in self.package_s:  # Package should not be placed at same location
                    flag = False
        return loc

    def is_state_danger(self, state):
        energy_levels = state[2*self.num_agents+2*self.num_packages:]
        return any(energy_levels < self.energy_threshold)

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
