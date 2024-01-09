from math import prod
import numpy as np
import gym
from gym import spaces
import pickle as pkl
import copy
from itertools import product

class PowerGrid(gym.Env):
    """
    3*4 grid. 2 power stations at 2 corners(left top and right bottom).
    positions from (0,0) to (3,4) as default.
    """
    def __init__(self, n_agents=2, n_packages=1,x_size=3,y_size=3,init_energy=5,energy_threshold=2,mod_rew=False,transition_mat_path = "./powergrid_tran_mat.pkl",read_file = False, overwrite_file=True) -> None:
        super().__init__()
        self.x_size=x_size
        self.y_size=y_size
        self.overwrite_file=overwrite_file
        self.read_file=read_file
        self.env_name = 'PowerGrid'
        self.transition_mat_path = transition_mat_path
        self.num_agents = n_agents
        self.num_packages = n_packages
        self.action_space = 4  # 2 directions for each agent
        
        self.joint_act_nums = np.power(2,self.num_agents)
        # observation space should be agent location + package location + agent energy
        self.observation_space = 2*self.num_agents + 2*self.num_packages + self.num_agents
        self.s = [np.array([0,0]) for _ in range(self.num_agents)]  # Store the current location of agents
        self.package_s = [[0,0] for _ in range(self.num_packages)]  # Store the current location of packages
        self.ene = np.zeros(self.num_agents)
        self.max_energy=init_energy
        self.mod_rew=mod_rew
        self.energy_threshold = energy_threshold
        self.ori_idx = None
        self.alter_idx = None
        self.all_states=None
        if read_file:
            self.transition_matrix = pkl.load(open(transition_mat_path, 'rb'))
        else:
            self.transition_matrix = self.generate_transition_matrix()
            pkl.dump(self.transition_matrix, open(transition_mat_path,'wb'))
        self.get_all_states()
        self.state_num = len(self.all_states)

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
                agent_loc = self.s[agent_idx]
                agent_new_loc = agent_loc + actions[agent_idx]
                if agent_new_loc[0]<0:
                    agent_new_loc[0]=0
                    reward[agent_idx] += -0.1
                elif agent_new_loc[0]>=self.x_size:
                    agent_new_loc[0]=self.x_size-1
                    reward[agent_idx] += -0.1
                if agent_new_loc[1]<0:
                    agent_new_loc[1]=0
                    reward[agent_idx] += -0.1
                elif agent_new_loc[1]>=self.y_size:
                    agent_new_loc[1]=self.y_size-1
                    reward[agent_idx] += -0.1
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
            self.s[i] = np.array(np.random.randint(0,self.x_size),np.random.randint(0,self.y_size))
            self.ene[i] = self.max_energy
        obs = np.concatenate((np.array(self.s).flatten(), np.array(self.package_s).flatten(), self.ene))
        return obs
    
    def power_station_locs(self):
        return [[0,0], [self.x_size-1, self.y_size-1]]

    def generate_new_package(self):
        # flag = True
        # all_locs = self.power_station_locs()
        # while(flag):
        #     loc = list(np.random.randint(0,self.x_size),np.random.randint(0,self.y_size))
        #     if loc not in all_locs:  # Package should not be at power station
        #         if loc not in self.package_s:  # Package should not be placed at same location
        #             flag = False
        # return loc
        return list(np.random.randint(0,self.x_size),np.random.randint(0,self.y_size))

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
                        ]  # 4 actions * n agents
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


    def get_all_states(self):
        ## Should only be called when generating transition matrix
        if self.all_states == None:
            self.all_states=[]
            agent_possible_location = product([i for i in range(0,self.x_size)],[i for i in range(0,self.y_size)])
            package_possible_location=product([i for i in range(0,self.x_size)],[i for i in range(0,self.y_size)])
            possible_energy_levels=[i for i in range(0,self.max_energy)]
            all_energy_levels=product(possible_energy_levels,repeat=self.num_agents)
            all_agent_positions = product(agent_possible_location,repeat=self.num_agents)
            all_package_positions = product(package_possible_location,repeat=self.num_packages)
            all_states =  product(all_agent_positions,all_package_positions,all_energy_levels)
            for i in all_states:
                self.all_states.append(list(sum(i[0],()))+list(sum(i[1],()))+list(i[2]))
            #self.all_states=[list(sum(i,())) for i in all_states]  # Flatten list of tuples into a list
        return self.all_states

    def generate_transition_matrix(self):
        
        result_mat = []
        for from_state in self.get_all_states():  # state vector form
            row_result = []
            for to_state in self.get_all_states():
                act_result = []
                for j_act in self.joint_acts():
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
            #print("Time used is %f" % (time.time()-start))
        return result_mat  # form is from, to, action
    
    def transition_prob(self, from_s, to_s, j_act):
        ## Calculate the transition probability from one state to another state by an action
        prob = 1
        actions = []
        for a in j_act:
            if a == 0:
                actions.append(np.array((0,1)))  # Top
            elif a == 1:
                actions.append(np.array((1,0)))  # Right
            elif a==2:
                actions.append(np.array((0,-1)))  # Down
            elif a == 3:
                actions.append(np.array((-1,0)))  # Left
        agent_old_s = np.array(from_s[:2*self.num_agents]).reshape(-1,2)
        agent_new_s = np.array(to_s[:2*self.num_agents]).reshape(-1,2)
        agent_old_e = list(from_s[2*self.num_agents+2*self.num_packages:])
        agent_new_e = list(to_s[2*self.num_agents+2*self.num_packages:])
        package_old_s = list(from_s[2*self.num_agents:2*self.num_agents+2*self.num_packages].reshape(-1,2).tolist())
        package_new_s = list(to_s[2*self.num_agents:2*self.num_agents+2*self.num_packages].reshape(-1,2).tolist())

        ## First step: Check the agent locations.
        for i in range(self.num_agents):  # Impossible movement
            agent_move_loc = agent_old_s[i]+np.array(actions[i])
            if agent_move_loc[0]<0:
                agent_move_loc[0]=0
            elif agent_move_loc[0]>=self.x_size:
                agent_move_loc[0]=self.x_size=1
            if agent_move_loc[1]<0:
                agent_move_loc[1]=0
            elif agent_move_loc[1]>=self.y_size:
                agent_move_loc[1]=self.y_size-1
            if any(agent_move_loc != agent_new_s[i]):
                return 0

        ## Second step: Check the agent energy levels.
        for i in range(self.num_agents):
            agent_loc = agent_new_s[i]
            if list(agent_loc) not in self.power_station_locs():  # agent i doesnot go to power station
                if agent_old_e[i]-1 != agent_new_e[i]:
                    return 0
            else:  # agent i go to power station
                if agent_new_e[i] != self.max_energy:
                    return 0

        ## Third step: Check the packages
        picked_packages = []
        for i in range(self.num_agents):
            agent_loc = list(agent_new_s[i])
            if agent_loc in list(package_old_s):  # Agent can pick a package
                package_picked_idx = package_old_s.index(agent_loc)
                picked_packages.append(package_picked_idx)
        picked_packages = list(set(picked_packages))  # Remove duplicated indices
        for i in range(self.num_packages):
            if i in picked_packages:  # picked, should be replaced
                prob = prob / (self.x_size*self.y_size)
            else:  # Not picked, should be at same location
                if package_new_s[i] != package_old_s[i]:
                    return 0
        return prob