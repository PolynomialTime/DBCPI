from linecache import cache
from ntpath import join
import numpy as np
import gym
from gym import spaces
import pickle as pkl
import copy
from itertools import product

class CaE(gym.Env):
    """
    Health for each agent
    """
    def __init__(self, n_agents=3,mod_rew=False) -> None:
        super().__init__()
        self.punishment_reward = 0.5
        self.all_states = None
        self.env_name = 'CaE'
        self.num_agents = n_agents
        self.explore_reward = 1
        self.farm_reward = 0.3
        #self.neg_reward = -0.1
        self.action_space = 2  # 2 movements
        self.joint_act_nums = np.power(self.action_space,self.num_agents)
        # observation space should be agent location + package location + agent energy
        self.observation_space = self.num_agents # 2 possible state for each agent
        self.s = np.zeros(self.observation_space)  # Store the current location of agents
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
    def step(self, act):
        ## For each agent, action: 0 for farming 1 for Prepare, 2 for Explore
        is_exploring = False
        done = False
        #reward_val = self.neg_reward * self.num_agents  # Shared reward for all agents
        visit_sequence = np.arange(self.num_agents)
        new_state = copy.copy(self.s)
        np.random.shuffle(visit_sequence)
        reward = self.cal_reward(self.s,act)
        for i in visit_sequence:
            if self.s[i] == 0: # Inside
                if act[i] == 0: # farm
                    new_state[i] = 0
                elif act[i] == 1 and (not is_exploring): # Explore from normal, go to prepared state
                    new_state[i] = 1
                    is_exploring = 1
                elif act[i] == 1 and is_exploring:
                    new_state[i] = 0
            elif self.s[i] == 1: # Outside
                new_state[i] = 0
            
        self.s = new_state
        
        #reward = np.repeat(reward_val, self.num_agents)
        obs = new_state
        return obs, reward, done, {}

    def reset(self):
        self.current_explore = 0
        s_id = np.random.choice(list(range(self.possible_states)))
        self.s = self.all_states[s_id]
        return self.s


    def is_state_danger(self, state):
        #return (2 in state)
        return state[0] == 1  # First agent is outside

    def joint_acts(self):
        return product([0, 1], repeat=self.num_agents)  # Top, Right, Down, Left

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
        self.ori_idx = [[[], []] for _ in range(self.num_agents)
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
            for n in product([0,1],repeat=self.num_agents):
                all_states.append(list(n))
        self.all_states = copy.deepcopy(all_states)
        for s in all_states:
            if sum(s) > 1:
                self.all_states.remove(s)
        return self.all_states


    def initialize_trans_prob_dict_sas(self):
        result_mat = []
        for from_state in self.get_all_states():  # Which state we come from
            row_result = []
            for j_act in self.joint_acts(): # action
                act_result = []
                for to_state in self.get_all_states():  # Which state to go to
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, aciton, to
    
    def transition_prob(self, from_state, to_state, joint_act):
        possibility = 1
        explore_agents = np.where(joint_act==1)[0]
        coll_agents = np.where(joint_act == 0)[0]
        comp_agents = []

        if len(coll_agents) != 0:
            for c_a in coll_agents:
                if to_state[c_a] != 0:
                    possibility = 0
                else:
                    pass
        """
        Exist explore action
        """

        if len(explore_agents) != 0:
            for id_agent in explore_agents:
                if from_state[id_agent] != 0:  # out * explore
                    if to_state[id_agent] != 0:
                        possibility = 0
                else:
                    comp_agents.append(id_agent)
        if len(comp_agents) > 0:
            n_comps = len(comp_agents)
            comp_to_states = to_state[comp_agents]
            if sum(comp_to_states) != 1:
                possibility = 0
            else:
                possibility /= n_comps
        return possibility
    def check_transition_prob(self):
        all_v = []
        for state in range(len(self.transition_matrix)):
            for action in range(len(self.transition_matrix[state])):
                out_prob = sum(self.transition_matrix[state][action])
                all_v.append(out_prob)
                if int(out_prob)!= 1:
                    print(out_prob)
                    print(self.all_states[state])
                    print(self.all_jacts[action])
                    print(self.transition_matrix[state][action])
                    return 0
        print(min(all_v)==max(all_v)==1)
        return all_v

    def vec_to_ind(self,obs):
        # Observation to id
        return self.all_states.index(obs)
    def ind_to_act(self, ind):
        return self.all_jacts[ind]
    def jointact_to_ind(self,jact):
        return self.all_jacts.index(jact)

    def init_bad_states(self):
        bad_ids = []
        for i in range(len(self.get_all_states())):
            state = self.get_all_states()[i]
            if self.is_state_danger(state):
                bad_ids.append(i)
        return bad_ids

    def cal_reward(self, state, action):
        #reward = self.neg_reward
        reward = 0
        candidates = []
        if self.mod_rew:
            if self.is_state_danger(state):
                reward -= self.punishment_reward
        for i in range(self.num_agents):
            if action[i] == 0:  # farm
                reward += self.farm_reward
            elif action[i] == 1:  # Explore
                if state[i] != 1: 
                    candidates.append(i)
        if len(candidates) >= 1:
            reward += self.explore_reward
        reward = np.repeat(reward,self.num_agents)
        return reward
    def init_reward_tables(self):
        rewards = np.zeros(shape=(self.possible_states,self.joint_act_nums,self.num_agents))
        #rewards += self.neg_reward*self.num_agents
        s_num = 0
        for state in self.get_all_states():
            a_num = 0
            for act in self.joint_acts():
                rewards[s_num][a_num] += self.cal_reward(state,act)
                a_num += 1
            s_num += 1
        return rewards

    def init_balance_states(self):
        s_0 = []
        s_1 = []
        for i in range(self.possible_states):
            s = self.all_states[i]
            if s[0] == 1:  # agent 0 is out
                s_0.append(i)
            if s[1] == 1 or s[2] == 1:  # agent 1 or 2 is out
                s_1.append(i)
        return np.array([np.array(s_0),np.array(s_1)])
if __name__ == '__main__':
    env = CaE()
    env.reset()
    #env.check_transition_prob()
    obs, reward, done, info = None, None, None, None
    for step in range(100):
        act = np.random.randint(0,2,env.num_agents)
        #act = np.repeat(0,env.num_agents)
        obs, reward, done, info = env.step(act)
    