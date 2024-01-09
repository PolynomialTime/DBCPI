import numpy as np
import gym
from gym import spaces
import pickle as pkl
import copy
from itertools import product

class Hunt(gym.Env):
    """
    Health for each agent
    """
    def __init__(self, n_agents=3, mod_rew=False,normalize_obs=True) -> None:
        super().__init__()
        self.punishment_reward = 1.5
        self.all_states = None
        self.env_name = 'Hunt'
        self.num_agents = n_agents
        self.hunt_reward = 1
        self.default_reward = 0
        #self.collect_reward = 1
        self.action_space = 2  # 2 movements
        self.joint_act_nums = np.power(self.action_space,self.num_agents)
        # observation space should be agent location + package location + agent energy
        self.observation_space = self.num_agents # 3 possible state for each agent
        self.s = np.zeros(self.observation_space)  # Store the current location of agents
        self.mod_rew=mod_rew
        self.normalize_obs=normalize_obs
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
        ## For each agent, action: 0 for Collecting 1 for Prepare, 2 for Explore
        done = False
        reward_val = self.default_reward * self.num_agents  # Shared reward for all agents
        visit_sequence = np.arange(self.num_agents)
        new_state = list(act)
        np.random.shuffle(visit_sequence)

        reward = self.cal_reward(self.s,act)
        self.s = new_state
        
        obs = new_state
        return obs, reward, done, {}

    def reset(self):
        self.current_explore = 0
        #self.s = [0 for _ in range(self.num_agents)]
        s_id = np.random.choice(list(range(self.possible_states)))
        self.s = self.all_states[s_id]
        return self.s


    def is_state_danger(self, state):
        return (np.sum(state) >= (self.num_agents-1))

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
        self.ori_idx = [[[] for _ in range(self.action_space)] for _ in range(self.num_agents)
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
            copy.deepcopy(self.ori_idx[i]) for _ in range(self.action_space)
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
        self.all_states = all_states
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
        if list(joint_act) == list(to_state):
            return 1
        else:
            return 0

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
        reward = np.zeros(self.num_agents)
        if self.is_state_danger(state):
            reward -= 3
        if self.mod_rew:
            if self.is_state_danger(state):
                reward -= self.punishment_reward
        for i in range(self.num_agents):
            if state[i] == 0:  # In
                if action[i] == 0:
                    reward += 0.5
                elif action[i] == 1:
                    reward[i]+=self.hunt_reward
                    reward += 0.1
            elif state[i] == 1:  # Out
                if action[i] == 0:
                    reward += 0.5
                elif action[i] == 1:
                    reward -= 0.5
                    reward[i] += 0.5
        return reward
    def init_reward_tables(self):
        rewards = np.zeros(shape=(self.possible_states,self.joint_act_nums,self.num_agents))
        rewards += self.default_reward*self.num_agents
        s_num = 0
        for state in self.get_all_states():
            a_num = 0
            for act in self.joint_acts():
                rewards[s_num][a_num] += self.cal_reward(state,act)
                a_num += 1
            s_num += 1
        return rewards

    def init_balance_states(self):
        """
        return 2 lists of states, for balancing target
        """
        s_0 = []
        s_1 = []
        for i in range(self.possible_states):
            s = self.all_states[i]
            if s[0] == 1:  # agent 0 is out
                s_0.append(i)
            if s[1] == 1:  # agent 1 is out
                s_1.append(i)
        return np.array([np.array(s_0),np.array(s_1)])

    def init_balance_states2(self):
        """
        return 2 lists of states, for balancing target
        """
        s_0 = []
        s_1 = []
        for i in range(self.possible_states):
            s = self.all_states[i]
            if s[0] == 1:  # agent 0 is out
                s_0.append(i)
            if s[1] == 1 or s[2] == 1:  # agent 1 is out
                s_1.append(i)
        return np.array([np.array(s_0),np.array(s_1)])
if __name__ == '__main__':
    env = Hunt()
    env.reset()
    #env.check_transition_prob()
    obs, reward, done, info = None, None, None, None
    for step in range(100):
        act = np.random.randint(0,2,env.num_agents)
        #act = np.repeat(0,env.num_agents)
        obs, reward, done, info = env.step(act)
    