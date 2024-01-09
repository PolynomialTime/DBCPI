from audioop import avg
from os import name
import numpy as np
import gym
from gym import spaces
import pickle as pkl
from itertools import product

class Lever(gym.Env):
    """
    A one dimension environment, discrete space and action.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self, read_file = False, transition_mat_path = "./onedim_tran_mat.pkl", n_agents=2, n_food = 1, mod_rew=False):
        self.possible_acts = [-1,0,1]
        self.env_name='OneDim'
        self.modrew = mod_rew
        self.action_space = 3 # -1,0,1 for each agent
        self.joint_act_nums = 9
        self.punish_reward = -1
        self.positive_reward = 10
        self.observation_space = n_agents + n_food # Position of agents, position of foods
        self.num_agents = n_agents
        self.scores = np.zeros(self.num_agents)
        self.num_foods = n_food
        self.all_states = self.init_all_states()
        self.joint_acts = self.init_joint_acts()
        self.dangerous_state_ids = None
        self.state = np.zeros(self.observation_space)
        self.bad_states = self.update_dangerous_states()
        self.possible_states = np.power(3,self.num_agents)*np.power(3,self.num_foods)  # state_number
        
        self.reward_table = self.init_reward_table()
        if read_file:
            self.transition_matrix = pkl.load(open(transition_mat_path, 'rb'))
        else:
            self.transition_matrix = self.initialize_trans_prob_dict_sas()
            pkl.dump(self.transition_matrix, open(transition_mat_path,'wb'))
        self.test_transition_matrix()
        pass
    
    def shape_policy(self):
        # This function only help policy matrix find state-action location with state vector
        acts = [-1,0,1]
        obs_min = np.repeat(-1,self.num_agents+self.num_foods)  # min_value for obs_vector
        return self.action_space, self.observation_space, self.possible_states, obs_min, acts

    def step(self, actions):
        reward = self.cal_reward(self.state,actions)
        done = False
        agent_old_pos = self.state[:self.num_agents]
        agent_new_pos = agent_old_pos + actions
        food_positions = self.state[self.num_agents:]
        for ap in range(len(agent_new_pos)):
            if agent_new_pos[ap] > 1:  # Fall
                agent_new_pos[ap] = 1
            elif agent_new_pos[ap] < -1:
                agent_new_pos[ap] = -1
        
        # Generate new foods
        for i in range(len(food_positions)):
            food_positions[i] = np.random.randint(-1,2)  # Generate new food
        self.state = np.append(agent_new_pos, food_positions)
        return self.state, reward, done, {}
    def is_state_danger(self,state):
        ap = state[:self.num_agents]
        if abs(sum(ap)) == 2:
            return True
        else:
            return False
    def init_all_states(self):
        l_all_states = []
        for s in product([-1,0,1],repeat=(self.num_agents+self.num_foods)):
            l_all_states.append(list(s))
        return l_all_states
    def get_all_states(self):
        return self.all_states
    def init_joint_acts(self):
        l_acts = []
        for a in product([-1,0,1],repeat=self.num_agents):
            l_acts.append(list(a))
        return l_acts

    def initialize_trans_prob_dict_ssa(self):
        #Initialize the transition probabilities among states
        result_mat = []
        for from_state in self.all_states:  # state vector form
            row_result = []
            for to_state in self.all_states:
                act_result = []
                for j_act in self.joint_acts:
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, to, action

    def initialize_trans_prob_dict_sas(self):
        result_mat = []
        for from_state in self.all_states:  # state vector form
            row_result = []
            for j_act in self.joint_acts: # action
                act_result = []
                for to_state in self.all_states:
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, aciton, to

    def transition_prob(self, s_from, s_to, act):
        """
        Check agent position, if same, then 1/3
        """
        agent_old_pos = s_from[:self.num_agents]
        agent_new_pos = agent_old_pos + act
        for ap in range(len(agent_new_pos)):
            if agent_new_pos[ap] > 1:  # Fall
                agent_new_pos[ap] = 1
            elif agent_new_pos[ap] < -1:
                agent_new_pos[ap] = -1
        agent_to_pos = s_to[:self.num_agents]
        if all(agent_to_pos == agent_new_pos):
            return 1/3
        else:
            return 0
    def reset(self):
        self.scores = np.zeros(self.num_agents)
        agent_pos = np.random.randint(-1,2,self.num_agents)
        food_pos = np.random.randint(-1,2,self.num_foods)
        self.state = np.append(agent_pos, food_pos)
        #self.test_transition_matrix()
        return self.state
    def eval_reset(self,foo_var):
        self.scores = np.zeros(self.num_agents)
        agent_pos = np.random.randint(-1,2,self.num_agents)
        food_pos = np.random.randint(-1,2,self.num_foods)
        self.state = np.append(agent_pos, food_pos)
        #self.test_transition_matrix()
        return self.state
    def test_transition_matrix(self):
        all_v = []
        for state in range(len(self.transition_matrix)):
            for action in range(len(self.transition_matrix[state])):
                out_prob = sum(self.transition_matrix[state][action])
                all_v.append(out_prob)
                if int(out_prob)!= 1:
                    print(out_prob)
        print(min(all_v))
        return all_v
        
    def vec_to_ind(self, state):
        ## Transfer a state vector to row_index of pi
        return self.all_states.index(list(state))
        

    def ind_to_vec(self, state_id):
        return self.all_states[state_id]

    def get_ori_idx(self, player):
        if player == 0:
            return [[0,1,2],[3,4,5],[6,7,8]]
        elif player == 1:
            return [[0,3,6],[1,4,7],[2,5,8]]
        else:
            return None

    def get_alter_idx(self, player):
        if player == 0:
            return [[[3,4,5],[6,7,8]],[[0,1,2],[6,7,8]],[[0,1,2],[3,4,5]]]
        elif player == 1:
            return [[[1,4,7],[2,5,8]],[[0,3,6],[2,5,8]],[[0,3,6],[1,4,7]]]
        else:
            return None

    def jointact_to_ind(self, jointact):
        return self.joint_acts.index(jointact)

    def ind_to_act(self, act_ind):
        return self.joint_acts[act_ind]

    def render(self, mode='human'):
        return None
    
    def update_dangerous_states(self):
        ##[-2,-2,***];[-2,-1,**]
        bad_ids = set()
        food_pos = product([-1,0,1],repeat=self.num_foods)
        for fp in food_pos:
            bad_ids.add(self.vec_to_ind(np.append([-1,-1],fp)))
            bad_ids.add(self.vec_to_ind(np.append([1,1],fp)))
        return bad_ids

    def close(self):
        return None
    def cal_reward(self,state,action):
        reward = np.zeros(self.num_agents)
        agent_old_pos = state[:self.num_agents]
        agent_new_pos = np.array(agent_old_pos) + np.array(action)
        for ap in range(len(agent_new_pos)):
            if agent_new_pos[ap] > 1:  # Fall
                agent_new_pos[ap] = 1
            elif agent_new_pos[ap] < -1:
                agent_new_pos[ap] = -1
        food_old_pos = state[self.num_agents:]
        for fp in food_old_pos:
            agent_get_food = np.zeros(self.num_agents)
            for i in range(self.num_agents):
                if fp == agent_new_pos[i]:
                    agent_get_food[i] = 1
            if sum(agent_get_food) == 2:
                pass
            elif sum(agent_get_food) == 0:
                reward = np.array([0.8,0.8])
            else:
                if agent_get_food[0] == 1:
                    reward[0] = 1
                    reward[1] = 0.2
                else:
                    reward[1] = 1
                    reward[0] = 0.2
        if self.modrew:
            if self.is_state_danger(state):
                reward += self.punish_reward
        return reward

    def init_reward_table(self):
        rewards = np.zeros(shape=(self.possible_states,self.joint_act_nums,self.num_agents))
        s_num = 0
        for state in self.all_states:
            a_num = 0
            for act in self.joint_acts:
                rewards[s_num][a_num] = self.cal_reward(state,act)
                a_num += 1
            s_num += 1
        return rewards
if __name__ == '__main__':
    env = Lever()
    obs = env.reset()
    done = False
    k = 0
    for t in range(1000):
        actions = np.random.randint(low=-1,high=2,size=env.num_agents)
        obs, reward, done, _ = env.step(actions)
        #print(reward)
    print("hello")
