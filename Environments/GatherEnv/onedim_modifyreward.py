from os import name
import numpy as np
import gym
from gym import spaces
import pickle as pkl
from itertools import product

class OneDimGather(gym.Env):
    """
    A one dimension environment, discrete space and action.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self, read_file = True, transition_mat_path = "./onedim_tran_mat.pkl", n_agents=2, n_food = 2):
        self.possible_acts = [-1,0,1]
        self.action_space = 3 # -1,0,1 for each agent
        self.joint_act_nums = 9
        self.observation_space = n_agents + n_food # Position of agents, position of foods
        self.num_agents = n_agents
        self.scores = np.zeros(self.num_agents)
        self.num_foods = n_food
        self.dangerous_state_ids = None
        self.state = np.zeros(self.observation_space)
        self.bad_states = self.update_dangerous_states()
        self.possible_states = np.power(3,self.num_agents)*np.power(3,self.num_foods)  # state_number
        if read_file:
            self.transition_matrix = pkl.load(open(transition_mat_path, 'rb'))
        else:
            self.transition_matrix = self.initialize_trans_prob_dict_sas()
            pkl.dump(self.transition_matrix, open(transition_mat_path,'wb'))
        pass
    
    def shape_policy(self):
        # This function only help policy matrix find state-action location with state vector
        acts = [-1,0,1]
        obs_min = np.repeat(-1,self.num_agents+self.num_foods)  # min_value for obs_vector
        return self.action_space, self.observation_space, self.possible_states, obs_min, acts

    def step(self, actions):
        reward = np.zeros(self.num_agents)
        done = False
        agent_old_pos = self.state[:self.num_agents]
        agent_new_pos = agent_old_pos + actions
        for ap in range(len(agent_new_pos)):
            if abs(agent_new_pos[ap]) > 1:  # Fall
                agent_new_pos[ap] = 1
            elif abs(agent_new_pos[ap]) < -1:
                agent_new_pos[ap] = -1
        # Assign penalized reward
        if sum(agent_new_pos) == 2 or sum(agent_new_pos) == -2:
            reward[0]-=0.1
            reward[1]-=0.1
        food_old_pos = self.state[self.num_agents:]
        # Assign award below
        for i in range(len(food_old_pos)):
            food_pos = food_old_pos[i]
            candidates_agent = np.arange(self.num_agents)
            agents_eat = candidates_agent[agent_new_pos == food_pos]
            if len(agents_eat) > 0:
                if len(agents_eat) > 1:  # More than one agent eat the food
                    agent_eat = np.random.choice(agents_eat)
                else:
                    agent_eat = agents_eat[0]
                reward[agent_eat] += 1  # Assign reward
                self.scores[agent_eat] += 1
                food_old_pos[i] = np.random.randint(-1,2)  # Generate new food
        self.state = np.append(agent_new_pos, food_old_pos)
        if max(self.scores) >= 10:
            done = True
        return self.state, reward, done, {}

    def all_states(self):
        return product([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1])
    def get_all_states(self):
        return product([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1])
    def joint_acts(self):
        return product([-1,0,1],[-1,0,1])

    def initialize_trans_prob_dict_ssa(self):
        #Initialize the transition probabilities among states
        result_mat = []
        for from_state in self.all_states():  # state vector form
            row_result = []
            for to_state in self.all_states():
                act_result = []
                for j_act in self.joint_acts():
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, to, action

    def initialize_trans_prob_dict_sas(self):
        result_mat = []
        for from_state in self.all_states():  # state vector form
            row_result = []
            for j_act in self.joint_acts(): # action
                act_result = []
                for to_state in self.all_states():
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, aciton, to

    def transition_prob(self, s_from, s_to, act):
        possibility = 1 
        agent_old_pos = s_from[:self.num_agents]
        agent_to_pos = s_to[:self.num_agents]
        agent_pos_dif = agent_to_pos - agent_old_pos
        if sum(agent_pos_dif>1) > 0:  # impossible
            return 0
        else:
            agent_new_pos = agent_old_pos + act
            for ap in range(len(agent_new_pos)):
                if abs(agent_new_pos[ap]) > 1:  # Fall
                    agent_new_pos[ap] = 1
                elif abs(agent_new_pos[ap]) < -1:
                    agent_new_pos[ap] = -1
            if np.any(agent_new_pos != agent_to_pos):  # agent impossible
                return 0
            else:  # agent pos possible
                food_old_pos = s_from[self.num_agents:]
                food_eaten = sum([food_old_pos == ap for ap in agent_new_pos])
                if sum(food_eaten) == 0:  # agent cant eat food
                    if sum(food_old_pos != s_to[self.num_agents:]) == 0:  # identical food position
                        return 1
                    else:
                        return 0
                else:  # agent can eat food
                    for fi in range(len(food_eaten)):
                        if food_eaten[fi] == 0: # food[fi] not eaten
                            if food_old_pos[fi] != s_to[self.num_agents+fi]:
                                return 0
                        else:  # food[fi] eaten
                            possibility *= 1/3
                    return possibility
    def reset(self):
        self.scores = np.zeros(self.num_agents)
        agent_pos = np.random.randint(-1,2,self.num_agents)
        food_pos = np.random.randint(-1,2,self.num_foods)
        self.state = np.append(agent_pos, food_pos)
        return self.state
        
    def vec_to_ind(self, state):
        ## Transfer a state vector to row_index of pi
        diff = state - [-1,-1,-1,-1]
        state_len = 4
        to_id = 0
        for i in range(state_len):
            to_id += np.power(3, 3 - i) * diff[i]
        if to_id > 80:
            print("Hi")
        return int(to_id)
        
    def trans_map(self, cint):
        if cint<0:
            return None
        elif cint<10:
            return str(cint)
        else:
            return chr(cint-10+65)

    def ten_any(self, n, origin):
        res = ''
        while origin:
            res = self.trans_map(origin % n)+','+res
            origin = origin // n
        return res

    def ind_to_vec(self, state_id):
        if state_id == 0:
            result = np.array([-1,-1,-1,-1])
        else:
            to_five = self.ten_any(n=3, origin=state_id).split(',')
            to_five.remove('')
            to_five = np.int32(to_five)
            if len(to_five) < 4:
                to_five = np.append(np.zeros(4-len(to_five)), to_five)
            result = np.array([-1,-1,-1,-1]) + to_five
        return result

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
        diff = jointact - np.array([-1, -1])
        to_id = 0
        act_len = len(diff)
        for i in range(act_len):
            to_id += np.power(self.action_space, act_len - 1 - i) * diff[i]
        return int(to_id)

    def ind_to_act(self, act_ind):
        if act_ind == 0:
            result=np.array([-1,-1])
        else:
            to_three = self.ten_any(n=3, origin=act_ind).split(',')
            to_three.remove('')
            to_three = np.int32(to_three)
            if len(to_three) < 2:
                to_three = np.append(np.zeros(2-len(to_three)), to_three)
            result = np.array([-1,-1]) + to_three
        return result

    def render(self, mode='human'):
        return None
    
    def update_dangerous_states(self):
        ##[-1,-1,*,*];[1,1,*,*]
        bad_ids = set()
        food_pos = product([-1,0,1],[-1,0,1])
        for fp in food_pos:
            bad_ids.add(self.vec_to_ind(np.append([-1,-1],fp)))
            bad_ids.add(self.vec_to_ind(np.append([1,1],fp)))
        return bad_ids

    def close(self):
        return None

    

if __name__ == '__main__':
    env = OneDimGather()
    obs = env.reset()
    done = False
    k = 0
    for states in env.transition_matrix:
        k += 1
        print("transition prob for state %d is %f " % (k, np.sum(states)))
    while done == False:
        actions = np.random.randint(low=-1,high=2,size=env.num_agents)
        obs, reward, done, _ = env.step(actions)
        #print(reward)
    print("hello")
