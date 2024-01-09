from os import name
import numpy as np
import gym
from gym import spaces
import copy
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
     
    def __init__(self, read_file = True, transition_mat_path = "./3agents_onedim_tran_mat.pkl", n_agents=3, n_food = 1):
        self.ori_idx = None
        self.alter_idx = None
        self.possible_acts = [0,1,2,3]
        self.action_space = 3 # -1,0,1 for each agent
        self.observation_space = n_agents + n_food # Position of agents, position of foods
        self.num_agents = n_agents
        self.num_foods = n_food
        self.all_states = self.set_all_states()
        self.joint_act_nums = np.power(self.action_space,self.num_agents)
        self.all_joint_acts = self.set_all_jacts()
        self.scores = np.zeros(self.num_agents)
        
        self.dangerous_state_ids = None
        self.state = np.zeros(self.observation_space)
        self.init_act_idx()
        self.possible_states = np.power(4,self.num_agents)*np.power(4,self.num_foods)  # state_number
        self.bad_states = self.update_dangerous_states()
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
            if agent_new_pos[ap] > 3:  # Fall
                agent_new_pos[ap] = 3
            if agent_new_pos[ap] < 0:
                agent_new_pos[ap] = 0
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
                food_old_pos[i] = np.random.randint(0,4)  # Generate new food
        self.state = np.append(agent_new_pos, food_old_pos)
        if max(self.scores) >= 10:
            done = True
        return self.state, reward, done, {}

    def set_all_states(self):
        states = product([0,1,2,3],repeat=(self.num_agents+self.num_foods))
        return [list(_) for _ in states]

    def get_all_states(self):
        return self.all_states

    def set_all_jacts(self):
        jacts = product([-1,0,1],repeat=self.num_agents)
        return [list(_) for _ in jacts]

    def get_joint_acts(self):
        return self.all_joint_acts

    def initialize_trans_prob_dict_ssa(self):
        #Initialize the transition probabilities among states
        result_mat = []
        for from_state in self.all_states():  # state vector form
            row_result = []
            for to_state in self.all_states():
                act_result = []
                for j_act in self.get_joint_acts():
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, to, action

    def initialize_trans_prob_dict_sas(self):
        result_mat = []
        for from_state in self.all_states:  # state vector form
            row_result = []
            for j_act in self.get_joint_acts(): # action
                act_result = []
                for to_state in self.all_states:
                    prob = self.transition_prob(np.array(from_state), np.array(to_state), np.array(j_act))
                    act_result.append(prob)
                row_result.append(act_result)
            result_mat.append(row_result)
        return result_mat  # form is from, aciton, to

    def transition_prob(self, s_from, s_to, act):
        possibility = 1 
        agent_old_pos = s_from[:self.num_agents]
        agent_to_pos = s_to[:self.num_agents]
        agent_new_pos = agent_old_pos + act
        for ap in range(len(agent_new_pos)):
            if agent_new_pos[ap] > 3:  # Fall
                agent_new_pos[ap] = 3
            elif agent_new_pos[ap] < 0:
                agent_new_pos[ap] = 0
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
                        possibility *= 1/4
                return possibility
    def reset(self):
        self.scores = np.zeros(self.num_agents)
        agent_pos = np.random.randint(0,4,self.num_agents)
        food_pos = np.random.randint(0,4,self.num_foods)
        self.state = np.append(agent_pos, food_pos)
        #self.test_transition_matrix()
        return self.state
    
    def test_transition_matrix(self):
        for state in self.transition_matrix:
            for action in state:
                out_prob = sum(action)
                if out_prob != 1:
                    print(out_prob)

    def is_state_danger(self, state_vec):
        agent_pos = state_vec[:self.num_agents]
        if (sum(agent_pos) == 0) or (sum(agent_pos) == 12):
            return True
        else:
            return False

    def vec_to_ind(self, state):
        ## Transfer a state vector to row_index of pi
        return self.all_states.index(list(state))

    def ind_to_vec(self, state_id):
        ## Input: id of state
        ## Output: state vector
        return self.all_states[state_id]


    def get_ori_idx(self, player):
        return self.ori_idx[player]

    def get_alter_idx(self, player):
        return self.alter_idx[player]

    def init_act_idx(self):
        # each agent has 2 actions 0,1
        # Initialize original actions idx lists and alternative actions idx lists.
        # Store as multi-dimensional lists
        # Can be called by index that represents agents
        act_num = self.action_space
        all_joint_acts = [list(i) for i in self.get_joint_acts()
                          ]  # get all joint actions as a list to read index
        ## Initialize original and alternative action index lists
        self.ori_idx = [[[] for _ in range(act_num)] for _ in range(self.num_agents)
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
            copy.deepcopy(self.ori_idx[i]) for _ in range(act_num)
            
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

    def jointact_to_ind(self, jointact):
        # Input: joint action pairs
        # Output: index of this action pair
        return self.all_joint_acts.index(jointact)

    def ind_to_act(self, act_ind):
        # Input: index of act
        # Output: vector of act
        return self.all_joint_acts[act_ind]

    def render(self, mode='human'):
        return None
    
    def update_dangerous_states(self):
        ##[-2,-2,***];[-2,-1,**]
        bad_ids = set()
        for i in range(len(self.all_states)):
            state = self.all_states[i]
            if self.is_state_danger(state):
                bad_ids.add(i)
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
