import gym
from .fimdp import nyc_parser 
from .fimdp import NYCtools
import ast
import networkx as nx
import numpy as np
import copy
import pickle as pkl
from itertools import product


class TrafficEnv(gym.core.Env):
    ENERGY_FULL = 2
    NUM_ANGLES = 8
    MAX_RELOADS = 1000
    REWARD_WEIGHTS = 10
    SLEEP_TIME = 0.5
    ENERGY_LOW_THRES = 0
    MAX_STEP_PER_EPISODE = 200
    MAX_DISTANCE = 0.2
    def __init__(self,
                 targets=None,
                 n_agents=4,
                 energy_thres=0.4,
                 datafile='./Environments/TrafficEnv/nyc.graphml',
                 modify_reward=False):
        '''
        datafile should be fixed
        energy_stations: a list of numbers. eg: ['42431659','42430367','1061531810']
        targets: a list of numbers describes the target locations.
        n_agents: number of agents. An integer.
        '''
        G = nx.MultiDiGraph(nx.read_graphml(datafile))
        for _, _, data in G.edges(data=True, keys=False):
            data['speed_mean'] = float(data['speed_mean'])
            data['speed_sd'] = float(data['speed_sd'])
            data['time_mean'] = float(data['time_mean'])
            data['time_sd'] = float(data['time_sd'])
            data['energy_levels'] = ast.literal_eval(data['energy_levels'])
        for _, data in G.nodes(data=True):
            data['reload'] = ast.literal_eval(data['reload'])
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])

        nodes_xy = []
        nodes_latlon = []
        node_name_to_index = {}
        node_index_to_name = []
        for i, node in enumerate(G.nodes.data()):
            name = str(node[0])
            node_name_to_index[name] = i
            node_index_to_name.append(name)
            nodes_xy.append([float(node[1]['x']), float(node[1]['y'])])
            nodes_latlon.append([float(node[1]['lat']), float(node[1]['lon'])])

        # Initialize parameters
        self.modify_reward=modify_reward
        self.energy_threshold=energy_thres
        self.num_agents = n_agents
        self.agent_visit_seq = None
        self.env_name = 'Traffic'
        self.nodes_xy = np.array(nodes_xy)
        self.nodes_latlon = np.array(nodes_latlon)
        self.node_name_to_index = node_name_to_index
        self.node_index_to_name = node_index_to_name
        self.num_nodes = len(self.node_index_to_name)
        self.s = [None for _ in range(self.num_agents)]
        self.stop_early = False
        self.fig = None
        self.use_render = False

        self.nodes_xy_vertical = self._rotate_nycstreet(
            nodes_xy - np.mean(nodes_xy, axis=0, keepdims=True))
        """
        Transaction describes the road network. Length is 1024, corresponds to 1024 nodes in the map
        coor is the current node location
        next describes the out edges(we use index here), each row(node) has 1 to 4 out edges, which indicates 1 to 4 ways out(or 1 to 4 actions)
        corr_diff describes 'absolute distance' to 1 to 4 candidate locations.
        Since most nodes have 2 out edges, ({2: 751, 3: 158, 1: 106, 4: 9})
        we want to decrease the action space into 2 actions.
        """
        
        self.transitions = []
        for _ in range(self.num_nodes):
            self.transitions.append({
                'coor': None,
                'next': [],
                'coor_diff': [],
                'reload': False,
                'target': False
            })

        for edge in G.edges:
            i_from = self.node_name_to_index[edge[0]]
            i_to = self.node_name_to_index[edge[1]]
            coor_from = self.nodes_xy_vertical[i_from]
            coor_to = self.nodes_xy_vertical[i_to]
            coor_diff = coor_to - coor_from
            if self.transitions[i_from]['coor'] is not None:
                assert np.all(
                    np.equal(self.transitions[i_from]['coor'], coor_from))
            else:
                self.transitions[i_from]['coor'] = coor_from
            self.transitions[i_from]['next'].append(i_to)
            self.transitions[i_from]['coor_diff'].append(coor_diff)

        path = './Environments/TrafficEnv/NYCstreetnetwork.json'
        m, targets = nyc_parser.parse(path)
        states_reload, states_target = NYCtools.extract_data(m, targets)

        for trans in self.transitions:
            trans['next'] = np.array(trans['next'])
            trans['coor_diff'] = np.array(trans['coor_diff'])

        # Trim actions below
        for trans in self.transitions:
            if len(trans['next']) > 2:  # Too much actions
                trans['next'] = trans['next'][:2]
                trans['coor_diff'] = trans['coor_diff'][:2]
            if len(trans['next']) < 2:  # only 1 act
                trans['next'] = np.append(trans['next'], trans['next'][0])
                trans['coor_diff'] = np.array([trans['coor_diff'][0],
                                                trans['coor_diff'][0]])

        con = list(self.test_connectivity())
        max_comp = list(max(con,key=len))
        self.max_comp = max_comp
        # random_targets = np.random.choice(max_comp,100,False)
        # np.save('./Environments/TrafficEnv/target100.npy',random_targets)
        ## Trim edges that exceed our component
        for i in range(len(self.transitions)):
            if i in max_comp:
                for j in range(len(self.transitions[i]['next'])):
                    if self.transitions[i]['next'][j] not in max_comp:
                        self.transitions[i]['next'][j] = self.transitions[i]['next'][1 - j]
                        self.transitions[i]['coor_diff'][j] = self.transitions[i]['coor_diff'][1 - j]
        # We want more targets, so we randomly select 100 nodes from 1024 nodes
        states_target = np.load(
            './Environments/TrafficEnv/target100.npy').tolist()
        # If we want fewer target, we need to cut the list
        states_reload = states_reload[:int(len(states_reload)/3)]
        for i, state in enumerate(states_reload):
            if i == self.MAX_RELOADS:
                break
            self.transitions[self.node_name_to_index[state]][
                'reload'] = True  # set reload
        for i, state in enumerate(states_target):
            self.transitions[state][
                'target'] = True  # set target

        # Generate consumption below
        for trans in self.transitions:
            trans['energy_consume'] = np.linalg.norm(trans['coor_diff'],
                                                        axis=1)
        
        self.transitions_bak = copy.deepcopy(self.transitions)
        self.coor_to_reload_index = {}
        i = 0
        for trans in self.transitions:
            if trans['reload']:
                x, y = trans['coor']
                self.coor_to_reload_index[str((x, y))] = i
                i = i + 1
        self.num_reloads = len(self.coor_to_reload_index)

        self.not_target = [
        ]  # Used to reset environment. Agents should not born at target stations
        for i in range(self.num_nodes):
            if i in max_comp:
                self.not_target.append(int(not self.transitions[i]['target']))
            else:
                self.not_target.append(0)

        for i in self.not_target:
            if i != 0:
                if i not in max_comp:
                    print("WRONG")
        self.not_target = self.not_target / np.sum(self.not_target)

        
        ## Initialize settings
        self.observation_space = 2*self.num_agents + self.num_agents  # Observe the x-y-coordinate of agents and all energy levels
        self.action_space = 2
        self.joint_act_nums = np.power(2,self.num_agents)
        self.ori_idx = None
        self.alter_idx = None
        self.init_act_idx()

    def _rotate_nycstreet(self, p):
        top_index = np.argmax(p[:, 1])
        left_index = np.argmin(p[:, 0])
        diff = p[top_index] - p[left_index]
        rot = np.arctan2(diff[1], diff[0])
        delta_rot = np.pi / 2 - rot
        mat = np.array([[np.cos(delta_rot),
                         np.sin(delta_rot)],
                        [-np.sin(delta_rot),
                         np.cos(delta_rot)]])
        p_rot = p.dot(mat)
        p_rot = p_rot / (1e-5 + np.mean(np.abs(p_rot)))
        return p_rot


    def generate_init_pos(self,n_games=100):
        init_pos = []
        for game in range(n_games):
            init_pos.append(np.random.choice(np.arange(self.num_nodes), self.num_agents, p=self.not_target, replace=False))  # Randomly place agents
        init_pos = np.array(init_pos)
        return init_pos
        
    def reset(self):
        self.energy_levels = [self.ENERGY_FULL for _ in range(self.num_agents)]  # Reset energy level
        self.alives = [True for _ in range(self.num_agents)]  # Restore agent's life
        self.transitions = copy.deepcopy(self.transitions_bak)  # Restore targets
        self.s = np.random.choice(np.arange(self.num_nodes), self.num_agents, p=self.not_target, replace=False)  # Randomly place agents
        self.agent_visit_seq = np.arange(4)
        #observation_space = agent_location + energy_station_location + target_location + energy_level
        ## Initialize observations below
        pos_agent = np.array([self.nodes_xy_vertical[self.s[i]] for i in range(self.num_agents)])
        obs = np.concatenate([pos_agent.flatten(),self.energy_levels])
        
        return obs

    def step(self, actions):
        reward = [-0.01 for _ in range(self.num_agents)]
        done = False
        np.random.shuffle(self.agent_visit_seq)  # Randomize sequence to assign reward fairly
        for i in self.agent_visit_seq:
            if self.alives[i]:  # Still alive
                action_i = actions[i]  # The action of agent i, from [0,1]
                next_s = self.transitions[self.s[i]]['next'][action_i]
                energy_cons = self.transitions[self.s[i]]['energy_consume'][action_i]
                self.energy_levels[i] -= energy_cons
                if self.transitions[next_s]['target'] == True:  # Assign target reward
                    reward[i] += 1
                    self.transitions[next_s]['target'] = False
                if self.transitions[next_s]['reload'] == True:  # Reload to full or reload half
                    reward[i] -= 0.2
                    self.energy_levels[i] = min(self.ENERGY_FULL, self.energy_levels[i]+self.ENERGY_FULL/2)
                if self.energy_levels[i] < 0:  # Agent die
                    self.alives[i] = False
                    reward[i] -=2
                self.s[i] = next_s
        if self.modify_reward:  # If modify reward
            danger = any(np.array(self.energy_levels)<self.energy_threshold)
            if danger:
                reward=[reward[i]-2.5 for i in range(self.num_agents)]
        pos_agent = np.array([self.nodes_xy_vertical[self.s[i]] for i in range(self.num_agents)])
        obs = np.concatenate([pos_agent.flatten(),self.energy_levels])
        if sum(self.alives) < 1:  # 全部死亡
            done = True
        return obs, reward, done, {}

    def render(self, mode='human'):
        return None

    def close(self):
        return None

    def is_state_danger(self, state):
        return any(np.array(state[-self.num_agents:]) < self.energy_threshold)

    def init_act_idx(self):
        # each agent has 2 actions 0,1
        # Initialize original actions idx lists and alternative actions idx lists.
        # Store as multi-dimensional lists
        # Can be called by index that represents agents
        all_joint_acts = [list(i) for i in self.joint_acts()
                          ]  # get all joint actions as a list to read index
        ## Initialize original and alternative action index lists
        all_ids = list(range(len(all_joint_acts)))
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

    def joint_acts(self):
        return product([0, 1], repeat=self.num_agents)

    def get_ori_idx(self, player):
        return self.ori_idx[player]

    def get_alter_idx(self, player):
        return self.alter_idx[player]

    def test_connectivity(self):
        graph = nx.DiGraph()
        all_nodes = list(range(1024))
        graph.add_nodes_from(all_nodes)  # Add nodes
        for i in range(len(self.transitions)):
            from_node = i
            edges = self.transitions[i]['next']
            for to_node in edges:
                graph.add_edge(from_node,to_node)
        con = nx.strongly_connected_components(graph)
        #print(con,type(con),list(con))
        return con

    def eval_reset(self, agent_init_pos):
        # Initialize based on given position
        self.energy_levels = [self.ENERGY_FULL for _ in range(self.num_agents)]  # Reset energy level
        self.alives = [True for _ in range(self.num_agents)]  # Restore agent's life
        self.transitions = copy.deepcopy(self.transitions_bak)  # Restore targets
        self.s = agent_init_pos
        self.agent_visit_seq = np.arange(4)
        pos_agent = np.array([self.nodes_xy_vertical[self.s[i]] for i in range(self.num_agents)])
        obs = np.concatenate([pos_agent.flatten(),self.energy_levels])
        return obs

if __name__ == '__main__':
    env = TrafficEnv(use_file=True)
    env.reset()
    env.test_connectivity()
    actions = np.random.randint(low=0, high=2, size=env.num_agents)
    for _ in range(500):
        actions = np.random.randint(low=0, high=2, size=env.num_agents)
        obs, rew, done, _ = env.step(actions)
        env.is_state_danger(obs)
    print("hello")
