import torch
from Functions.Core import Transaction, Trajectory
import numpy as np

def cal_risk(occupancy_measure, environment, sample_size, sample_states):
    ## objective of our optimization problem.
    #start = time.time()
    if environment.env_name=='examples/particle_gather_multiagent.py':  # Is mujoco gather
        risk_ids = [environment.is_state_danger(s*10) for s in sample_states]
    elif environment.env_name == 'Traffic':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    elif environment.env_name == 'PowerGrid':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    elif environment.env_name == "OneDim":
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    occupancy_measure = occupancy_measure.reshape(sample_size,
                                                  environment.joint_act_nums)
    densities = torch.sum(occupancy_measure,axis=1)  # density of states
    
    risk_oms = densities[list(risk_ids)]
    result = torch.sum(risk_oms)
    #print("Cal risk time %f" % (time.time() - start))
    return result

def normalized_risk(occupancy_measure, environment, sample_size, sample_states):
    ## objective of our optimization problem.
    #start = time.time()
    if environment.env_name=='examples/particle_gather_multiagent.py':  # Is mujoco gather
        risk_ids = [environment.is_state_danger(s*10) for s in sample_states]
    elif environment.env_name == 'Traffic':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    elif environment.env_name == 'PowerGrid':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    elif environment.env_name == 'OneDim':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    occupancy_measure = occupancy_measure.reshape(sample_size,
                                                  environment.joint_act_nums)
    densities = torch.sum(occupancy_measure,axis=1)  # density of states
    densities = densities / torch.max(densities)
    risk_oms = densities[list(risk_ids)]
    result = torch.sum(risk_oms)
    #print("Cal risk time %f" % (time.time() - start))
    return result

def risk_constraint(occupancy_measure, environment, sample_size, sample_states, threshold):
    return cal_risk(occupancy_measure, environment, sample_size, sample_states) - threshold

def cal_qval(occupancy_measure, environment, qtables, sample_size, sample_states):
    if environment.env_name=='examples/particle_gather_multiagent.py':  # Is mujoco gather
        risk_ids = [environment.is_state_danger(s*10) for s in sample_states]
    elif environment.env_name == 'Traffic':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    elif environment.env_name == 'PowerGrid':
        risk_ids = [environment.is_state_danger(s) for s in sample_states]
    occupancy_measure = occupancy_measure.reshape(sample_size,
                                                  environment.joint_act_nums)
    risk_oms = occupancy_measure[list(risk_ids)]
    #result = torch.mean(risk_oms)
    om_sums = torch.sum(risk_oms,1).reshape(-1,1)
    om_policy = risk_oms / om_sums
    result = torch.tensor(0)
    #qtables = [qtable1,qtable2]
    for qtable in qtables:
        risk_qt = qtable[list(risk_ids)]
        q_mul_pi = torch.mul(om_policy, risk_qt)
        #mean_q = torch.mean(q_mul_pi,0)  # mean among all risk states
        sum_val = torch.sum(q_mul_pi)
        result = result + sum_val
    return result


def calculate_regret(occupancy_measure, state_num, joint_act_num, q_table,
                     ori_acts, alter_acts):
    """
    occupancy_measure: matrix, form s*a
    state_num: number of states
    joint_act_num: number of joint actions
    q_table: matrix for agent i, s*a
    ori_acts: original action indicies for agenti
    alter_acts: alternative indicies for agenti
    """
    #start = time.time()
    # regret = sum[ rho(s,a-i)*[q(s,a',a-i) - q(s,a,a-i)] ] for all joint actions under state s
    occupancy_measure = occupancy_measure.reshape(state_num, joint_act_num)
    regret_val = torch.tensor(np.zeros(state_num))
    for state_i in range(len(occupancy_measure)):  # For each state
        real_Qs = torch.tensor([
            q_table[state_i][act_ids] for act_ids in ori_acts
        ])  # Q-value of alter actions  # 3*3
        alter_Qs = torch.tensor(
            [q_table[state_i][act_ids] for act_ids in alter_acts])  # 3
        rhos = torch.tensor(
            [occupancy_measure[state_i][act_ids] for act_ids in ori_acts])  # 3
        regret = torch.sum(torch.mul(rhos, torch.sub(alter_Qs, real_Qs)))
        regret_val[state_i] = regret
    #print("Cal regret time %f" % (time.time() - start))
    return regret_val


def regret_constraint(occupancy_measure, state_num, joint_act_num, q_table,
                      ori_idx, alter_idx):
    results = torch.tensor([])
    for i in range(len(ori_idx)):  # For each possible action
        ori_act = ori_idx[i]
        alter_acts = alter_idx[i]  # n_acts - 1
        for possible_alter_act in alter_acts:  # How many possible alternative action pairs
            regret_val = calculate_regret(occupancy_measure, state_num,
                                          joint_act_num, q_table, ori_act,
                                          possible_alter_act)
            results = torch.cat((results, regret_val))
    return results  # Shape should be (n_act * (n_act - 1)) * n_states


def bellmanflow_constraint(occupancy_measure, state_n, jact_n, sample_states,
                           prev_s):
    #start = time.time()
    occupancy_measure = occupancy_measure.reshape(state_n, jact_n)
    ## for any s \sum_{a}(rho(s,a))=\rho_0(s) + sum_{s',a}(\gamma*P(s|s',a)*rho(s',a))
    ## For Monte-Carlo \sum_{a}(rho(s,a))=\rho_0(s) + \gamma*sum_{a}(rho(s_previous,a_previous))
    rho_0 = 1 / state_n
    bellman_flow_vals = torch.tensor(np.zeros(state_n))
    for row_id in range(len(occupancy_measure)):
        row = occupancy_measure[row_id]
        sum_om = torch.sum(row)
        prev_s_id = None
        if prev_s[row_id] is not None:
            for i in range(state_n):
                if all(sample_states[i] == prev_s[row_id]):
                    prev_s_id = i
                    break
        prev_rho = torch.sum(
            occupancy_measure[prev_s_id]) if (prev_s_id is not None) else 0
        bellman_flow_vals[row_id] = sum_om - rho_0 - prev_rho
    #print("Cal bellman_flow time %f" % (time.time() - start))
    return bellman_flow_vals  # this value should be equal to zero


def pick_action(state, om_net, env):
    all_joint_acts = [joint_act for joint_act in env.joint_acts()]
    occ_measures = np.array([
        om_net.forward(list(state) + list(jact)).detach().numpy()[0]
        for jact in all_joint_acts
    ])
    act_probs = np.divide(occ_measures, np.sum(occ_measures))
    action_id = int(
        np.random.choice(a=np.arange(len(act_probs)), size=1, p=act_probs))
    action = all_joint_acts[action_id]
    return action


def generate_history(hor, om_net, env, num_game=50):
    history = Trajectory(max_len=num_game *
                         hor)  # History generated by a specific policy
    policy_buffer = [[], []]
    all_joint_acts = [joint_act for joint_act in env.joint_acts()]
    for _ in range(num_game):  # Each game
        obs = env.reset()
        for t in range(hor):  # Each step
            if len(policy_buffer[0]) != 0 and list(obs) in policy_buffer[0]:
                act_probs = policy_buffer[1][policy_buffer[0].index(list(obs))]
            else:
                occ_measures = np.array([
                    om_net.forward(list(obs) + list(jact)).detach().numpy()[0]
                    for jact in all_joint_acts
                ])
                act_probs = np.divide(occ_measures, np.sum(occ_measures))
                policy_buffer[0].append(list(obs))
                policy_buffer[1].append(act_probs)
                #policy_buffer[obs] = occ_measures
            action_id = int(
                np.random.choice(a=np.arange(len(act_probs)),
                                 size=1,
                                 p=act_probs))
            
            action = all_joint_acts[action_id]
            new_obs, reward, done, info = env.step(action)
            if done:  # Terminate state
                new_obs = None
            trans = Transaction(obs, action, reward, new_obs)
            history.add(trans)
            obs = new_obs
            if done:
                break
    return history


def ori_alter_acts():
    # Return action-alternative_action pairs.
    # Used to calculate regret, since regret receives (state, original action, alternative action) as input
    return [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]


def monte_carlo_sampling(occupancy_measure, env, horizon):
    samples = generate_history(hor=horizon,
                               om_net=occupancy_measure,
                               env=env,
                               num_game=4)
    all_samples = samples.get_sample(size=200)
    sample_states = np.array([sample.state for sample in all_samples])
    sample_states = np.array(list(set([tuple(t) for t in sample_states])))
    transition_between_s = np.array([(sample.state, sample.state_next)
                                     for sample in all_samples])
    #transition_between_s = np.array(list(set([tuple(t) for t in transition_between_s])))

    sample_size = len(sample_states)
    # Get s' in bellman_flow constraint for all states
    prev_states = [None for _ in range(sample_size)]
    for i in range(len(sample_states)):
        state = sample_states[i]
        prev_s_eq = [all(t[1] == state) for t in transition_between_s]
        if sum(prev_s_eq) > 0:
            prev_s_id = prev_s_eq.index(True)
            prev_s = transition_between_s[prev_s_id][0]
        else:
            prev_s = None
        prev_states[i] = prev_s

    return sample_states, prev_states


def eval_risk(env, horizon, initialize_file, occupancy_measure):
    ## env: environment
    ## horizon: int, number of steps
    ## initialize_file: file used to initialize the environment
    ## occupancy_measure: the network used to generate actions
    init_pos = np.load(initialize_file)
    result = np.repeat(0,repeats=horizon)
    gamma = 0.99
    discounted_vector = np.array([np.power(gamma,h) for h in range(horizon)])
    n_games = len(init_pos)
    for initialize_position in init_pos:
        state = env.eval_reset(initialize_position)
        for h in range(horizon):
            result[h] += env.is_state_danger(state)
            act = pick_action(state,occupancy_measure,env)
            new_state, reward, done, info = env.step(act)
            if done:  # All agents die. All steps after are defined as dangerous states
                result[h+1:] += 1
                break
            state = new_state
    result = np.divide(result, n_games)  # Frequency to possibility
    result = np.multiply(result, discounted_vector)  # Discounted result
    estimated_risk = np.sum(result)
    return estimated_risk

