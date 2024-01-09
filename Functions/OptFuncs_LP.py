from os import environ
import numpy as np
import numba as nb
from numba import jit
import torch

"""
Value target below
"""
def value_target(om,env,func,target_value):
    return np.abs(func(om,env) - target_value)


"""
MDCE target below
"""

def cal_neg_nonrisk_reward(occupancy_measure, environment):
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    reward_table = environment.reward_table
    sum_agent_r = np.sum(reward_table,2)
    r_mul_rho = np.multiply(occupancy_measure,sum_agent_r)
    sum_r = np.sum(r_mul_rho)


    risk_ids = environment.bad_states
    risk_oms = occupancy_measure[list(risk_ids)]
    risk_reward = sum_agent_r[list(risk_ids)]
    risk_mul_rho = np.multiply(risk_oms, risk_reward)
    risk_r = np.sum(risk_mul_rho)
    return -(sum_r - risk_r)

def cal_neg_total_reward(occupancy_measure, environment):
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    reward_table = environment.reward_table
    sum_agent_r = np.sum(reward_table,2)
    r_mul_rho = np.multiply(occupancy_measure,sum_agent_r)
    sum_r = torch.sum(r_mul_rho)
    return -sum_r


def cal_risk_reward(occupancy_measure, environment):
    risk_ids = environment.bad_states
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    risk_oms = occupancy_measure[list(risk_ids)]
    reward_table = environment.reward_table
    sum_agent_r = np.sum(reward_table,2)
    risk_reward = sum_agent_r[list(risk_ids)]
    r_mul_rho = np.multiply(risk_oms, risk_reward)
    sum_r = np.sum(r_mul_rho)
    return sum_r


def cal_abs_rew(occupancy_measure, environment):
    risk_ids = environment.bad_states
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    risk_oms = occupancy_measure[list(risk_ids)]
    reward_table = environment.reward_table
    sum_agent_r = np.sum(reward_table,2)
    risk_reward = sum_agent_r[list(risk_ids)]
    r_mul_rho = np.multiply(risk_oms, np.abs(risk_reward))
    sum_r = np.sum(r_mul_rho)
    return sum_r

def cal_risk(occupancy_measure, environment):
    ## objective of our optimization problem.
    #start = time.time()
    risk_ids = environment.bad_states
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    risk_oms = occupancy_measure[list(risk_ids)]
    result = np.sum(risk_oms)
    #for row in risk_ids:
    #    result += np.sum(occupancy_measure[row])
    #print("Cal risk time %f" % (time.time() - start))
    return result



"""
Constraints below
"""

#@jit(nopython=True)
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
    regret_val = torch.zeros(state_num)
    for state_i in range(len(occupancy_measure)):  # For each state
        real_Qs = torch.tensor([q_table[state_i][act_ids] for act_ids in ori_acts
                            ])  # Q-value of alter actions  3
        alter_Qs = torch.tensor(
            [q_table[state_i][act_ids] for act_ids in alter_acts])  ## 3
        rhos = torch.tensor(
            [occupancy_measure[state_i][act_ids] for act_ids in ori_acts])
        diff_Q = torch.subtract(alter_Qs, real_Qs)
        regret = torch.sum(np.multiply(rhos, diff_Q))
        regret_val[state_i] = regret
    #print("Cal regret time %f" % (time.time() - start))
    return regret_val


def regret_constraint(occupancy_measure, state_num, joint_act_num, q_table,
                      ori_idx, alter_idx):
    results = np.array([])
    for i in range(len(ori_idx)):  # For each possible action
        ori_act = ori_idx[i]
        alter_acts = alter_idx[i]
        for possible_alter_act in alter_acts:  # How many possible alternative action pairs
            regret_val = calculate_regret(occupancy_measure, state_num, joint_act_num,
                             q_table, nb.typed.List(ori_act), nb.typed.List(possible_alter_act))
            results = np.append(results, regret_val)
    return -results  # Shape should be (n_act * (n_act - 1)) * n_states

@jit(nopython=True)
def regret_constraint_onestate(occupancy_measure, state_id,q_table,
                      ori_idx, alter_idx):
    results = np.zeros(len(ori_idx))
    sub_qt = q_table[state_id]
    for i in range(len(ori_idx)):  # For each possible action
        alter_acts = alter_idx[i]
        for possible_alter_act in alter_acts:  # How many possible alternative action pairs
            ori_act = ori_idx[i]
            real_Qs = np.array([sub_qt[act_ids] for act_ids in ori_act
                            ])  # Q-value of alter actions  3
            alter_Qs = np.array(
                [sub_qt[act_ids] for act_ids in possible_alter_act])  ## 3
            rhos = np.array(
                [occupancy_measure[act_ids] for act_ids in ori_act])
            diff_Q = np.subtract(alter_Qs, real_Qs)
            regret = np.sum(np.multiply(rhos, diff_Q))
            results[i] = regret
    return -results  # Shape should be (n_act * (n_act - 1)) * n_states

#@jit(nopython=True)
def bellmanflow_constraint(occupancy_measure, state_n, jact_n, tranm):
    #start = time.time()
    occupancy_measure = occupancy_measure.reshape(state_n, jact_n)
    ## for any s \sum_{a}(rho(s,a))=\rho_0(s) + sum_{s',a}(\gamma*P(s|s',a)*rho(s',a))
    bellman_flow_vals = torch.zeros(state_n)
    #tranm = np.array(environment.transition_matrix)
    for row_id in range(len(occupancy_measure)):
        rho_0 = 1/state_n
        row = occupancy_measure[row_id]
        sum_om = torch.sum(row)
        tran_to_row = torch.tensor(tranm[:, :,
                            row_id])  ## from each state, by any action, to current state
        sum_transition_rho = torch.sum(torch.mul(tran_to_row,
                                                occupancy_measure))
        sum_transition_rho = sum_transition_rho * 0.99
        bellman_flow_vals[row_id] = sum_om - rho_0 - sum_transition_rho
    #print("Cal bellman_flow time %f" % (time.time() - start))
    return bellman_flow_vals  # this value should be equal to zero
    
@jit(nopython=True)
def bellmanflow_constraint_onestate(occupancy_measure, state_n, tranm,state_id,static_om):
    #start = time.time()
    ## for any s \sum_{a}(rho(s,a))=\rho_0(s) + sum_{s',a}(\gamma*P(s|s',a)*rho(s',a))
    rho_0s = np.zeros(state_n)
    rho_0s[0] = 1
    #tranm = np.array(environment.transition_matrix)
    sum_om = np.sum(occupancy_measure)
    tran_to_row = tranm[:,:,state_id]
    sum_transition_rho = np.sum(np.multiply(tran_to_row,static_om))
    sum_transition_rho *= 0.99
    result = sum_om - rho_0s[state_id] - sum_transition_rho
    return result  # this value should be equal to zero

def cal_risk_onestate(occupancy_measure):
    return sum(occupancy_measure)

def bellmanflow_target_onestate(occupancy_measure, state_n, tranm,state_id,static_om):
    #start = time.time()
    ## for any s \sum_{a}(rho(s,a))=\rho_0(s) + sum_{s',a}(\gamma*P(s|s',a)*rho(s',a))
    rho_0 = 1 / state_n
    #tranm = np.array(environment.transition_matrix)
    sum_om = np.sum(occupancy_measure)
    tran_to_row = tranm[:,:,state_id]
    sum_transition_rho = np.sum(np.multiply(tran_to_row,static_om))
    sum_transition_rho *= 0.99
    result = sum_om - rho_0 - sum_transition_rho
    return np.abs(result)  # this value should be equal to zero

def risk_constraint(occupancy_measure, environment, threshold):
    risk_ids = environment.bad_states
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    risk_oms = occupancy_measure[list(risk_ids)]
    #result = np.sum(risk_oms) - threshold
    result = threshold - np.sum(risk_oms)
    return result

def risk_constraint_valuetarget(occupancy_measure, environment, target, threshold):
    risk_ids = environment.bad_states
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    risk_oms = occupancy_measure[list(risk_ids)]
    #result = np.sum(risk_oms) - threshold
    result = abs(target - np.sum(risk_oms))
    return threshold - result

"""
Balance target below
"""
def balance_risk(occupancy_measure, environment):
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    balance_states = environment.balance_states
    results = []
    for states_ids in balance_states:
        risk_oms = occupancy_measure[list(states_ids)]
        results.append(np.sum(risk_oms))
    return np.abs(results[0] - results[1])

def balance_risk_param(occupancy_measure, environment):
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    balance_states = environment.balance_states
    results = []
    for states_ids in balance_states:
        risk_oms = occupancy_measure[list(states_ids)]
        results.append(torch.sum(risk_oms))
    return torch.abs(results[0] - results[1])


def balance_reward(occupancy_measure, environment):
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    balance_states = environment.balance_states
    results = []
    reward_table = environment.reward_table
    sum_agent_r = np.sum(reward_table,2)
    for states_ids in balance_states:
        risk_oms = occupancy_measure[list(states_ids)]
        risk_reward = sum_agent_r[list(states_ids)]
        r_mul_rho = np.multiply(risk_oms, risk_reward)
        sum_r = np.sum(r_mul_rho)
        results.append(sum_r)
    return np.abs(results[0] - results[1])

def balance_abs_reward(occupancy_measure, environment):
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    balance_states = environment.balance_states
    results = []
    reward_table = environment.reward_table
    sum_agent_r = np.sum(reward_table,2)
    for states_ids in balance_states:
        risk_oms = occupancy_measure[list(states_ids)]
        risk_reward = sum_agent_r[list(states_ids)]
        r_mul_rho = np.multiply(risk_oms, np.abs(risk_reward))
        sum_r = np.sum(r_mul_rho)
        results.append(sum_r)
    return np.abs(results[0] - results[1])

def risk_constraint_balance(occupancy_measure, environment, threshold):
    balance_states = environment.balance_states
    occupancy_measure = occupancy_measure.reshape(environment.possible_states,
                                                  environment.joint_act_nums)
    om_s1, om_s2 = occupancy_measure[balance_states[0]], occupancy_measure[balance_states[1]]
    #result = np.sum(risk_oms) - threshold
    result = abs(sum(sum(om_s1) - sum(om_s2)))
    return threshold - result