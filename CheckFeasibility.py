'''
Step1. Update Q-function
Step2. Update policy
'''
from asyncio import tasks
import csv
import time
from Environments.CaE import CaE
from Environments.Hunt import Hunt
from Environments.HuntCoop import CoopHunt
from Environments.FairGamble import FairGamble
from Functions.Core import *
import numpy as np
from Environments.GatherEnv.Lever import Lever
import arguments
import scipy.optimize as opt
from numba import jit
from Functions.OptFuncs_LP import *

def balance_sampling(env,policy,steps):
    balance_target = env.balance_states
    balance_list = [[],[]]
    s = env.reset()
    for t in range(steps):
        state_id = env.vec_to_ind(s)
        action_id = pick_action(state_id, policy)
        action = env.ind_to_act(action_id)
        s_new, r, done, info = env.step(action)
        if state_id in balance_target[0]:
            balance_list[0].append(t)
        if state_id in balance_target[1]:
            balance_list[1].append(t)

        s = s_new
    return balance_list


def value_target_sampling(env, policy, steps, target_v):
    lim_v = np.sum(np.array([np.power(0.99, i) for i in range(1000)]))
    target_percent = target_v / lim_v
    s = env.reset()
    percent_list = []
    distance_list = []
    s_counter = 0
    for t in range(steps):
        state_id = env.vec_to_ind(s)
        action_id = pick_action(state_id, policy)
        action = env.ind_to_act(action_id)
        s_new, r, done, info = env.step(action)
        if env.is_state_danger(s):
            s_counter += 1
        current_percent = s_counter / (t + 1)
        percent_list.append(current_percent)
        distance_list.append(current_percent - target_percent)
        s = s_new
    return percent_list, distance_list


def main(obj_name, algo_name, env_name, task_name, file_name):

    if env_name == 'Lever':
        env_used = Lever
    elif env_name == 'CaE':
        env_used = CaE
    elif env_name == 'Hunt':
        env_used = Hunt
    elif env_name == 'FairGamble':
        env_used = FairGamble
    elif env_name == 'CoopHunt':
        env_used = CoopHunt
    arglist = arguments.parse_args()
    #algo_name = 'q-based'
    
    
    if task_name == 'balance':
        balance_sampling_file =  "./Result/" + 'BalanceResult' +'---'+ str(
                            time.localtime().tm_mon) + '-' + str(
                                time.localtime().tm_mday) + '-' + str(
                                    time.localtime().tm_hour) + '-' + str(
                                        time.localtime().tm_min) + '-' + str(
                                            time.localtime().tm_sec) + ".csv"
    # Some global variables
    if 'value' in task_name:
        target_sampling_file = "./Result/" + 'ValueTargetResult' + '---' + str(
            time.localtime().tm_mon) + '-' + str(
                time.localtime().tm_mday) + '-' + str(
                    time.localtime().tm_hour) + '-' + str(
                        time.localtime().tm_min) + '-' + str(
                            time.localtime().tm_sec) + ".csv"
    #Some global variables
    foo_env = env_used()
    policy = JointPolicy(env=env_used())
    om_file = np.load(file_name).reshape(foo_env.possible_states,-1)
    policy.derive_pi(occupancy_measure=om_file)
    for epc in range(10):  # Each experiment
        if task_name == 'balance':
            samples = balance_sampling(foo_env,policy,1500)
            with open(balance_sampling_file,'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(samples[0])
                writer.writerow(samples[1])
        if 'value' in task_name:
            target_value = float(task_name.split('-')[1])
            percent_list, distance_list = value_target_sampling(foo_env,policy,1500,target_value)
            with open(target_sampling_file,'a',newline='') as f:
                writer2 = csv.writer(f)
                writer2.writerow(percent_list)
                writer2.writerow(distance_list)
                writer2.writerow([])
        # if 'value' in task_name:
        #     if result <= 1:
        #         target_value = float(task_name.split('-')[1])
        #         f2 = open(target_sampling_file,'a',newline='')
        #         percent_list, distance_list = value_target_sampling(env,policy,500,target_value)
        #         writer2 = csv.writer(f2)
        #         writer2.writerow(percent_list)
        #         writer2.writerow(distance_list)
        #         writer2.writerow([])


if __name__ == "__main__":
    #main('DBCE', 'DBCE', 'CaE', 'balance',file_name='CaEbalanceDBCE.npy')
    main('DBCE', 'DBCE', 'Hunt', 'value-10',file_name='Huntvalue-10DBCE.npy')