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

# def balance_sampling(env,policy,steps):
#     lim_v = np.sum(np.array([np.power(0.99,i) for i in range(1000)]))
#     s = env.reset()
#     for t in range(steps):
#         state_id = env.vec_to_ind(s)
#         action_id = pick_action(state_id, policy)
#         action = env.ind_to_act(action_id)
#         s_new, r, done, info = env.step(action)
#         if state_id in balance_target[0]:
#             balance_list[0].append(t)
#         if state_id in balance_target[1]:
#             balance_list[1].append(t)

#         s = s_new
#     return balance_list


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



def main(obj_name, algo_name, env_name, task_name, constraint_threshold=0.05):

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
    horizon = arglist.horizon
    if algo_name == 'ModRew':
        csv_header = [
            'Environment', 'Task', 'SolutionConcept', 'Method', 'Epoch',
            'Iteration', 'Risk', 'ExpectedReward', 'BFViolated', 'RegretViolated',
            'observe_convergence','check_CE','check_CE_origame'
        ]
    else:
        csv_header = [
            'Environment', 'Task', 'SolutionConcept', 'Method', 'Epoch',
            'Iteration', 'Risk', 'ExpectedReward', 'BFViolated', 'RegretViolated',
            'observe_convergence','check_CE'
        ]
    file_name = "./Result/" + obj_name + algo_name + env_name + task_name + '---' + str(
        time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_mday) + '-' + str(
                time.localtime().tm_hour) + '-' + str(
                    time.localtime().tm_min) + '-' + str(
                        time.localtime().tm_sec) + ".csv"
    f = open(file_name, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(csv_header)
    f.close()
    # if task_name == 'balance':
    #     balance_sampling_file =  "./Result/" + 'BalanceResult' +'---'+ str(
    #                         time.localtime().tm_mon) + '-' + str(
    #                             time.localtime().tm_mday) + '-' + str(
    #                                 time.localtime().tm_hour) + '-' + str(
    #                                     time.localtime().tm_min) + '-' + str(
    #                                         time.localtime().tm_sec) + ".csv"
    # Some global variables
    if 'value' in task_name:
        target_sampling_file = "./Result/" + 'ValueTargetResult' + '---' + str(
            time.localtime().tm_mon) + '-' + str(
                time.localtime().tm_mday) + '-' + str(
                    time.localtime().tm_hour) + '-' + str(
                        time.localtime().tm_min) + '-' + str(
                            time.localtime().tm_sec) + ".csv"
    #Some global variables
    if algo_name == 'ModRew':
        foo_env = env_used(mod_rew=True)
    else:
        foo_env = env_used()
    state_n = foo_env.possible_states
    jact_n = foo_env.joint_act_nums
    gamma_v = np.sum(np.array([np.power(0.99, i) for i in range(horizon)]))
    ori_idx = [foo_env.get_ori_idx(i) for i in range(foo_env.num_agents)]
    alter_idx = [foo_env.get_alter_idx(i) for i in range(foo_env.num_agents)]
    target_epc = 5
    success_epc = 0
    epc = 0
    #while success_epc < target_epc:
    for epc in range(int(arglist.num_runs)):  # Each experiment
        om = None
        if algo_name == 'ModRew':
            env = env_used(mod_rew=True)
            policy = JointPolicy(env=env_used(mod_rew=True))
            non_mod_env = env_used(mod_rew=False)
        else:
            env = env_used()
            policy = JointPolicy(env=env_used())
        env.reset()
        qt_tables = [
            QTable(state_num=env.possible_states,
                   action_num=env.action_space,
                   agent_num=env.num_agents) for _ in range(env.num_agents)
        ]
        """
        Set objective function
        """
        if task_name == 'MDCE':
            arg_used = env
            if obj_name == 'DBCE':
                obj_func = cal_risk
            elif obj_name == 'MinRew':
                obj_func = cal_risk_reward
            elif obj_name == 'MinAbsRew':
                obj_func = cal_abs_rew
            elif obj_name == 'MaxRew':
                obj_func = cal_neg_total_reward
        elif 'value' in task_name:
            target_value = float(task_name.split('-')[1])
            obj_func = value_target
            if obj_name == 'DBCE':
                arg_used = (env, cal_risk, target_value)
            elif obj_name == 'MinRew':
                arg_used = (env, cal_risk_reward, target_value)
            elif obj_name == 'MinAbsRew':
                arg_used = (env, cal_abs_rew, target_value)
            elif obj_name == 'MaxRew':
                arg_used = (env, cal_neg_total_reward, target_value)
        elif task_name == 'balance':
            arg_used = env
            if obj_name == 'DBCE':
                obj_func = balance_risk
            elif obj_name == 'MinRew':
                obj_func = balance_reward
            elif obj_name == 'MinAbsRew':
                obj_func = balance_abs_reward
            elif obj_name == 'MaxRew':
                obj_func = cal_neg_total_reward
        ## Start Learning Below
        old_om = None
        old_qts = None
        stop_threshold = 0.01 * env.possible_states * env.joint_act_nums * (
            env.num_agents + 1)
        om_stop_thres = 0.01 * env.possible_states * env.joint_act_nums
        for iter in range(int(arglist.max_iter)):
            f = open(file_name, 'a', newline='')
            writer = csv.writer(f)
            print("iteration %d" % (iter + 1))
            ## Use current policy generate history
            #history = generate_history(horizon, policy=policy, env=env)
            ## Train Q-net according to the history
            #print("### Updating Q-table for game %d ###" % (iter + 1))
            if iter == 0:
                TD_update_q(qt_tables, policy, env, int(arglist.q_max_iter))
            #print("### Solve the CE ###")
            """
            Objective is minimizing sum among a of density of s*,a
            Constraint1 is occumeasure*regret <= 0
            Constraint2 is bellmanflow
            """
            state_n = env.possible_states
            jact_n = env.joint_act_nums
            qt_table_vals = [qt_tables[i].qt for i in range(env.num_agents)]
            tran_m = np.array(env.transition_matrix)
            #start = time.time()
            """
            Set all constraints
            """
            all_constraints = []
            for i in range(env.num_agents):
                all_constraints.append({
                    'type':
                    'ineq',
                    'fun':
                    regret_constraint,
                    'args': (state_n, jact_n, qt_table_vals[i], ori_idx[i],
                             alter_idx[i])
                })
            all_constraints.append({
                'type': 'eq',
                'fun': bellmanflow_constraint,
                'args': (state_n, jact_n, tran_m)
            })
            if algo_name == 'constrained':
                if task_name == 'MDCE':
                    all_constraints.append({
                        'type': 'ineq',
                        'fun': risk_constraint,
                        'args': (env, constraint_threshold)
                    })
                elif 'value' in task_name:
                    target_value = float(task_name.split('-')[1])
                    all_constraints.append({
                        'type':
                        'ineq',
                        'fun':
                        risk_constraint_valuetarget,
                        'args': (env, target_value, constraint_threshold)
                    })
                elif task_name == 'balance':
                    all_constraints.append({
                        'type': 'ineq',
                        'fun': risk_constraint_balance,
                        'args': (env, constraint_threshold)
                    })
            all_constraints = tuple(all_constraints)
            result_om = opt.minimize(
                fun=obj_func,
                args=arg_used,
                #method='SLSQP',
                x0=np.repeat(
                    gamma_v / (env.possible_states * env.joint_act_nums),
                    env.possible_states * env.joint_act_nums).reshape(
                        env.possible_states, env.joint_act_nums),
                bounds=[
                    (1e-5, None)
                    for _ in range(env.possible_states * env.joint_act_nums)
                ],
                options={
                    'disp': True,
                    'maxiter': 100
                },
                constraints=all_constraints)
            #print("Time use %f" % (time.time() - start))
            om = result_om.x
            if task_name == 'MDCE':
                result = cal_risk(om, env)
            elif 'value' in task_name:
                target_value = float(task_name.split('-')[1])
                result = value_target(om, env, cal_risk, target_value)
            elif task_name == 'balance':
                result = balance_risk(om, env)
            print(
                "#####################    Risk now is %f  for %s  %s   ##########################"
                % (result, obj_name, algo_name))
            reg_vals = [
                -regret_constraint(om, state_n, jact_n, qt_table_vals[i],
                                   ori_idx[i], alter_idx[i])
                for i in range(env.num_agents)
            ]
            bf_val = bellmanflow_constraint(om, state_n, jact_n, tran_m)
            print("BF Constraint Violated: %f" % max(abs(bf_val)))
            max_reg_vals = [np.max(reg_vals[i]) for i in range(env.num_agents)]
            print("Regret Constraint Violated: %f" % (max(max_reg_vals)))
            policy.derive_pi(
                om.reshape(env.possible_states, env.joint_act_nums))
            exp_rew = -cal_neg_total_reward(om, env)
            if old_om is not None:
                dif_om = np.sum(np.abs(old_om - om))
                old_om = copy.deepcopy(om)
            else:
                old_om = copy.deepcopy(om)
                dif_om = 9999
            if old_qts is not None:
                dif_qts = np.sum(
                    np.abs(np.array(qt_table_vals) - np.array(old_qts)))
                old_qts = copy.deepcopy(qt_table_vals)
            else:
                old_qts = copy.deepcopy(qt_table_vals)
                dif_qts = 9999

            ## Update Q and calculate regret to check CE
            if algo_name == 'ModRew':
                cp_tables = copy.deepcopy(qt_tables)
                TD_update_q(cp_tables, policy, non_mod_env, int(arglist.q_max_iter))
                reg_vals_new = [
                    -regret_constraint(om, state_n, jact_n, cp_tables[i].qt,
                                   ori_idx[i], alter_idx[i])
                    for i in range(env.num_agents)
                ]
                TD_update_q(qt_tables, policy, env, int(arglist.q_max_iter))
                reg_vals_new2 = [
                    -regret_constraint(om, state_n, jact_n, qt_tables[i].qt,
                                   ori_idx[i], alter_idx[i])
                    for i in range(env.num_agents)
                ]
                max_reg_vals_update2 = [np.max(reg_vals_new2[i]) for i in range(env.num_agents)]
            else:
                TD_update_q(qt_tables, policy, env, int(arglist.q_max_iter))
                reg_vals_new = [
                    -regret_constraint(om, state_n, jact_n, qt_table_vals[i],
                                   ori_idx[i], alter_idx[i])
                    for i in range(env.num_agents)
                ]
            max_reg_vals_update = [np.max(reg_vals_new[i]) for i in range(env.num_agents)]
            if algo_name != 'ModRew':
                if dif_qts + dif_om <= stop_threshold:  # or dif_om <= om_stop_thres:
                    print("Prune at iteration %d" % (iter + 1))
                    writer.writerow([
                        env_name, task_name, obj_name, algo_name,
                        str(epc),
                        str(iter),
                        str(float(result)),
                        str(float(exp_rew)),
                        str(float(max(abs(bf_val)))),
                        str(float(max(max_reg_vals))), 'True',
                        str(float(max(max_reg_vals_update))),
                    ])
                    f.close()
                    #break
                else:
                    writer.writerow([
                        env_name, task_name, obj_name, algo_name,
                        str(epc),
                        str(iter),
                        str(float(result)),
                        str(float(exp_rew)),
                        str(float(max(abs(bf_val)))),
                        str(float(max(max_reg_vals))), 'False',
                        str(float(max(max_reg_vals_update))),
                    ])
                    f.close()
                    #break
            else:
                if dif_qts + dif_om <= stop_threshold:  # or dif_om <= om_stop_thres:
                    print("Prune at iteration %d" % (iter + 1))
                    writer.writerow([
                        env_name, task_name, obj_name, algo_name,
                        str(epc),
                        str(iter),
                        str(float(result)),
                        str(float(exp_rew)),
                        str(float(max(abs(bf_val)))),
                        str(float(max(max_reg_vals))), 'True',
                        str(float(max(max_reg_vals_update))),
                        str(float(max(max_reg_vals_update2)))
                    ])
                    f.close()
                    #break
                else:
                    writer.writerow([
                        env_name, task_name, obj_name, algo_name,
                        str(epc),
                        str(iter),
                        str(float(result)),
                        str(float(exp_rew)),
                        str(float(max(abs(bf_val)))),
                        str(float(max(max_reg_vals))), 'False',
                        str(float(max(max_reg_vals_update))),
                        str(float(max(max_reg_vals_update2)))
                    ])
                    f.close()
                    #break
        #epc += 1
        if result <= 5 and obj_name=='DBCE':
        #     success_epc += 1
            np.save(env_name+task_name+obj_name+'.npy',om)
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
    main('DBCE','DBCE','FairGamble','MDCE')
    # env_name = 'FairGamble'
    # task = 'MDCE'
    # #main('MaxRew', 'ModRew', env_name, task)
    # main('DBCE', 'DBCE', env_name, task)
    # for obj_name in ['MaxRew']:
    #     main(obj_name, 'constrained', env_name, task)
    #     main(obj_name, 'ModRew', env_name, task)
    #     main(obj_name, 'constrained', env_name, task, constraint_threshold=5)

    # task = 'value-60'
    # main('DBCE', 'DBCE', env_name, task)
    # for obj_name in ['MaxRew']:
    #     main(obj_name, 'constrained', env_name, task)
    #     main(obj_name, 'constrained', env_name, task, constraint_threshold=5)

    # task = 'balance'
    # main('DBCE', 'DBCE', env_name, task)
    # for obj_name in ['MaxRew']:
    #     main(obj_name, 'constrained', env_name, task)
    #     main(obj_name, 'constrained', env_name, task, constraint_threshold=5)

    # task = 'value-10'
    # main('DBCE', 'DBCE', env_name, task)
    # for obj_name in ['MaxRew']:
    #     main(obj_name, 'constrained', env_name, task)
    #     main(obj_name, 'constrained', env_name, task, constraint_threshold=5)