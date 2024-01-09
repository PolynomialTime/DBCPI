'''
Step1. Update Q-function
Step2. Update policy
'''
import csv
import time
from Environments.CaE import CaE
from Environments.Hunt import Hunt
from Functions.Core import *
import numpy as np
from Environments.GatherEnv.Lever import Lever
import arguments
import scipy.optimize as opt
from numba import jit
from Functions.OptFuncs_LP import *


def main(obj_name, algo_name, env_name, task_name):
    
    if env_name == 'Lever':
        env_used = Lever
    elif env_name == 'SurviveSmall':
        env_used = CaE
    elif env_name == 'Hunt':
        env_used = Hunt
    arglist = arguments.parse_args()
    #algo_name = 'q-based'
    horizon = arglist.horizon

    csv_header = ['Environment','Task','SolutionConcept', 'Method', 'Epoch', 'Iteration', 'Risk', 'ExpectedReward', 'BFViolated', 'RegretViolated']
    file_name = "./Result/" + obj_name + algo_name + env_name + task_name +'---'+ str(
        time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_mday) + '-' + str(
                time.localtime().tm_hour) + '-' + str(
                    time.localtime().tm_min) + '-' + str(
                        time.localtime().tm_sec) + ".csv"
    f = open(file_name, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(csv_header)
    f.close()
    # Some global variables
    if algo_name == 'ModRew':
        foo_env = env_used(mod_rew=True)
    else:
        foo_env = env_used()
    state_n = foo_env.possible_states
    jact_n = foo_env.joint_act_nums
    gamma_v = np.sum(np.array([np.power(0.99,i) for i in range(horizon)]))
    ori_idx = [foo_env.get_ori_idx(i) for i in range(foo_env.num_agents)]
    alter_idx = [foo_env.get_alter_idx(i) for i in range(foo_env.num_agents)]
    tran_m = np.array(foo_env.transition_matrix)

    
    for epc in range(int(arglist.num_runs)):  # Each experiment
        om = None
        if algo_name == 'ModRew':
            env = env_used(mod_rew=True)
            policy = JointPolicy(env=env_used(mod_rew=True))
        else:
            env = env_used()
            policy = JointPolicy(env=env_used())
        env.reset()
        qt_tables = [QTable(state_num=env.possible_states,
                                    action_num=env.action_space,agent_num=
                                    env.num_agents) for _ in range(env.num_agents)]

        f = open(file_name, 'a', newline='')
        writer = csv.writer(f)
        s = env.reset()
        
        """
        Set objective function
        """
        if task_name == 'MDCE':
            arg_used = env
            if obj_name =='DBCE':
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
            if obj_name =='DBCE':
                arg_used = (env,cal_risk,target_value)
            elif obj_name == 'MinRew':
                arg_used = (env,cal_risk_reward,target_value)
            elif obj_name == 'MinAbsRew':
                arg_used = (env,cal_abs_rew,target_value)
            elif obj_name == 'MaxRew':
                arg_used = (env,cal_neg_total_reward,target_value)
        elif task_name == 'balance':
            arg_used = env
            if obj_name =='DBCE':
                obj_func = balance_risk
            elif obj_name == 'MinRew':
                obj_func = balance_reward
            elif obj_name == 'MinAbsRew':
                obj_func = balance_abs_reward
            elif obj_name == 'MaxRew':
                print("Max global reward method is not implemented for balancing target")
                raise NameError
        static_om = np.repeat(gamma_v / (env.possible_states * env.joint_act_nums),
                                env.possible_states * env.joint_act_nums).reshape(
                                    env.possible_states, env.joint_act_nums)
        for iter in range(100000):
            if iter % 100 == 0:
                print("iteration %d" % (iter + 1))

            old_tables = [copy.deepcopy(qt_tables[i]) for i in range(env.num_agents)]
            state_id = env.vec_to_ind(s)

            ## Solve LP below
            qt_table_vals = [qt_tables[i].qt for i in range(env.num_agents)]
            x = np.repeat(1 / env.joint_act_nums,
                            env.joint_act_nums).reshape(
                                env.joint_act_nums,)
            #test_table = regret_constraint_onestate(x,0,env.joint_act_nums,qt_table_vals[0],ori_idx[0],alter_idx[0])
            #test_bf = bellmanflow_constraint_onestate(x,env.possible_states,tran_m,0,static_om)
            """
            Set all constraints
            """
            all_constraints = []
            for i in range(env.num_agents):
                all_constraints.append({'type':'ineq','fun':regret_constraint_onestate,'args':(state_id,qt_table_vals[i],np.array(ori_idx[i]),np.array(alter_idx[i]))})
            if not env.is_state_danger(s):
                object_function_this_iter = bellmanflow_target_onestate
                arg_this_iter = (state_n, tran_m,state_id,static_om)
            else:
                object_function_this_iter = cal_risk_onestate
                all_constraints.append({'type':'eq','fun':bellmanflow_constraint_onestate,'args':(state_n,tran_m,state_id,static_om)})
                arg_this_iter = ()
            # if algo_name == 'constrained':
            #     if task_name == 'MDCE':
            #         all_constraints.append({'type':'ineq','fun':risk_constraint,'args':(env,0.05)})
            #     elif 'value' in task_name:
            #         target_value = float(task_name.split('-')[1])
            #         all_constraints.append({'type':'ineq','fun':risk_constraint_valuetarget,'args':(env,target_value,0.05)})
            all_constraints = tuple(all_constraints)

            result_om = opt.minimize(
                fun=object_function_this_iter,
                args=arg_this_iter,
                #method='SLSQP',
                x0=np.repeat(1 / env.joint_act_nums,
                            env.joint_act_nums).reshape(
                                env.joint_act_nums,),
                bounds=[
                    (0.001,None)
                    for _ in range(env.joint_act_nums)
                ],
                options={
                    'disp': False,
                    'maxiter': 10000
                },
                constraints=all_constraints)
            
            #print("Time use %f" % (time.time() - start))
            om = result_om.x
            static_om[state_id] = om  # Update static occupancy measure
            policy.derive_pi(
                static_om.reshape(env.possible_states, env.joint_act_nums))
            action_id = pick_action(state_id, policy)
            action = env.ind_to_act(action_id)
            s_new, r, done, info = env.step(action)
            #if (not done) and (s_new is not None):
            s_new_id = env.vec_to_ind(s_new)
            pi_a_snew = policy.policy[s_new_id]
            for i in range(env.num_agents):
                qt_tables[i].update_q(r[i], state_id,action_id,s_new_id,pi_a_snew)
            dif_qs = [abs(np.sum(qt_tables[i].qt - old_tables[i].qt)) for i in range(env.num_agents)]
            #loss = sum(dif_qs)
            # if round(loss,4) == round(old_loss,4):
            #     #print("Terminate at iter %d" % (n + 1))
            #     break
            # if sum(dif_qs) < 0.01*env.num_agents:
            #     #print("Terminate at iter %d" % (n + 1))
            #     break
            #old_loss = loss
            
            if task_name == 'MDCE':
                result = cal_risk(static_om,env)
            elif 'value' in task_name:
                target_value = float(task_name.split('-')[1])
                result = value_target(static_om,env,cal_risk,target_value)
            elif task_name == 'balance':
                result = balance_risk(static_om,env)
            reg_vals = [-regret_constraint(static_om,state_n, jact_n, qt_table_vals[i], ori_idx[i], alter_idx[i]) for i in range(env.num_agents)]
            bf_val = bellmanflow_constraint(static_om,state_n,jact_n,tran_m)
            max_reg_vals = [np.max(reg_vals[i]) for i in range(env.num_agents)]
            if iter % 100 == 0:
                print("Iteration %d" % iter)
                print("#####################    Risk now is %f  for %s  %s   ##########################" % (result,obj_name,algo_name))
                print("BF Constraint Violated: %f" % max(abs(bf_val)))
                print("Regret Constraint Violated: %f" % (max(max_reg_vals)))
            # policy.derive_pi(
            #     om.reshape(env.possible_states, env.joint_act_nums))
            exp_rew = -cal_neg_total_reward(static_om,env)
            # writer.writerow(
            #     [env_name,
            #     task_name,
            #     obj_name,
            #      algo_name,
            #      str(epc),
            #      str(iter),
            #      str(float(result)),
            #      str(float(exp_rew)),
            #      str(float(max(abs(bf_val)))),
            #      str(float(max(max_reg_vals)))])
            s = s_new
        f.close()
            
if __name__ == "__main__":
    main('DBCE','DBCE','Lever','MDCE')
