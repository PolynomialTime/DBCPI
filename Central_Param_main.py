'''
Step1. Update Q-function
Step2. Update policy
'''
import csv
import time
from Functions.NN import *
from Environments.CaEParam import CaEParam
from Functions.Core import *
import numpy as np
import arguments
import scipy.optimize as opt
from Functions.OptFuncs_LP import *

def init_nn_inputs(env):
    l = []
    for state in env.all_states:
        obs = env.state_to_obs(state)
        for act in env.all_jacts:
            arr_act = np.array(act).reshape(-1)
            nn_input = np.concatenate((obs,arr_act))
            l.append(nn_input)
    return np.array(l).reshape(-1)

def main(obj_name, algo_name, env_name, task_name, constraint_threshold=0.05):
    env_used = CaEParam
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
    epc = 0
    #while success_epc < target_epc:
    for epc in range(int(arglist.num_runs)):  # Each experiment
        ## Initialize parameters for Lagrangian
        om_net = CentralOMNet(state_shape=foo_env.observation_space, state_num=foo_env.possible_states,jact_num=foo_env.joint_act_nums,n_agents=foo_env.num_agents,act_size=1)
        om_optimizer = torch.optim.Adam(om_net.parameters(),
                                        lr=arglist.learning_rate)
        lambdas = torch.ones(foo_env.possible_states)  # Lambda for regret
        mus = torch.ones(foo_env.possible_states)  # mu for bellmanflow
        env = env_used()
        policy = JointPolicy(env=env_used())
        
        tran_m = np.array(env.transition_matrix)
        #policy.derive_pi_nn(om_net)
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
            elif obj_name == 'MaxRew':
                obj_func = cal_neg_total_reward
        elif 'value' in task_name:
            target_value = float(task_name.split('-')[1])
            obj_func = value_target
            if obj_name == 'DBCE':
                arg_used = (env, cal_risk, target_value)
            elif obj_name == 'MaxRew':
                arg_used = (env, cal_neg_total_reward, target_value)
        elif task_name == 'balance':
            arg_used = env
            if obj_name == 'DBCE':
                obj_func = balance_risk_param
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
            Lagrangian method below
            """
            inputs = init_nn_inputs(env)

            """
            Dual Ascent here
            """

            for dual_iter in range(100):
                ## Update Theta
                ## Update Lambda and Mu
                ## Step1. calculate L(theta, lambda, mu)
                ## Step2. loss=L, minimize L to get theta
                ## Step3. update lambda and mu by learning rate and gradient(regret and bf-vals for each state.)
                if dual_iter % 2 == 0:
                    oms = om_net.forward(inputs)
                    risk = obj_func(oms,env)  # F(theta)
                    qt_table_vals = [qt_tables[i].qt for i in range(env.num_agents)]
                    reg_vals = torch.tensor([
                        -regret_constraint(oms, state_n, jact_n, qt_table_vals[i],
                                        ori_idx[i], alter_idx[i])
                        for i in range(env.num_agents)
                    ])
                    bf_val = bellmanflow_constraint(oms, state_n, jact_n, tran_m)
                    Lagrangian = risk
                    gx_agents =torch.sum(reg_vals,axis=0).reshape(4,-1)
                    gx = torch.sum(gx_agents,axis=1)
                    mu_hx = torch.sum(torch.mul(bf_val,mus))
                    lamb_gx = torch.sum(torch.mul(gx,lambdas))
                    Lagrangian = Lagrangian+mu_hx+lamb_gx
                    om_optimizer.zero_grad()
                    loss = Lagrangian
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(om_net.parameters(), float(arglist.grad_norm_clip))
                    om_optimizer.step()
                    
                else:
                    oms = om_net.forward(inputs).detach()
                    risk = obj_func(oms,env)  # F(theta)
                    qt_table_vals = [qt_tables[i].qt for i in range(env.num_agents)]
                    reg_vals = torch.tensor([
                        -regret_constraint(oms, state_n, jact_n, qt_table_vals[i],
                                        ori_idx[i], alter_idx[i])
                        for i in range(env.num_agents)
                    ])
                    gx_agents =torch.sum(reg_vals,axis=0).reshape(4,-1)
                    gx = torch.sum(gx_agents,axis=1)
                    bf_val = bellmanflow_constraint(oms, state_n, jact_n, tran_m)
                    for s in range(env.possible_states):
                        lambdas[s] += float(arglist.learning_rate)*gx[s]
                        if lambdas[s] <0:
                            lambdas[s] = 1e-4
                        mus[s] += float(arglist.learning_rate)*bf_val[s]
            #policy.derive_pi_nn(om_net) # Update policy
            policy.derive_pi(oms.detach().numpy())
            if task_name == 'MDCE':
                result = cal_risk(oms, env)
            elif 'value' in task_name:
                target_value = float(task_name.split('-')[1])
                result = value_target(oms, env, cal_risk, target_value)
            elif task_name == 'balance':
                result = balance_risk_param(oms, env)
            print(
                "#####################    Risk now is %f  for %s  %s   ##########################"
                % (result, obj_name, algo_name))
            
            print("BF Constraint Violated: %f" % max(abs(bf_val)))
            max_reg_vals = [torch.max(reg_vals[i]) for i in range(env.num_agents)]
            print("Regret Constraint Violated: %f" % (max(max_reg_vals)))
            exp_rew = -cal_neg_total_reward(oms, env)


            ## Update Q and calculate regret to check CE
            
            TD_update_q(qt_tables, policy, env, int(arglist.q_max_iter))
            reg_vals_new = [
                -regret_constraint(oms, state_n, jact_n, qt_table_vals[i],
                                ori_idx[i], alter_idx[i])
                for i in range(env.num_agents)
            ]
            max_reg_vals_update = [np.max(reg_vals_new[i]) for i in range(env.num_agents)]
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
            
        #epc += 1
        if result <= 5 and obj_name=='DBCE':
            np.save(env_name+task_name+obj_name+'.npy',oms)



if __name__ == "__main__":
    main('DBCE','DBCE','CaEParam','balance')
    main('DBCE','DBCE','CaEParam','value-10')
    main('DBCE','DBCE','CaEParam','MDCE')
