'''
Step1. Update Q-function
Step2. Update policy
'''
import torch
import time
import os
import csv
from Functions.NN import QNet, OMNet
from Functions.OptFuncs import *
import numpy as np
from Environments.TrafficEnv.Traffic import TrafficEnv
import arguments
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(argv):
    algo_name = argv[0]
    obj_name = argv[1]
    csv_header = ['Method', 'Epoch', 'Iteration', 'EvalRisk', 'NormalizedSumOM']
    file_name = "./Result/" + algo_name + obj_name + "Traffic_log" + str(
        time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_mday) + '-' + str(
                time.localtime().tm_hour) + '-' + str(
                    time.localtime().tm_min) + '-' + str(
                        time.localtime().tm_sec) + ".csv"
    f = open(file_name, 'a', newline='')
    loss_file = "./Result/" + algo_name + obj_name + "Traffic_loss" + str(
        time.localtime().tm_mon) + '-' + str(
            time.localtime().tm_mday) + '-' + str(
                time.localtime().tm_hour) + '-' + str(
                    time.localtime().tm_min) + '-' + str(
                        time.localtime().tm_sec) + ".txt"
    
    writer = csv.writer(f)
    writer.writerow(csv_header)
    f.close()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    arglist = arguments.parse_args()
    gamma = arglist.gamma
    horizon = int(arglist.horizon)  #arglist.horizon

    for epc in range(int(arglist.num_runs)):  # Each experiment
        #f.write("Epoch    " + str(epc+1) + '\n')
        f = open(file_name, 'a', newline='')
        f2 = open(loss_file, 'a')
        writer = csv.writer(f)
        if algo_name == 'ModRew':
            env = TrafficEnv(modify_reward=True, energy_thres=0.8,n_agents=6)
        else:
            env = TrafficEnv(energy_thres=0.8,n_agents=6)
        env.reset()
        # init_pos = env.generate_init_pos()
        # np.save(arglist.init_path, init_pos)
        q_net = QNet(state_shape=env.observation_space,
                     n_agents=env.num_agents)
        om_net = OMNet(state_shape=env.observation_space,
                       n_agents=env.num_agents)
        lambda_vals = np.ones(env.num_agents)
        if algo_name == 'Constrained':
            risk_cons_val = 1
        mu_val = 1
        qnet_optimizer = torch.optim.Adam(q_net.parameters(),
                                          lr=arglist.learning_rate)
        om_optimizer = torch.optim.Adam(om_net.parameters(),
                                        lr=arglist.learning_rate)
        for iter in range(int(arglist.max_iter)):
            start = time.time()
            """
            Update Q-network start
            """
            ## Use current policy generate history
            history = generate_history(horizon, om_net=om_net, env=env)
            print("##### History generated #####")
            ## Train Q-net according to the history
            print("### Updating Q-network for game %d ###" % (iter + 1))
            for n in range(int(arglist.q_net_max_iter)):
            #for n in range(2):
                samples = history.get_sample(size=300)
                y_s = []
                yhat_s = []
                for sample in samples:
                    current_yhat = q_net.forward(
                        list(sample.state) + list(sample.action)).to(
                            device, torch.float)
                    if sample.state_next is None:  # Terminate state
                        current_y_i = torch.tensor(sample.reward).to(
                            device, torch.float)
                    else:
                        next_act = pick_action(sample.state_next, om_net, env)
                        current_y_i = torch.tensor(sample.reward).to(
                            device, torch.float) + gamma * q_net.forward(
                                list(sample.state_next) + list(next_act)).to(
                                    device, torch.float)
                    y_s.append(current_y_i.detach())
                    yhat_s.append(current_yhat)
                loss = q_net.loss_func(torch.stack(y_s, 0),
                                       torch.stack(yhat_s, 0))
                qnet_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), float(arglist.grad_norm_clip))
                qnet_optimizer.step()

            print("### Dual Ascent Start ###")
            """
            Optimize by Lagrangian
            L(x) = f(x)+lambda*g(x)+mu*h(x), g(x) is inequality constraint, h(x) is equality constraint.
            Minimize L to get om_net
            Maximize inf_L to get lambda and mu
            """
            # Initialize parameters
            jact_n = env.joint_act_nums
            ori_idx = [env.get_ori_idx(i) for i in range(env.num_agents)]
            alter_idx = [env.get_alter_idx(i) for i in range(env.num_agents)]
            ## Initialize indicator for dual ascent
            minimize_L_flag = True
            maximize_infl_flag = False
            # result = None
            for dual_iter in range(int(arglist.dual_ascent_max_iter)):
                ## Monte-carlo sampling based on current occupancy measure
                sample_states, prev_states = monte_carlo_sampling(om_net, env, horizon=int(arglist.horizon))
                sample_size = len(sample_states)
                if algo_name == 'Constrained':
                    avg_samples = np.mean(sample_states,0)
                row_i = 0
                state_act_pairs = []
                for state in sample_states:
                    for joint_act in env.joint_acts():
                        state_act_pairs.append(list(state) + list(joint_act))
                    row_i += 1
                q_values = q_net.forward(state_act_pairs).detach().to(
                    device, torch.float)
                qt_tables = [
                    q_values[:, i].reshape(sample_size, -1)
                    for i in range(env.num_agents)
                ]

                acts = ori_alter_acts()
                state_alteract_pairs = []
                for act in acts:
                    state_alteract_pairs += [
                        list(state) + act for state in sample_states
                    ]
                if minimize_L_flag:
                    start = time.time()
                    print(
                        "###### Minimizing Lagrangian to optimize occupancy measure network"
                    )

                    for _ in range(int(arglist.dual_ascent_inner_iter)):
                        ## Update occupancy measures
                        row_i = 0
                        om_vals = om_net.forward(state_act_pairs).to(
                            device, torch.float)
                        if algo_name == 'DBCE':
                            f_x = cal_risk(om_vals, env, sample_size,
                                        sample_states)
                        else:
                            f_x = cal_qval(om_vals, env, qt_tables, sample_size,
                                        sample_states)
                        if obj_name == 'value-target':
                            f_x = torch.abs(f_x - float(arglist.target_value))
                        g_x = [
                            -regret_constraint(om_vals, sample_size, jact_n,
                                              qt_tables[i], ori_idx[i],
                                              alter_idx[i]).to(
                                                  device, torch.float)
                            for i in range(env.num_agents)
                        ]
                        if algo_name == 'Constrained':
                            g_cons_x = risk_constraint(om_vals, env, sample_size, sample_states, float(arglist.risk_threshold))
                        h_x = bellmanflow_constraint(om_vals, sample_size,
                                                     jact_n, sample_states,
                                                     prev_states).to(
                                                         device, torch.float)
                        Lagrangian_x = f_x + torch.sum(h_x * torch.tensor(mu_val).to(device, torch.float))
                        if algo_name == 'Constrained':
                            Lagrangian_x = Lagrangian_x + g_cons_x * risk_cons_val
                        for i in range(env.num_agents):
                            Lagrangian_x = Lagrangian_x + torch.sum(
                                g_x[i] * lambda_vals[i])
                        om_optimizer.zero_grad()
                        loss = Lagrangian_x
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(om_net.parameters(), float(arglist.grad_norm_clip))
                        om_optimizer.step()
                        print("Loss value is %f" % loss)
                        print("********Current Target Value is %f *****" % f_x)
                    f2.write("**************************************************************************************************")
                    f2.write("Dual Iteration %d  Optimizing occupancy measure network" % dual_iter)
                    f2.write("Loss value is %f" % loss)
                    f2.write("********Current Target Value is %f *****" % f_x)
                    maximize_infl_flag = True
                    minimize_L_flag = False
                    print("###### Optimize occupancy measure time %f" %
                          (time.time() - start))
                elif maximize_infl_flag:
                    #torch.autograd.set_detect_anomaly(True)
                    print(
                        "###### Fix our occupancy measure network, then update lambda and mu"
                    )
                    start = time.time()
                    row_i = 0
                    ## Update occupancy measures
                    om_vals = om_net.forward(state_act_pairs).detach().to(
                        device, torch.float)
                    for _ in range(int(arglist.dual_ascent_inner_iter)):
                        ## Update lambda and mu
                        g_x = [
                            -regret_constraint(om_vals, sample_size, jact_n,
                                              qt_tables[i], ori_idx[i],
                                              alter_idx[i]).to(
                                                  device, torch.float)
                            for i in range(env.num_agents)
                        ]
                        h_x = bellmanflow_constraint(om_vals, sample_size,
                                                     jact_n, sample_states,
                                                     prev_states).to(
                                                         device, torch.float)
                        
                        if algo_name == 'Constrained':
                            g_cons_x = risk_constraint(om_vals, env, sample_size, sample_states, float(arglist.risk_threshold))
                            risk_cons_val += float(arglist.beta)*torch.mean(g_cons_x)
                            if risk_cons_val <= 0:
                                risk_cons_val = 1e-4
                        for i in range(env.num_agents):
                            lambda_vals[i] += float(arglist.beta)*torch.mean(g_x[i])
                            if lambda_vals[i] <= 0:
                                lambda_vals[i] = 1e-4
                        mu_val += float(arglist.beta)*torch.mean(h_x)
                    maximize_infl_flag = False
                    minimize_L_flag = True
                    f2.write("**************************************************************************************************")
                    f2.write("Dual Iteration %d  Optimizing lambda and mu network" % dual_iter)
                    print("###### Optimize lambda and mu time %f" %
                          (time.time() - start))
            #f.write("Target value is " + str(result) + '\n')
            #risk_val = cal_risk(om_vals, env, sample_size, sample_states)
            risk_val = eval_risk(env, int(arglist.horizon),arglist.init_path,om_net)
            norm_om = normalized_risk(om_vals, env, sample_size, sample_states)
            print("###########################################################")
            print("###########################################################")
            print("Iteration %d" % iter)
            print("Time used %f" % (time.time() - start))
            writer.writerow(
                [algo_name,
                 str(epc),
                 str(iter),
                 str(float(risk_val)),
                 str(float(norm_om))])
        f.close()
        f2.close()


if __name__ == "__main__":
    main(['DBCE', 'min-val'])
    
    