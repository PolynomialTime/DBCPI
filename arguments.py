
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Traffic Environment")

    # environment
    parser.add_argument("--file_path", type=str, default="./Environments/TrafficEnv/NYC.json", help="path of the traffic net file")
    parser.add_argument("--init_path", type=str, default="./Environments/TrafficEnv/init_pos.npy", help="path of the traffic net file")
    parser.add_argument("--horizon", type=int, default=100, help="horizon of games")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--max_iter', type=int, default=250, help="Number of games in one experiment")
    parser.add_argument("--grad_norm_clip", type=float, default=10, help="clip value for grad clip")
    parser.add_argument("--beta", type=float, default=0.01, help="learning rate for lambda and mu")
    # core training parameters
    parser.add_argument("--q_max_iter", type=int, default=1000, help="Max iterations to train Q network")
    parser.add_argument("--dual_ascent_max_iter", type=int, default=50, help="Max iterations of dual ascent")
    parser.add_argument("--dual_ascent_inner_iter", type=int, default=2, help="Max iterations of dual ascent(inner loop)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning_rate")
    parser.add_argument("--risk_threshold", type=float, default=0.1, help="threshold for risk constraint")
    parser.add_argument("--target-value", type=float, default=0.8, help="target value for target correlated equilibrium")
    # experiment control parameters
    parser.add_argument("--num_runs", type=int, default=3, help="number of independent runs")
    parser.add_argument("--num_agents", type=int, default=2, help="number of agents")

    # checkpointing
    parser.add_argument("--save_results_dir", type=str, default="./results_output/",
                        help="directory which results are output to")

    return parser.parse_args()