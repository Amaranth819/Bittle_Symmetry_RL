from rsl_rl.rsl_rl.runners import OnPolicyRunner
from bittle_rl_gym.utils.helpers import class_to_dict
import time
import os


class BittlePPO():
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [400, 400]
        critic_hidden_dims = [400, 400]
        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 3e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'Bittle'
        run_name = ''

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt



def create_alg_runner(env, alg_cfg = BittlePPO(), log_root = 'exps/'):
    if log_root == None:
        log_dir = None
    else:
        curr_time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_dir = os.path.join(log_root, f'{alg_cfg.runner.experiment_name}-{alg_cfg.runner.algorithm_class_name}-{curr_time_str}')
    alg_cfg_dict = class_to_dict(alg_cfg)
    runner = OnPolicyRunner(env, alg_cfg_dict, log_dir = log_dir, device = env.device)
    
    # Load the pretrained model if resuming
    resume = alg_cfg.runner.resume
    if resume:
        resume_path = alg_cfg.runner.resume_path
        print(f'Loading model from {resume_path}')
        runner.load(resume_path, load_optimizer = True)

    return runner, alg_cfg