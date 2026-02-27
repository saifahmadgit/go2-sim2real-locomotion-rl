import argparse
import os
import pickle
import shutil

import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 60.0,
        "kd": 2.0,
        "termination_if_roll_greater_than": 25,
        "termination_if_pitch_greater_than": 25,
        "termination_if_z_vel_greater_than":100.0,
        "termination_if_y_vel_greater_than":100.0,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "episode_length_s": 3.0,
        "resampling_time_s": 2.0,
        "action_scale": 0.65,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
         "crouch_speed": 5.0,

        # -------- Domain Randomization --------
        "friction_range": (0.4, 0.9),

        "kp_scale_range": (0.4, 1.5),
        "kd_scale_range": (0.25, 2.0),

        "push_enable": True,
        "push_interval_s": 1.0,
        "push_prob": 1.0,
        "push_force_range": (0.0, 0.0),
        "push_z_scale": 0.0,
        "push_duration_s": 0.15,
        "push_direction_mode": "random",
        # -------------------------------------
    }

    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "jump_apex_height": 0.55,
        "jump_apex_sigma": 0.06,

        "reward_scales": {
            "jump_impulse": 6.0,     
            "jump_apex": 20.0,       
            "xy_stability": 12.0,    
            "orientation": 3.0,      
            "no_shake": 1.0,         
            "crouch": 6.0,           
        }
    }


    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 8,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.6,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-jump")
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=4096)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    log_dir = f"logs/{args.exp_name}"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()
