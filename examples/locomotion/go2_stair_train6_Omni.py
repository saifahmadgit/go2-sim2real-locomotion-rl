import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from go2_env_stair6_Omni import Go2Env


def get_train_cfg(exp_name, max_iterations, resume_path=None):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.003,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": resume_path is not None,
            "resume_path": resume_path,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 500,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def get_cfgs():
    # ================================================================
    # PLS config
    # ================================================================
    pls_enable = True
    pls_kp_range = [10.0, 70.0]
    pls_kp_default = 40.0
    pls_kp_action_scale = 20.0

    num_pos_actions = 12
    num_stiffness_actions = 4 if pls_enable else 0
    num_actions_total = num_pos_actions + num_stiffness_actions
    torque_limits = [23.7, 23.7, 45.0] * 4

    # ================================================================
    # Height scan (square for omnidirectional)
    # ================================================================
    height_scan_cfg = {
        "num_x": 11,
        "num_y": 11,
        "x_range": [-0.5, 0.5],
        "y_range": [-0.5, 0.5],
    }
    num_height_scan = height_scan_cfg["num_x"] * height_scan_cfg["num_y"]  # 121

    # ================================================================
    # Stairwell terrain config
    # ================================================================
    terrain_cfg = {
        "enabled": True,
        "horizontal_scale": 0.05,
        "vertical_scale": 0.005,

        # Stair geometry
        "step_depth_m": 0.39,
        "steps_per_flight": 10,
        "num_flights_per_side": 2,       # 2 flights with landing between
        "flat_landing_m": 1.5,
        "flat_bottom_m": 2.0,
        "flat_surround_m": 4.0,

        # Difficulty range
        "step_height_min": 0.02,
        "step_height_max": 0.135,
        "num_difficulty_levels": 9,
        "wells_per_row": 3,

        "height_scan": height_scan_cfg,
    }

    # ================================================================
    # DR
    # ================================================================
    kp_nominal = 60.0
    kd_nominal = 2.0

    # ================================================================
    # Curriculum
    # ================================================================
    curriculum_cfg = {
        "enabled": True,
        "level_init": 0.0,
        "level_min": 0.0,
        "level_max": 1.0,

        "ema_alpha": 0.03,

        "ready_timeout_rate": 0.60,
        "ready_tracking": 0.45,
        "ready_fall_rate": 0.35,
        "ready_streak": 5,

        "hard_fall_rate": 0.40,
        "hard_streak": 2,

        "step_up": 0.01,
        "step_down": 0.03,
        "cooldown_updates": 5,
        "update_every_episodes": 4096,

        "mix_prob_current": 0.80,
        "mix_level_low": 0.00,
        "mix_level_high": 0.50,

        "friction_easy": [0.6, 0.8],
        "kp_easy": [0.90 * kp_nominal, 1.10 * kp_nominal],
        "kd_easy": [0.75 * kd_nominal, 1.25 * kd_nominal],
        "kp_factor_easy": [0.95, 1.05],
        "kd_factor_easy": [0.95, 1.05],
        "mass_shift_easy": [-0.2, 0.5],
        "com_shift_easy": [-0.005, 0.005],
        "leg_mass_shift_easy": [-0.1, 0.1],
        "gravity_offset_easy": [-0.2, 0.2],
        "motor_strength_easy": [0.97, 1.03],
        "push_start": 0.3,
        "push_interval_easy_s": 10.0,
        "delay_easy_max_steps": 0,
        "global_dr_update_interval": 200,
    }

    # ================================================================
    # Env config
    # ================================================================
    env_cfg = {
        "num_actions": num_actions_total,
        "num_pos_actions": num_pos_actions,

        "pls_enable": pls_enable,
        "pls_kp_range": pls_kp_range,
        "pls_kp_default": pls_kp_default,
        "pls_kp_action_scale": pls_kp_action_scale,

        "kp": kp_nominal,
        "kd": kd_nominal,
        "torque_limits": torque_limits,

        "simulate_action_latency": True,

        "foot_names": ["FR_calf", "FL_calf", "RR_calf", "RL_calf"],
        "foot_contact_threshold": 3.0,

        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],

        "termination_if_roll_greater_than": 50,
        "termination_if_pitch_greater_than": 50,
        "termination_if_z_vel_greater_than": 100.0,
        "termination_if_y_vel_greater_than": 100.0,

        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 30.0,
        "resampling_time_s": 5.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,

        "terrain": terrain_cfg,
        "curriculum": curriculum_cfg,

        # DR hard ranges
        "friction_range": [0.3, 1.25],
        "kp_factor_range": [0.8, 1.2],
        "kd_factor_range": [0.8, 1.2],
        "kp_range": [50.0, 70.0],
        "kd_range": [1.0, 5.0],
        "obs_noise": {"ang_vel": 0.2, "gravity": 0.05, "dof_pos": 0.01, "dof_vel": 1.5},
        "obs_noise_level": 1.0,
        "action_noise_std": 0.1,
        "push_interval_s": 5.0,
        "push_force_range": [-150.0, 150.0],
        "push_duration_s": [0.05, 0.2],
        "init_pos_z_range": [0.38, 0.45],
        "init_euler_range": [-5.0, 5.0],
        "mass_shift_range": [-1.0, 3.0],
        "com_shift_range": [-0.03, 0.03],
        "leg_mass_shift_range": [-0.5, 0.5],
        "gravity_offset_range": [-1.0, 1.0],
        "motor_strength_range": [0.9, 1.1],
        "min_delay_steps": 0,
        "max_delay_steps": 1,

        # Two-phase DR
        "dr_schedule": {
            "phase1_level": 0.15,
            "terrain_gate": 0.50,
        },
    }

    # ================================================================
    # Observation config
    # ================================================================
    num_obs = 3 + 3 + 3 + 12 + 12 + num_actions_total

    num_privileged_extra = (3 + 1 + 12 + 12 + 12 + 1 + 3 + 4 + 3 + 3 + 1
                           + 1                     # well difficulty level
                           + num_height_scan)       # height scan (121)
    num_privileged_obs = num_obs + num_privileged_extra

    obs_cfg = {
        "num_obs": num_obs,
        "num_privileged_obs": num_privileged_obs,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    # ================================================================
    # Rewards — omnidirectional stair climbing
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.15,
        "feet_air_time_target": 0.1,
        "lin_vel_z_deadzone": 0.15,

        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,

            "alive": 0.2,

            "lin_vel_z": -1.0,
            "base_height": -0.1,
            "action_rate": -0.01,
            "similar_to_default": -0.05,
            "orientation_roll_only": -5.0,
            "dof_acc": -2.5e-7,
            "dof_vel": -5e-4,
            "ang_vel_xy": -0.05,

            "feet_air_time": 0.2,
            "foot_slip": -0.15,
            "foot_clearance": -0.5,
            "joint_tracking": -0.1,

            "energy": 0.0,
            "torque_load": 0.0,

            "stand_still": -0.5,
            "stand_still_vel": -2.0,
            "feet_stance": -0.3,
            # NOTE: NO forward_progress
        },
    }

    # ================================================================
    # Commands — forward-biased omnidirectional
    # ================================================================
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.3, 0.8],
        "lin_vel_y_range": [-0.4, 0.4],
        "ang_vel_range": [-0.5, 0.5],
        "cmd_curriculum": False,
        "compound_commands": True,
        "rel_standing_envs": 0.05,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairwell-omni-v1")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, resume_path=args.resume)

    if args.resume is None and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Print summary
    print("\n" + "=" * 70)
    print("  STAIRWELL TERRAIN — OMNIDIRECTIONAL STAIR CLIMBING")
    print("=" * 70)
    tc = env_cfg["terrain"]
    print(f"  Terrain       : {tc['num_difficulty_levels']} stairwells "
          f"({tc['wells_per_row']}×{tc['num_difficulty_levels']//tc['wells_per_row']})")
    print(f"  Step heights  : {tc['step_height_min']*100:.0f}cm → {tc['step_height_max']*100:.1f}cm")
    print(f"  Step depth    : {tc['step_depth_m']*100:.0f}cm")
    print(f"  Profile/side  : {tc['steps_per_flight']} stairs → landing → {tc['steps_per_flight']} stairs")
    print(f"  Commands      : x={command_cfg['lin_vel_x_range']} (forward-biased)")
    print(f"                  y={command_cfg['lin_vel_y_range']}")
    print(f"                  yaw={command_cfg['ang_vel_range']}")
    print(f"  DR schedule   : two-phase (gate={env_cfg['dr_schedule']['terrain_gate']})")
    print(f"  Curriculum    : metric-gated")
    print(f"  Obs           : {obs_cfg['num_obs']} actor / {obs_cfg['num_privileged_obs']} critic")
    if args.resume:
        print(f"  Resume from   : {args.resume}")
    print("=" * 70 + "\n")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()