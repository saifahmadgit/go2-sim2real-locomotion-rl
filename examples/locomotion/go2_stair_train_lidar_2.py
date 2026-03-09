"""
train_stair5_info.py
=====================
Training script for Go2 stair climbing with 4-value stair info.

Actor obs (53 dims):
  ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12)
  + actions(16) + stair_info(4)

  stair_info = [edge_distance, stair_height, stair_depth, direction]
    - edge_distance: 0..0.27m scaled by 1/0.27 (from LIDAR edge detection)
    - stair_height:  always positive, scaled by 1/0.20 (user-provided)
    - stair_depth:   always positive, scaled by 1/0.30 (user-provided)
    - direction:     +1 ascending, -1 descending, 0 flat

Curriculum: stair HEIGHT only (2cm → 20cm across 13 rows)
Stair DEPTH: variable 0.22–0.30m per row (not curriculum-linked)

Sim-to-Real Deployment:
  1. Train with this script
  2. On real Go2: measure stair height & depth, input as constants
  3. LIDAR edge detector provides edge_distance & direction automatically
  4. Policy obs = 49 proprio + 4 stair info = 53 dims
"""

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

from go2_env_stair_lidar_2 import Go2Env


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
    # Stair info configuration (replaces LIDAR grid)
    #
    # 4 values: edge_distance, stair_height, stair_depth, direction
    #
    # Edge detection: scans 0→0.27m ahead, finds first height change.
    # Height & depth: from terrain config per row (real robot: user input).
    # Direction: auto from height change sign.
    #
    # Noise is applied only to LIDAR-sourced values (edge, direction)
    # with minor noise on user-measured values (height, depth).
    # ================================================================
    stair_info_cfg = {
        "enabled": True,

        "edge_detection": {
            "max_range": 0.27,       # LIDAR reliable range
            "threshold": 0.02,       # 2cm height change = edge
            "num_samples": 14,       # ~2cm spacing over 0.27m
        },

        # Observation scaling (bring to ~[0,1] range)
        "edge_dist_scale": 3.7,     # 1/0.27
        "height_scale": 5.0,        # 1/0.20
        "depth_scale": 3.3,         # 1/0.30
        "direction_scale": 1.0,     # already [-1, 0, +1]

        "noise": {
            "edge_noise_std": 0.04,         # ~15% of 0.27m range
            "height_noise_std": 0.01,       # ±5% of 0.20m max
            "depth_noise_std": 0.01,        # ±5% of 0.30m
            "direction_flip_prob": 0.03,    # 3% wrong direction
            "full_blackout_prob": 0.05,     # 5% LIDAR dead
            "latency_prob": 0.10,           # 10% stale data
        },
    }

    num_stair_info = 4  # edge_dist, height, depth, direction

    # ================================================================
    # Terrain config (with per-row variable depth)
    # ================================================================
    terrain_cfg = {
        "enabled": True,
        "horizontal_scale": 0.05,
        "vertical_scale": 0.005,
        "num_difficulty_rows": 13,
        "row_width_m": 6.0,
        "step_depth_range": [0.22, 0.30],   # variable depth, NOT curriculum-linked
        "num_steps": 6,
        "num_flights": 4,
        "step_height_min": 0.02,             # curriculum: easy
        "step_height_max": 0.20,             # curriculum: hard
        "flat_before_m": 2.0,
        "flat_top_m": 1.5,
        "flat_gap_m": 1.5,
        "flat_after_m": 2.0,
    }

    # ================================================================
    # DR config
    # ================================================================
    friction_range = [0.3, 1.25]
    kp_factor_range = [0.8, 1.2]
    kd_factor_range = [0.8, 1.2]
    kp_nominal = 60.0
    kd_nominal = 2.0
    kp_range = [50.0, 70.0]
    kd_range = [1.0, 5.0]

    obs_noise = {
        "ang_vel": 0.2,
        "gravity": 0.05,
        "dof_pos": 0.01,
        "dof_vel": 1.5,
    }

    mass_shift_range = [-1.0, 3.0]
    com_shift_range = [-0.03, 0.03]
    leg_mass_shift_range = [-0.5, 0.5]
    gravity_offset_range = [-1.0, 1.0]
    motor_strength_range = [0.9, 1.1]

    push_force_range = [-150.0, 150.0]
    push_duration_s = [0.05, 0.2]

    # ================================================================
    # Curriculum (HEIGHT only)
    # ================================================================
    curriculum_cfg = {
        "enabled": True,
        "level_init": 0.2,
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
    # Environment config
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

        # Penalize contact on front thigh links (prevents knee-planting on stairs)
        "body_contact_names": ["FR_thigh", "FL_thigh"],
        "body_contact_threshold": 1.0,

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

        "termination_if_roll_greater_than": 45,
        "termination_if_pitch_greater_than": 45,
        "termination_if_z_vel_greater_than": 100.0,
        "termination_if_y_vel_greater_than": 100.0,

        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 25.0,
        "resampling_time_s": 5.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,

        "terrain": terrain_cfg,
        "stair_info": stair_info_cfg,
        "curriculum": curriculum_cfg,

        "friction_range": friction_range,
        "kp_factor_range": kp_factor_range,
        "kd_factor_range": kd_factor_range,
        "kp_range": kp_range,
        "kd_range": kd_range,
        "obs_noise": obs_noise,
        "obs_noise_level": 1.0,
        "action_noise_std": 0.1,
        "push_interval_s": 5.0,
        "push_force_range": push_force_range,
        "push_duration_s": push_duration_s,
        "init_pos_z_range": [0.38, 0.45],
        "init_euler_range": [-5.0, 5.0],
        "mass_shift_range": mass_shift_range,
        "com_shift_range": com_shift_range,
        "leg_mass_shift_range": leg_mass_shift_range,
        "gravity_offset_range": gravity_offset_range,
        "motor_strength_range": motor_strength_range,
        "min_delay_steps": 0,
        "max_delay_steps": 1,

        "dr_schedule": {
            "phase1_level": 0.15,
            "terrain_gate": 0.50,
        },
    }

    # ================================================================
    # Observation config
    # ================================================================
    num_proprio = 3 + 3 + 3 + 12 + 12 + num_actions_total  # 49
    num_obs = num_proprio + num_stair_info                   # 49 + 4 = 53

    # Privileged extra:
    #   lin_vel(3) + friction(1) + kp_factors(12) + kd_factors(12)
    #   + motor_strength(12) + mass_shift(1) + com_shift(3) + leg_mass(4)
    #   + gravity_offset(3) + push_force(3) + delay(1) + terrain_row(1)
    #   + clean_stair_info(4)
    num_privileged_extra = 3 + 1 + 12 + 12 + 12 + 1 + 3 + 4 + 3 + 3 + 1 + 1 + num_stair_info
    num_privileged_obs = num_obs + num_privileged_extra       # 53 + 60 = 113

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
    # Reward config
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.15,
        "feet_air_time_target": 0.1,
        "lin_vel_z_deadzone": 0.15,

        "reward_scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,
            "forward_progress": 0.4,
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
            "body_contact": -2.0,
        },
    }

    # ================================================================
    # Command config
    # ================================================================
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.8],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "cmd_curriculum": False,
        "compound_commands": True,
        "rel_standing_envs": 0.05,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairs-v5-info")
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

    # ================================================================
    # Print config summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  TRAINING CONFIG — STAIR CLIMBING v5 (4-value stair info)")
    print("=" * 70)

    si = env_cfg.get("stair_info", {})
    if si.get("enabled"):
        n_cfg = si.get("noise", {})
        print(f"  {'Stair info':30s}: ENABLED (4 values)")
        print(f"    [0] edge_distance     : 0..0.27m, scale={si.get('edge_dist_scale')}")
        print(f"    [1] stair_height      : positive, scale={si.get('height_scale')}")
        print(f"    [2] stair_depth       : positive, scale={si.get('depth_scale')}")
        print(f"    [3] direction         : +1/-1/0,  scale={si.get('direction_scale')}")
        print(f"    Noise:")
        print(f"      edge_std={n_cfg.get('edge_noise_std')}m  "
              f"height_std={n_cfg.get('height_noise_std')}m  "
              f"depth_std={n_cfg.get('depth_noise_std')}m")
        print(f"      dir_flip={n_cfg.get('direction_flip_prob')}  "
              f"blackout={n_cfg.get('full_blackout_prob')}  "
              f"latency={n_cfg.get('latency_prob')}")

    tc = env_cfg.get("terrain", {})
    if tc.get("enabled"):
        print(f"  {'Terrain':30s}: ENABLED")
        print(f"    Height curriculum    : {tc['step_height_min']*100:.0f}cm → {tc['step_height_max']*100:.0f}cm")
        print(f"    Depth (non-curriculum): {tc['step_depth_range'][0]*100:.0f}cm → {tc['step_depth_range'][1]*100:.0f}cm")
        print(f"    Rows: {tc['num_difficulty_rows']}, Flights: {tc['num_flights']}, Steps: {tc['num_steps']}")

    pls = env_cfg.get("pls_enable", False)
    print(f"  {'PLS':30s}: {'ON' if pls else 'OFF'}")
    print(f"  {'Action space':30s}: {env_cfg['num_actions']}")
    print(f"  {'Actor obs':30s}: {obs_cfg['num_obs']}  (49 proprio + 4 stair_info)")
    print(f"  {'Privileged critic obs':30s}: {obs_cfg['num_privileged_obs']}")
    print("=" * 70)

    print("\n  SIM-TO-REAL DEPLOYMENT:")
    print("  ─────────────────────")
    print("  1. Train: python train_stair5_info.py")
    print("  2. On real Go2:")
    print("     a. Measure stair height and depth")
    print("     b. Start deploy: python go2_deploy_stairs_info.py --height 0.15 --depth 0.25")
    print("     c. LIDAR auto-detects edge_distance and direction")
    print("     d. Policy gets 53-dim obs = 49 proprio + 4 stair_info")
    print("  ─────────────────────\n")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"))

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