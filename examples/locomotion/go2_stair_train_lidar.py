"""
train_stair4_lidar.py
=====================
Training script for Go2 stair climbing with LIDAR observations
in both actor and critic networks.

Key design — Minimal 5×3 grid for sim-to-real robustness:
  - Actor obs: 49 proprio + 15 LIDAR terrain scan = 64 dims
  - Critic obs includes dense privileged scan (45 pts)
  - Aggressive noise/dropout forces proprioceptive fallback
  - Fewer, larger cells → more LIDAR points per cell → smaller sim-to-real gap

Grid layout (body frame, forward = +x):
    x: [0.00, 0.15, 0.30, 0.50, 0.80] m    5 rows (~stair-step spacing)
    y: [-0.15, 0.00, 0.15] m                3 cols (center + sides)
    Cell size ~15cm → 8-10 L1 points per cell even in near field

Sim-to-Real Deployment Pipeline:
  1. Train with this script (actor learns to use 15-point terrain scan)
  2. Export actor network weights
  3. On real Go2: L1 point cloud → project to 5×3 body-frame grid →
     compute height per cell → subtract base z → clip → scale
  4. Feed into actor network alongside proprioceptive obs
  5. Policy outputs joint position targets (same as before)
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

from go2_env_stair_lidar import Go2Env


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
        "save_interval": 1000,
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
    # LIDAR scan configuration  ← MINIMAL 5×3 GRID
    #
    # Optimized for sim-to-real transfer on the Go2 L1 LIDAR:
    #
    # Actor scan: 5×3 = 15 points, forward-biased, non-uniform x spacing
    #   - x_points chosen to align with typical stair tread depths (~25cm)
    #   - 15cm cell size ensures 8-10 LIDAR points per cell even near-field
    #   - 3 lateral columns detect yaw misalignment
    #   - On real robot: L1 point cloud → voxel filter → project to this
    #     exact grid → height per cell → subtract base z
    #
    # Critic scan: 9×5 = 45 points, dense surround (privileged, not deployed)
    #
    # Noise model is AGGRESSIVE to prevent policy overfitting sim:
    #   - 15% per-cell dropout (was 5%): many cells go missing
    #   - 5% full blackout: entire scan dead, must use proprio alone
    #   - 2cm Gaussian noise (matches L1 spec)
    #   - 10% latency (processing delay)
    # ================================================================
    lidar_cfg = {
        "enabled": True,

        "actor_scan": {
            # Non-uniform x spacing: denser near feet, sparser far ahead
            "x_points": [0.0, 0.15, 0.30, 0.50, 0.80],   # 5 rows, ~stair-step spacing
            "y_points": [-0.15, 0.0, 0.15],                # 3 cols, center + sides
        },

        "critic_scan": {
            "num_x": 9,                      # Dense forward/backward
            "num_y": 5,                      # Wide lateral
            "x_range": [-0.4, 0.8],          # See behind too (privileged)
            "y_range": [-0.3, 0.3],          # Wider lateral view
        },

        "noise": {
            "height_noise_std": 0.02,        # ±2cm (matches L1 accuracy spec)
            "dropout_prob": 0.15,            # 15% per-cell dropout (AGGRESSIVE)
            "full_blackout_prob": 0.05,      # 5% entire scan dead (forces proprio fallback)
            "latency_prob": 0.1,             # 10% chance of 1-step-old scan
            "vertical_offset_std": 0.01,     # 1cm mounting vibration
        },

        "height_clip": 1.0,                 # Clip heights to ±1.0m
        "height_scale": 1.0,                # No extra scaling (heights in metres)
    }

    num_actor_lidar = 5 * 3    # 15 (from x_points × y_points)
    num_critic_lidar = lidar_cfg["critic_scan"]["num_x"] * lidar_cfg["critic_scan"]["num_y"]  # 45

    # ================================================================
    # Height scan config (legacy, kept for backward compat in terrain builder)
    # ================================================================
    height_scan_cfg = {
        "num_x": 9,
        "num_y": 5,
        "x_range": [-0.4, 0.8],
        "y_range": [-0.3, 0.3],
    }

    # ================================================================
    # Terrain config
    # ================================================================
    terrain_cfg = {
        "enabled": True,
        "horizontal_scale": 0.05,
        "vertical_scale": 0.005,
        "num_difficulty_rows": 13,
        "row_width_m": 6.0,
        "step_depth_m": 0.25,
        "num_steps": 6,
        "num_flights": 4,
        "step_height_min": 0.02,
        "step_height_max": 0.20,
        "flat_before_m": 2.0,
        "flat_top_m": 1.5,
        "flat_gap_m": 1.5,
        "flat_after_m": 2.0,
        "height_scan": height_scan_cfg,
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

    push_force_range = [0.0, 0.0]
    push_duration_s = [0.05, 0.2]

    # ================================================================
    # Curriculum
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

        # --- Terrain ---
        "terrain": terrain_cfg,

        # --- LIDAR ---  ← NEW
        "lidar": lidar_cfg,

        # --- Curriculum ---
        "curriculum": curriculum_cfg,

        # --- DR ---
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
    # Observation config  ← UPDATED for 5×3 LIDAR
    # ================================================================
    # Proprioceptive (same as before):
    #   ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12)
    #   + actions(16) = 49
    num_proprio = 3 + 3 + 3 + 12 + 12 + num_actions_total  # 49

    # Actor obs = proprioceptive + actor LIDAR scan (5×3 = 15)
    num_obs = num_proprio + num_actor_lidar                  # 49 + 15 = 64

    # Privileged extra (critic only):
    #   lin_vel(3) + friction(1) + kp_factors(12) + kd_factors(12)
    #   + motor_strength(12) + mass_shift(1) + com_shift(3) + leg_mass(4)
    #   + gravity_offset(3) + push_force(3) + delay(1) + terrain_row(1)
    #   + critic_lidar_scan(45)
    num_privileged_extra = (3 + 1 + 12 + 12 + 12 + 1 + 3 + 4 + 3 + 3 + 1
                           + 1                      # terrain_row
                           + num_critic_lidar)       # critic LIDAR scan (45)
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
    # Reward config
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.12,
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
        },
    }

    # ================================================================
    # Command config
    # ================================================================
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.4, 0.6],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
        "cmd_curriculum": False,
        "compound_commands": True,
        "rel_standing_envs": 0.05,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairs-v4-lidar")
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
    print("  TRAINING CONFIG — STAIR CLIMBING v4 + LIDAR (5×3 sim-to-real)")
    print("=" * 70)

    lidar = env_cfg.get("lidar", {})
    if lidar.get("enabled"):
        a_cfg = lidar.get("actor_scan", {})
        c_cfg = lidar.get("critic_scan", {})
        n_cfg = lidar.get("noise", {})
        # Support both x_points and num_x formats
        if "x_points" in a_cfg:
            na = len(a_cfg["x_points"]) * len(a_cfg["y_points"])
            actor_desc = f"{len(a_cfg['x_points'])}×{len(a_cfg['y_points'])} = {na} pts"
        else:
            na = a_cfg.get("num_x", 0) * a_cfg.get("num_y", 0)
            actor_desc = f"{a_cfg.get('num_x',0)}×{a_cfg.get('num_y',0)} = {na} pts"
        nc = c_cfg.get("num_x", 0) * c_cfg.get("num_y", 0)
        print(f"  {'LIDAR':30s}: ENABLED (5×3 sim-to-real)")
        print(f"    Actor scan           : {actor_desc}")
        if "x_points" in a_cfg:
            print(f"      x_points           : {a_cfg['x_points']}")
            print(f"      y_points           : {a_cfg['y_points']}")
        else:
            print(f"      x_range            : {a_cfg.get('x_range')}")
            print(f"      y_range            : {a_cfg.get('y_range')}")
        print(f"    Critic scan          : {c_cfg.get('num_x',0)}×{c_cfg.get('num_y',0)} = {nc} pts (privileged)")
        print(f"    Noise (AGGRESSIVE sim-to-real):")
        print(f"      height_noise_std   : {n_cfg.get('height_noise_std', 0)}m (L1 accuracy: ±2cm)")
        print(f"      dropout_prob       : {n_cfg.get('dropout_prob', 0)} (per-cell)")
        print(f"      full_blackout_prob : {n_cfg.get('full_blackout_prob', 0)} (entire scan)")
        print(f"      latency_prob       : {n_cfg.get('latency_prob', 0)} (processing delay)")
        print(f"      vertical_offset    : {n_cfg.get('vertical_offset_std', 0)}m (vibration)")
    else:
        print(f"  {'LIDAR':30s}: DISABLED")

    tc = env_cfg.get("terrain", {})
    if tc.get("enabled"):
        print(f"  {'Terrain':30s}: ENABLED")
        print(f"    Step height range    : {tc['step_height_min']*100:.0f}cm → {tc['step_height_max']*100:.0f}cm")

    pls = env_cfg.get("pls_enable", False)
    print(f"  {'PLS':30s}: {'ON' if pls else 'OFF'}")
    print(f"  {'Action space':30s}: {env_cfg['num_actions']}")
    print(f"  {'Actor obs':30s}: {obs_cfg['num_obs']}  (proprio={obs_cfg['num_obs'] - na} + lidar={na})")
    print(f"  {'Privileged critic obs':30s}: {obs_cfg['num_privileged_obs']}")

    if args.resume:
        print(f"  {'Resuming from':30s}: {args.resume}")
        print(f"  ⚠ Actor dim changed ({obs_cfg['num_obs']}): cannot resume from non-LIDAR or 55-pt checkpoint!")
        print(f"    Train from scratch or use a 5×3 LIDAR-compatible checkpoint.")

    print("=" * 70)

    # --- Sim-to-Real deployment notes ---
    print("\n  SIM-TO-REAL DEPLOYMENT PIPELINE:")
    print("  ─────────────────────────────────")
    print("  1. Train: python train_stair4_lidar.py")
    print("  2. Export: actor weights from logs/go2-stairs-v4-lidar/model_XXXX.pt")
    print("  3. On real Go2:")
    print("     a. Read L1 LIDAR point cloud (unitree_sdk2 ChannelSubscriber)")
    print("     b. Project to body-frame 5×3 grid:")
    if "x_points" in a_cfg:
        print(f"        x = {a_cfg['x_points']} m")
        print(f"        y = {a_cfg['y_points']} m")
    print(f"        = {na} height cells")
    print("     c. For each cell: max_z of LIDAR points within cell")
    print("     d. Subtract robot base z, clip to ±1.0m")
    print("     e. Concatenate with 49 proprioceptive obs → 64-dim input")
    print("     f. Feed to actor network → joint position targets")
    print("  ─────────────────────────────────\n")

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