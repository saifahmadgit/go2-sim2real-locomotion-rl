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

from go2_env_stair import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations, resume_path=None):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.003,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,  # Lower LR for fine-tuning from walk policy
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
    # PLS config (same as original)
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
    # Height scan config for privileged critic observations
    # ================================================================
    height_scan_cfg = {
        "num_x": 11,  # 11 points forward/backward
        "num_y": 7,  # 7 points lateral
        "x_range": [-0.5, 0.5],  # 1.0m coverage in x (body frame)
        "y_range": [-0.3, 0.3],  # 0.6m coverage in y (body frame)
    }
    num_height_scan = height_scan_cfg["num_x"] * height_scan_cfg["num_y"]  # 77

    # ================================================================
    # Terrain config — sim2real targeted: 39cm depth, up to 15cm height
    # ================================================================
    terrain_cfg = {
        "enabled": True,
        # Heightfield resolution
        "horizontal_scale": 0.05,  # 5cm per cell (good for stair edges)
        "vertical_scale": 0.005,  # 0.5cm per height unit
        # Grid layout
        "num_difficulty_rows": 13,  # 13 difficulty levels (0-12)
        "row_width_m": 6.0,  # 6m wide per row (enough for lateral motion)
        # Stair geometry — matched to real stairs for sim2real
        "step_depth_m": 0.39,  # 39cm tread depth (real stair measurement)
        "num_steps": 6,  # 6 steps up + 6 steps down per flight
        "num_flights": 4,  # 4 up-down cycles per row
        # Step height range across difficulty rows
        "step_height_min": 0.02,  # Row 0: 2cm (barely a bump)
        "step_height_max": 0.15,  # Row 12: 15cm (real stair riser height)
        # Flat sections between stair flights
        "flat_before_m": 2.0,  # Flat runway before first flight
        "flat_top_m": 1.5,  # Flat platform between up/down
        "flat_gap_m": 1.5,  # Flat gap between flights
        "flat_after_m": 2.0,  # Flat runway after last flight
        # Height scan for privileged critic observations
        "height_scan": height_scan_cfg,
    }

    # ================================================================
    # DR config (same as original, slightly relaxed for terrain learning)
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
    # Curriculum (RELAXED thresholds for stair learning)
    # ================================================================
    curriculum_cfg = {
        "enabled": True,
        "level_init": 0.65,  # Start at ~row 8 (model already handles row 9 of old 10-row terrain)
        "level_min": 0.0,
        "level_max": 1.0,
        "ema_alpha": 0.03,
        # CONSERVATIVE thresholds (prevent runaway advancement)
        "ready_timeout_rate": 0.60,  # 60% survive to timeout
        "ready_tracking": 0.45,  # Lower tracking bar (stairs slow robot)
        "ready_fall_rate": 0.35,  # Up to 35% falls OK
        "ready_streak": 5,  # 5 consecutive checks (was 3 — too fast)
        "hard_fall_rate": 0.40,  # Retreat sooner (was 0.50)
        "hard_streak": 2,
        "step_up": 0.01,  # Slower advance (was 0.02 — caused collapse)
        "step_down": 0.03,  # Fast retreat when struggling
        "cooldown_updates": 5,  # More cooldown between changes (was 3)
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
        "push_start": 0.3,  # Pushes start at level 0.3
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
        "termination_if_roll_greater_than": 45,
        "termination_if_pitch_greater_than": 45,
        "termination_if_z_vel_greater_than": 100.0,
        "termination_if_y_vel_greater_than": 100.0,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 25.0,  # Longer episodes for stair traversal
        "resampling_time_s": 5.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,
        # --- Terrain ---
        "terrain": terrain_cfg,
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
        # --- Two-phase DR schedule ---
        # Phase 1: Robot learns stairs with minimal DR disturbance
        # Phase 2: Once stairs are mastered, DR ramps up for robustness
        "dr_schedule": {
            "phase1_level": 0.15,  # Mild DR during stair learning
            "terrain_gate": 0.50,  # DR starts ramping at terrain level 0.50 (skill learned)
        },
    }

    # ================================================================
    # Observation config
    # ================================================================
    num_obs = 3 + 3 + 3 + 12 + 12 + num_actions_total

    # Privileged: base obs + lin_vel(3) + friction(1) + kp_factors(12) + kd_factors(12)
    #             + motor_strength(12) + mass_shift(1) + com_shift(3) + leg_mass(4)
    #             + gravity_offset(3) + push_force(3) + delay(1) + terrain_row(1)
    #             + height_scan(77)                                       ← NEW
    num_privileged_extra = (
        3
        + 1
        + 12
        + 12
        + 12
        + 1
        + 3
        + 4
        + 3
        + 3
        + 1
        + 1  # terrain_row
        + num_height_scan
    )  # height scan (77)
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
    # Reward config — tuned for stair climbing with TERRAIN-RELATIVE fixes
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.17,  # was 0.12 for the version which works in sim and in real works 1 out of 5 times
        "feet_air_time_target": 0.1,
        "lin_vel_z_deadzone": 0.15,  # NEW: allow 0.15 m/s z-vel without penalty (stair climbing)
        "reward_scales": {
            # --- Tracking ---
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,
            # --- Forward progress (NEW — direct stair climbing incentive) ---
            "forward_progress": 0.4,  # Reward +x displacement per step
            # --- Regularisation (ADJUSTED for terrain-relative rewards) ---
            "lin_vel_z": -1.0,  # Reduced (was -1.5) + deadzone handles small z-vel
            "base_height": -0.1,  # Gentle hint only (robot needs freedom to bob on stairs)
            "action_rate": -0.01,
            "similar_to_default": -0.05,  # Relaxed (was -0.1, stop penalizing stair postures)
            "orientation_roll_only": -5.0,  # Only penalize ROLL, allow pitch (stair tilt)
            "dof_acc": -2.5e-7,
            "dof_vel": -5e-4,
            "ang_vel_xy": -0.05,
            # --- Gait quality (critical for stairs!) ---
            "feet_air_time": 0.2,
            "foot_slip": -0.15,
            "foot_clearance": -0.5,  # INCREASED from -0.2 (THE key stair reward, now terrain-relative)
            "joint_tracking": -0.1,
            # --- Energy ---
            "energy": 0.0,
            "torque_load": 0.0,
            # --- Standing ---
            "stand_still": -0.5,
            "stand_still_vel": -2.0,
            "feet_stance": -0.3,
        },
    }

    # ================================================================
    # Command config — bias forward motion for stair encounters
    # ================================================================
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.8],  # Forward only (stairs are in +x)
        "lin_vel_y_range": [0.0, 0.0],  # No lateral (corridor terrain)
        "ang_vel_range": [0.0, 0.0],  # No turning (corridor terrain)
        "cmd_curriculum": False,  # No need — commands are fixed
        "compound_commands": True,
        "rel_standing_envs": 0.05,  # 5% standing (so robot learns zero-command)
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairs-v5-39cm")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., logs/go2-stairs-v4/model_5000.pt)",
    )
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
    print("  TRAINING CONFIG — STAIR CLIMBING v5 (39cm depth, 15cm height — sim2real)")
    print("=" * 70)

    tc = env_cfg.get("terrain", {})
    if tc.get("enabled"):
        print(f"  {'Terrain':30s}: ENABLED")
        print(f"    Difficulty rows      : {tc['num_difficulty_rows']}")
        print(f"    Step height range    : {tc['step_height_min'] * 100:.0f}cm → {tc['step_height_max'] * 100:.0f}cm")
        print(f"    Steps per flight     : {tc['num_steps']}")
        print(f"    Tread depth          : {tc['step_depth_m'] * 100:.0f}cm (39cm real stair)")
        print(f"    Heightfield res      : h={tc['horizontal_scale']}m  v={tc['vertical_scale']}m")
        hs = tc.get("height_scan", {})
        if hs:
            print(
                f"    Height scan          : {hs.get('num_x', 0)}×{hs.get('num_y', 0)} = "
                f"{hs.get('num_x', 0) * hs.get('num_y', 0)} points (privileged critic obs)"
            )
    else:
        print(f"  {'Terrain':30s}: DISABLED (flat ground)")

    pls = env_cfg.get("pls_enable", False)
    print(f"  {'PLS':30s}: {'ON' if pls else 'OFF'}")
    print(f"  {'Action space':30s}: {env_cfg['num_actions']}")
    print(f"  {'Actor obs':30s}: {obs_cfg['num_obs']}")
    print(f"  {'Privileged critic obs':30s}: {obs_cfg['num_privileged_obs']}")
    print(f"  {'Episode length':30s}: {env_cfg['episode_length_s']}s")

    alg = train_cfg["algorithm"]
    print(f"  {'Learning rate':30s}: {alg['learning_rate']}")
    if args.resume:
        print(f"  {'Resuming from':30s}: {args.resume}")
        print("  NOTE: Actor loads from checkpoint, critic re-initialises (dim mismatch expected)")

    # Rewards
    print("-" * 70)
    print("  Rewards (pre-dt scaling):")
    stair_tuned = {
        "foot_clearance",
        "lin_vel_z",
        "base_height",
        "similar_to_default",
        "forward_progress",
        "orientation_roll_only",
    }
    for name, scale in reward_cfg["reward_scales"].items():
        marker = " ← STAIR-TUNED" if name in stair_tuned else ""
        print(f"    {name:25s}: {scale}{marker}")

    print(f"\n  feet_height_target : {reward_cfg['feet_height_target']}m ← INCREASED for stairs")
    print(f"  lin_vel_z_deadzone : {reward_cfg['lin_vel_z_deadzone']}m/s ← allow z-vel on stairs")
    print("  base_height/foot_clearance are now TERRAIN-RELATIVE")
    print("  orientation_roll_only: penalizes ROLL only (pitch allowed for stairs)")

    # DR schedule
    dr_sched = env_cfg.get("dr_schedule", {})
    print("-" * 70)
    if dr_sched:
        print("  DR schedule          : TWO-PHASE")
        print(
            f"    Phase 1 (stairs)   : DR level = {dr_sched.get('phase1_level', '?')} "
            f"(until terrain >= {dr_sched.get('terrain_gate', '?')})"
        )
        print(f"    Phase 2 (harden)   : DR ramps {dr_sched.get('phase1_level', '?')} → 1.0")
    else:
        print("  DR schedule          : coupled to terrain (old behaviour)")

    # Curriculum
    cc = env_cfg.get("curriculum", {})
    print("-" * 70)
    print(f"  Curriculum enabled     : {cc.get('enabled', False)}")
    if cc.get("enabled"):
        print(f"    level_init           : {cc.get('level_init')}")
        print(
            f"    ready thresholds     : timeout>={cc.get('ready_timeout_rate')}, "
            f"tracking>={cc.get('ready_tracking')}, fall<={cc.get('ready_fall_rate')}"
        )
        print(f"    ready_streak         : {cc.get('ready_streak')} (conservative)")
        print(f"    step_up              : {cc.get('step_up')} (slow advance)")
    print("  Spawn distribution     : 40% frontier / 30% near / 30% easy")

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
