"""
go2_train_terrain.py — Fine-tune Go2 walking policy on terrain.

Uses TerrainManager for terrain generation.
50/50 flat vs custom heightmap (random_uniform bumps) via subterrain grid.

Based on go2_train_test8.py (the checkpoint's actual training script).
Changes from the original:
  1. Ground: flat URDF plane → TerrainManager (50% flat / 50% random_uniform)
  2. foot_clearance reward: -0.1 → -0.3
  3. feet_air_time reward:  0.2 → 0.3
  4. Resume from existing checkpoint (full resume: weights + optimizer)
  5. level_init: 0.10 → 0.35
  6. base_init_pos z: 0.42 → 0.45
  7. init_pos_z_range: [0.38, 0.45] → [0.43, 0.48]

Usage:
    python go2_train_terrain.py -e go2-terrain-v1 \\
        --resume_path logs/go2-DR-Omni-kp-learn-DiffTerrain/model_188000.pt \\
        --max_iterations 220000
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata

import numpy as np

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from go2_env_Omni_kplearn_varied_terrain import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# =====================================================================
# Custom heightmap generation (same as eval script)
# =====================================================================


def generate_heightmap(min_h, max_h, size_x, size_y, h_scale, smoothing=3):
    """Generate a numpy heightmap with exact height range in meters."""
    res_x = int(size_x / h_scale)
    res_y = int(size_y / h_scale)

    if min_h == 0.0 and max_h == 0.0:
        hf = np.zeros((res_y, res_x), dtype=np.float32)
    else:
        hf = np.random.uniform(min_h, max_h, (res_y, res_x)).astype(np.float32)
        if smoothing > 1:
            try:
                from scipy.ndimage import uniform_filter

                hf = uniform_filter(hf, size=smoothing).astype(np.float32)
            except ImportError:
                pass
    return hf


def get_train_cfg(exp_name, max_iterations, resume_path=None):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
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
            "resume_path": resume_path if resume_path else None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 1000,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


# =====================================================================
# TERRAIN MODE SELECTOR — choose between subterrain grid or heightmap
# =====================================================================
# Set to "grid" for subterrain grid (50% flat / 50% random_uniform tiles)
# Set to "heightmap" for custom numpy heightmap (50% flat / 50% bumpy area)
TERRAIN_MODE = "heightmap"

# Heightmap-specific settings (only used when TERRAIN_MODE == "heightmap")
HEIGHTMAP_SIZE_X = 15.0  # terrain width in meters
HEIGHTMAP_SIZE_Y = 15.0  # terrain depth in meters
HEIGHTMAP_H_SCALE = 0.05  # meters per pixel (smaller = finer detail)
HEIGHTMAP_MIN_H = -0.08  # min bump height in meters
HEIGHTMAP_MAX_H = 0.08  # max bump height in meters
HEIGHTMAP_SMOOTHING = 3  # uniform_filter kernel size


def make_terrain_cfg_grid():
    """
    50/50 flat vs bumpy using subterrain grid.
    2x2 grid: 2 flat tiles + 2 random_uniform tiles = 50% each.
    """
    return {
        "enabled": True,
        "subterrain_types": [
            ["flat_terrain", "random_uniform_terrain"],
            ["random_uniform_terrain", "flat_terrain"],
        ],
        "subterrain_size": (8.0, 8.0),
        "horizontal_scale": 0.25,
        "vertical_scale": 0.005,
        "randomize": True,
        "spawn_height_offset": 0.05,
        "boundary_margin": 1.0,
        "subterrain_parameters": {
            "random_uniform_terrain": {
                "min_height": -0.08,
                "max_height": 0.08,
            },
        },
    }


def make_terrain_cfg_heightmap():
    """
    50/50 flat vs bumpy using custom numpy heightmap.
    Left half is flat (zeros), right half has random bumps.
    """
    res_x = int(HEIGHTMAP_SIZE_X / HEIGHTMAP_H_SCALE)
    res_y = int(HEIGHTMAP_SIZE_Y / HEIGHTMAP_H_SCALE)

    hf = np.zeros((res_y, res_x), dtype=np.float32)

    # Right half: random bumps
    half_x = res_x // 2
    bumpy = np.random.uniform(
        HEIGHTMAP_MIN_H,
        HEIGHTMAP_MAX_H,
        (res_y, res_x - half_x),
    ).astype(np.float32)

    if HEIGHTMAP_SMOOTHING > 1:
        try:
            from scipy.ndimage import uniform_filter

            bumpy = uniform_filter(bumpy, size=HEIGHTMAP_SMOOTHING).astype(np.float32)
        except ImportError:
            pass

    hf[:, half_x:] = bumpy

    print("\n[Terrain] Custom 50/50 heightmap:")
    print(f"  shape      : {hf.shape}")
    print(f"  flat half  : columns 0-{half_x - 1}")
    print(f"  bumpy half : columns {half_x}-{res_x - 1}")
    print(f"  bump range : [{bumpy.min():.4f}, {bumpy.max():.4f}] m")

    return {
        "enabled": True,
        "height_field": hf,
        "horizontal_scale": HEIGHTMAP_H_SCALE,
        "vertical_scale": 1.0,  # heights already in meters
        "spawn_height_offset": 0.05,
        "boundary_margin": 1.0,
    }


def get_cfgs():
    # ================================================================
    # PLS config — IDENTICAL to go2_train_test8.py
    # ================================================================
    pls_enable = True
    pls_kp_range = [10.0, 70.0]
    pls_kp_default = 40.0
    pls_kp_action_scale = 20.0

    num_pos_actions = 12
    num_stiffness_actions = 4 if pls_enable else 0
    num_actions_total = num_pos_actions + num_stiffness_actions

    # ================================================================
    # Torque limits — IDENTICAL
    # ================================================================
    torque_limits = [23.7, 23.7, 45.0] * 4

    # ================================================================
    # DR maxima — IDENTICAL
    # ================================================================
    friction_enable = True
    friction_range = [0.3, 1.25]

    kp_kd_factor_enable = True
    kp_factor_range = [0.8, 1.2]
    kd_factor_range = [0.8, 1.2]

    kp_nominal = 60.0
    kd_nominal = 2.0
    kp_range = [50.0, 70.0]
    kd_range = [1.0, 5.0]

    obs_noise_enable = True
    obs_noise_level = 1.0
    obs_noise = {
        "ang_vel": 0.2,
        "gravity": 0.05,
        "dof_pos": 0.01,
        "dof_vel": 1.5,
    }

    action_noise_enable = True
    action_noise_std = 0.1

    push_enable = True
    push_interval_s = 5.0
    push_force_range = [-150.0, 150.0]
    push_duration_s = [0.05, 0.2]

    init_pose_enable = True
    # ----- CHANGED: z range raised for terrain bumps -----
    init_pos_z_range = [0.43, 0.48]
    init_euler_range = [-5.0, 5.0]

    mass_enable = True
    mass_shift_range = [-1.0, 3.0]
    com_shift_range = [-0.03, 0.03]

    leg_mass_enable = True
    leg_mass_shift_range = [-0.5, 0.5]

    dynamic_payload_enable = False

    simulate_action_latency = True

    gravity_offset_enable = True
    gravity_offset_range = [-1.0, 1.0]

    motor_strength_enable = True
    motor_strength_range = [0.9, 1.1]

    delay_enable = True
    min_delay_steps = 0
    max_delay_steps = 1

    # ================================================================
    # Curriculum — IDENTICAL except level_init
    # ================================================================
    curriculum_enable = True
    curriculum_cfg = {
        "enabled": curriculum_enable,
        # ----- CHANGED: start higher since base skill exists -----
        "level_init": 0.20,
        "level_min": 0.0,
        "level_max": 1.0,
        "ema_alpha": 0.03,
        "ready_timeout_rate": 0.80,
        "ready_tracking": 0.75,
        "ready_fall_rate": 0.15,
        "ready_streak": 4,
        "hard_fall_rate": 0.25,
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
        "push_start": 0.0,
        "push_interval_easy_s": 10.0,
        "delay_easy_max_steps": 0,
        "global_dr_update_interval": 200,
    }

    # ================================================================
    # Terrain config — 50/50 flat vs custom terrain
    # ================================================================
    if TERRAIN_MODE == "heightmap":
        terrain_cfg = make_terrain_cfg_heightmap()
    else:
        terrain_cfg = make_terrain_cfg_grid()

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
        "simulate_action_latency": simulate_action_latency,
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
        # ----- CHANGED: z raised 0.42 → 0.45 for terrain bumps -----
        "base_init_pos": [0.0, 0.0, 0.45],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 5.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,
        "curriculum": curriculum_cfg,
        # Terrain — built by TerrainManager
        "terrain": terrain_cfg,
    }

    # Apply DR flags — IDENTICAL
    if friction_enable:
        env_cfg["friction_range"] = friction_range
    if kp_kd_factor_enable:
        env_cfg["kp_factor_range"] = kp_factor_range
        env_cfg["kd_factor_range"] = kd_factor_range
    env_cfg["kp_range"] = kp_range
    env_cfg["kd_range"] = kd_range
    if obs_noise_enable:
        env_cfg["obs_noise"] = obs_noise
        env_cfg["obs_noise_level"] = obs_noise_level
    if action_noise_enable:
        env_cfg["action_noise_std"] = action_noise_std
    if push_enable:
        env_cfg["push_interval_s"] = push_interval_s
        env_cfg["push_force_range"] = push_force_range
        env_cfg["push_duration_s"] = push_duration_s
    if init_pose_enable:
        env_cfg["init_pos_z_range"] = init_pos_z_range
        env_cfg["init_euler_range"] = init_euler_range
    if mass_enable:
        env_cfg["mass_shift_range"] = mass_shift_range
        env_cfg["com_shift_range"] = com_shift_range
    if leg_mass_enable:
        env_cfg["leg_mass_shift_range"] = leg_mass_shift_range
    if gravity_offset_enable:
        env_cfg["gravity_offset_range"] = gravity_offset_range
    if motor_strength_enable:
        env_cfg["motor_strength_range"] = motor_strength_range
    if delay_enable:
        env_cfg["min_delay_steps"] = min_delay_steps
        env_cfg["max_delay_steps"] = max_delay_steps

    # ================================================================
    # Observation config — IDENTICAL
    # ================================================================
    num_obs = 3 + 3 + 3 + 12 + 12 + num_actions_total
    num_privileged_extra = 3 + 1 + 12 + 12 + 12 + 1 + 3 + 4 + 3 + 3 + 1
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
    # Reward config — IDENTICAL except foot_clearance and feet_air_time
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "feet_air_time_target": 0.1,
        "reward_scales": {
            # --- Tracking ---
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,
            # --- Regularisation ---
            "lin_vel_z": -2.0,
            "base_height": -0.6,
            "action_rate": -0.01,
            "similar_to_default": -0.1,
            "orientation_penalty": -5.0,
            "dof_acc": -2.5e-7,
            "dof_vel": -5e-4,
            "ang_vel_xy": -0.05,
            # --- Gait quality ---
            # ----- CHANGED: feet_air_time 0.2 → 0.3 -----
            "feet_air_time": 0.3,
            "foot_slip": -0.1,
            # ----- CHANGED: foot_clearance -0.1 → -0.3 -----
            "foot_clearance": -0.3,
            "joint_tracking": -0.1,
            # --- Energy / torque ---
            "energy": 0.0,
            "torque_load": 0.0,
            # --- Standing ---
            "stand_still": -0.5,
            "stand_still_vel": -2.0,
            "feet_stance": -0.3,
        },
    }

    # ================================================================
    # Command config — IDENTICAL
    # ================================================================
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-1.0, 1.0],
        "cmd_curriculum": True,
        "cmd_curriculum_start_frac": 0.1,
        "compound_commands": True,
        "rel_standing_envs": 0.1,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-terrain-v1")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=220000)
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to .pt checkpoint (e.g. logs/go2-DR-Omni-kp-learn-DiffTerrain/model_188000.pt)",
    )
    args = parser.parse_args()

    # ==================================================================
    # CRITICAL: Validate checkpoint exists BEFORE building env.
    # ==================================================================
    if args.resume_path is not None:
        if not os.path.isfile(args.resume_path):
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"  CHECKPOINT NOT FOUND: '{args.resume_path}'\n"
                f"  Cannot resume training — check the path and try again.\n"
                f"  Without this check, rsl_rl would silently train from scratch!\n"
                f"{'=' * 60}"
            )
        print(f"[Resume] Checkpoint verified: {args.resume_path}")
        print(f"[Resume] File size: {os.path.getsize(args.resume_path) / 1e6:.1f} MB")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, args.resume_path)

    # Don't wipe logs when resuming into an existing experiment
    if args.resume_path is None and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ================================================================
    # Print config summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  TERRAIN FINE-TUNING — go2_train_terrain.py")
    print("  Changes from go2_train_test8.py:")
    print(f"    terrain mode: {TERRAIN_MODE}")
    tcfg = env_cfg["terrain"]
    if TERRAIN_MODE == "grid":
        n_flat = sum(row.count("flat_terrain") for row in tcfg["subterrain_types"])
        n_total = len(tcfg["subterrain_types"]) * len(tcfg["subterrain_types"][0])
        print(
            f"    grid: {len(tcfg['subterrain_types'])}x{len(tcfg['subterrain_types'][0])} "
            f"({n_flat} flat / {n_total - n_flat} bumpy = "
            f"{100 * n_flat / n_total:.0f}%/{100 * (n_total - n_flat) / n_total:.0f}%)"
        )
    else:
        print(f"    heightmap: {HEIGHTMAP_SIZE_X}x{HEIGHTMAP_SIZE_Y}m, bumps [{HEIGHTMAP_MIN_H}, {HEIGHTMAP_MAX_H}]m")
    print(f"    foot_clearance: -0.1 → {reward_cfg['reward_scales']['foot_clearance']}")
    print(f"    feet_air_time:   0.2 → {reward_cfg['reward_scales']['feet_air_time']}")
    print(f"    level_init:     0.10 → {env_cfg['curriculum']['level_init']}")
    print(f"    base_init_z:    0.42 → {env_cfg['base_init_pos'][2]}")
    print(f"    init_pos_z_range: → {env_cfg.get('init_pos_z_range', 'N/A')}")
    print(f"    resume: {args.resume_path or '(none)'}")
    print("=" * 70)

    pls = env_cfg.get("pls_enable", False)
    print(
        f"  {'PLS (Per-Leg Stiffness)':30s}: {'ON   Kp=' + str(env_cfg['pls_kp_range']) + '  scale=' + str(env_cfg['pls_kp_action_scale']) if pls else 'OFF'}"
    )
    print(f"  {'Action space':30s}: {env_cfg['num_actions']} ({'12 pos + 4 stiffness' if pls else '12 pos'})")
    print(f"  {'Torque limits':30s}: {env_cfg.get('torque_limits', 'NONE')}")
    print(f"  {'Actor obs':30s}: {obs_cfg['num_obs']}")
    print(f"  {'Privileged critic obs':30s}: {obs_cfg['num_privileged_obs']}")

    alg = train_cfg["algorithm"]
    print("-" * 70)
    print("  PPO:")
    print(f"    learning_rate        : {alg['learning_rate']}")
    print(f"    entropy_coef         : {alg['entropy_coef']}")
    print(f"    desired_kl           : {alg['desired_kl']}")
    print(f"    num_steps_per_env    : {train_cfg['num_steps_per_env']}")

    dr_items = {
        "Friction (GLOBAL)": ("friction_range", lambda: str(env_cfg["friction_range"])),
        "Kp factor DR (per-env)": ("kp_factor_range", lambda: str(env_cfg["kp_factor_range"])),
        "Kd factor DR (per-env)": ("kd_factor_range", lambda: str(env_cfg["kd_factor_range"])),
        "Obs noise": ("obs_noise", lambda: f"level={env_cfg.get('obs_noise_level', 0.0)}"),
        "Action noise": ("action_noise_std", lambda: f"std={env_cfg['action_noise_std']} rad"),
        "Pushes": ("push_force_range", lambda: f"{env_cfg['push_force_range']} N  every {env_cfg['push_interval_s']}s"),
        "Mass shift (GLOBAL)": ("mass_shift_range", lambda: f"{env_cfg['mass_shift_range']} kg"),
        "CoM shift (GLOBAL)": ("com_shift_range", lambda: f"{env_cfg['com_shift_range']} m"),
        "Leg mass shift (GLOBAL)": ("leg_mass_shift_range", lambda: f"{env_cfg['leg_mass_shift_range']} kg"),
        "Dynamic payload": ("dynamic_payload_range", lambda: "DISABLED"),
        "Gravity offset (per-env)": ("gravity_offset_range", lambda: f"{env_cfg['gravity_offset_range']} m/s²"),
        "Motor strength (per-env)": ("motor_strength_range", lambda: f"{env_cfg['motor_strength_range']}"),
        "Action delay": ("max_delay_steps", lambda: f"{env_cfg['min_delay_steps']}-{env_cfg['max_delay_steps']} steps"),
        "Init pose": (
            "init_euler_range",
            lambda: f"z={env_cfg['init_pos_z_range']}  euler=±{env_cfg['init_euler_range'][1]}°",
        ),
    }
    print("-" * 70)
    for label, (key, fmt) in dr_items.items():
        status = f"ON   {fmt()}" if key in env_cfg else "OFF"
        print(f"  {label:30s}: {status}")

    cc = env_cfg.get("curriculum", {})
    print("-" * 70)
    print(f"  Curriculum enabled     : {cc.get('enabled', False)}")
    if cc.get("enabled", False):
        print(f"  level_init             : {cc.get('level_init')}")
        print(
            f"  ready thresholds       : timeout>={cc.get('ready_timeout_rate')}, "
            f"tracking>={cc.get('ready_tracking')}, fall<={cc.get('ready_fall_rate')}"
        )
        print(f"  hard threshold         : fall>={cc.get('hard_fall_rate')}")
        print(f"  step_up / step_down    : {cc.get('step_up')} / {cc.get('step_down')}")
        print(f"  global_dr_update_int   : {cc.get('global_dr_update_interval', 'N/A')}")

    print("-" * 70)
    print("  Rewards (pre-dt scaling):")
    for name, scale in reward_cfg["reward_scales"].items():
        print(f"    {name:25s}: {scale}")
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
