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

from go2_env_test4 import Go2Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
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
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def get_cfgs():
    # ================================================================
    # DR maxima (HARD ceiling — curriculum ramps from easy to these)
    # ================================================================

    friction_enable = True
    friction_range = [0.4, 1.2]

    kp_kd_enable = True
    kp_nominal = 60.0
    kd_nominal = 2.0
    kp_range = [50.0, 70.0]
    kd_range = [2.0, 5.0]

    obs_noise_enable = True
    # FIX: obs_noise_level is the hard-max multiplier applied as:
    #   component * obs_scale * obs_noise_level
    # Was 0.2, making effective noise ~5x below hardware reality.
    # At 1.0, full curriculum gives official-grade noise:
    #   ang_vel: 0.2 * 0.25 * 1.0 = 0.05 obs-space → 0.2 rad/s raw ✓
    #   dof_vel: 1.5 * 0.05 * 1.0 = 0.075 obs-space → 1.5 rad/s raw ✓
    obs_noise_level = 1.0  # FIX: was 0.2
    obs_noise = {
        "ang_vel": 0.2,       # official: 0.2 rad/s
        "gravity": 0.05,      # official: 0.05
        "dof_pos": 0.01,      # official: 0.01 rad (was 0.02)
        "dof_vel": 1.5,       # FIX: was 1.0, official: 1.5 rad/s
    }

    action_noise_enable = True
    action_noise_std = 0.1  # FIX: was 0.3 (too aggressive). 0.1 is standard.

    push_enable = True
    push_interval_s = 2.0
    push_force_range = [-100.0, 100.0]

    init_pose_enable = True
    init_pos_z_range = [0.38, 0.45]      # slight z variation (was fixed 0.42)
    init_euler_range = [-5.0, 5.0]        # FIX: ±5° tilt (was 0.0 — no perturbation)

    # FIX: mass DR now enabled with curriculum scaling
    mass_enable = True  # was False
    mass_shift_range = [-0.5, 0.5]
    com_shift_range = [-0.02, 0.02]

    simulate_action_latency = True

    # ================================================================
    # Curriculum (metric-gated, your architecture preserved)
    # ================================================================
    curriculum_enable = True
    curriculum_cfg = {
        "enabled": curriculum_enable,

        "level_init": 0.10,
        "level_min": 0.0,
        "level_max": 1.0,

        "ema_alpha": 0.03,

        # FIX: ready_timeout_rate 0.90 → 0.70 (was too strict, kept curriculum stuck)
        "ready_timeout_rate": 0.70,   
        "ready_tracking": 0.70,
        "ready_fall_rate": 0.30,
        "ready_streak": 2,

        "hard_fall_rate": 0.35,
        "hard_streak": 2,

        "step_up": 0.02,
        "step_down": 0.03,
        "cooldown_updates": 2,

        "update_every_episodes": 4096,

        "mix_prob_current": 0.80,
        "mix_level_low": 0.00,
        "mix_level_high": 0.50,

        "friction_easy": [0.7, 0.8],
        "kp_easy": [0.90 * kp_nominal, 1.10 * kp_nominal],
        "kd_easy": [0.75 * kd_nominal, 1.25 * kd_nominal],

        # NEW: mass DR easy ranges (small perturbation at start)
        "mass_shift_easy": [-0.1, 0.1],
        "com_shift_easy": [-0.005, 0.005],

        "push_start": 0.30,
        "push_interval_easy_s": 6.0,
    }

    # ================================================================
    # Environment config
    # ================================================================
    env_cfg = {
        "num_actions": 12,

        "kp": kp_nominal,
        "kd": kd_nominal,

        "simulate_action_latency": simulate_action_latency,

        # Foot link names — must match URDF exactly.
        # Used by feet_air_time reward. If these don't match your URDF,
        # the env will raise a RuntimeError listing available link names.
        "foot_names": ["FR_calf", "FL_calf", "RR_calf", "RL_calf"],
        "foot_contact_threshold": 3.0,  # N, force threshold for ground contact

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

        "termination_if_roll_greater_than": 20,   # degrees (your original, intentionally tight)
        "termination_if_pitch_greater_than": 20,   # degrees
        "termination_if_z_vel_greater_than": 100.0,
        "termination_if_y_vel_greater_than": 100.0,

        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,  # FIX: was 4.0, official uses 10s
        "action_scale": 0.25,
        "clip_actions": 100.0,

        "curriculum": curriculum_cfg,
    }

    # Apply DR flags
    if friction_enable:
        env_cfg["friction_range"] = friction_range
    if kp_kd_enable:
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
    if init_pose_enable:
        env_cfg["init_pos_z_range"] = init_pos_z_range
        env_cfg["init_euler_range"] = init_euler_range
    if mass_enable:
        env_cfg["mass_shift_range"] = mass_shift_range
        env_cfg["com_shift_range"] = com_shift_range

    # ================================================================
    # Observation config
    # ================================================================
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,   # keeping your value (official legged_gym also uses 0.25)
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
        "feet_height_target": 0.075,
        "feet_air_time_target": 0.5,  # NEW: target swing duration per foot (seconds)

        "reward_scales": {
            # --- Tracking (increased to match official weighting) ---
            "tracking_lin_vel": 1.5,       # was 1.0, official: 1.5
            "tracking_ang_vel": 0.75,      # was 0.2, official: 0.75

            # --- Regularization ---
            "lin_vel_z": -2.0,             # was -1.0, official: -2.0
            "base_height": -50.0,          # kept (your value for Go2)
            "action_rate": -0.01,          # was -0.005, official: -0.01
            "similar_to_default": -0.1,    # was -0.1, official: ~-0.5 to -0.7
            "orientation_penalty": -2.5,   # was -1.0, bumped (covers flat_orientation)
            "dof_acc": -2.5e-7,            # was 0.0 (disabled), now active
            "dof_vel": -5e-4,             # was 0.0 (disabled), now active

            # --- NEW: gait quality ---
            "feet_air_time": 2.0
            ,          # official: 0.1 (encourages proper swing timing)

            # --- NEW: energy efficiency ---
            "energy": -2e-5,               # 

            # --- Existing ---
            "torque_load": -0.001,

            # --- Standing (activate if using standing envs) ---
            "stand_still": 0.0,           # was 0.0, now penalizes joint deviation when stopped
        },
    }

    # ================================================================
    # Command config — with curriculum + omnidirectional
    # ================================================================
    command_cfg = {
        "num_commands": 3,

        # FULL ranges (hard max — curriculum ramps from 10% to 100%)
        "lin_vel_x_range": [-1.0, 1.0],   # FIX: was [0.0, 1.0] forward-only
        "lin_vel_y_range": [-0.3, 0.3],    # FIX: was [0, 0] — no lateral
        "ang_vel_range": [-1.0, 1.0],      # FIX: was [0, 0] — no turning

        # NEW: command curriculum (ranges expand with curriculum level)
        "cmd_curriculum": True,
        "cmd_curriculum_start_frac": 0.1,  # start at 10% of full range

        # NEW: standing environments (10% of envs get zero commands)
        "rel_standing_envs": 0.1,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ================================================================
    # Print config summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  TRAINING CONFIG (v3: fixed noise + cmd curriculum + new rewards)")
    print("=" * 70)

    dr_items = {
        "Friction (HARD)":        ("friction_range",   lambda: str(env_cfg["friction_range"])),
        "Kp range (HARD)":        ("kp_range",         lambda: str(env_cfg["kp_range"])),
        "Kd range (HARD)":        ("kd_range",         lambda: str(env_cfg["kd_range"])),
        "Obs noise (HARD)":       ("obs_noise",        lambda: f'level={env_cfg.get("obs_noise_level", 0.0)}  {env_cfg["obs_noise"]}'),
        "Action noise (HARD)":    ("action_noise_std",  lambda: f'std={env_cfg["action_noise_std"]} rad'),
        "Pushes (HARD)":          ("push_force_range",  lambda: f'{env_cfg["push_force_range"]} N  every {env_cfg["push_interval_s"]}s'),
        "Mass shift (HARD)":      ("mass_shift_range",  lambda: f'{env_cfg["mass_shift_range"]} kg'),
        "CoM shift (HARD)":       ("com_shift_range",   lambda: f'{env_cfg["com_shift_range"]} m'),
        "Init pose":              ("init_euler_range",   lambda: f'z={env_cfg["init_pos_z_range"]}  euler=±{env_cfg["init_euler_range"][1]}°'),
    }
    for label, (key, fmt) in dr_items.items():
        status = f"ON   {fmt()}" if key in env_cfg else "OFF"
        print(f"  {label:22s}: {status}")

    latency = "1 step (fixed)" if env_cfg.get("simulate_action_latency", True) else "OFF"
    print(f"  {'Action latency':22s}: {latency}")

    # Curriculum
    cc = env_cfg.get("curriculum", {})
    print("-" * 70)
    print(f"  Curriculum enabled     : {cc.get('enabled', False)}")
    if cc.get("enabled", False):
        print(f"  level_init             : {cc.get('level_init')}")
        print(f"  update_every_episodes  : {cc.get('update_every_episodes')}")
        print(f"  ready thresholds       : timeout>={cc.get('ready_timeout_rate')}, "
              f"tracking>={cc.get('ready_tracking')}, fall<={cc.get('ready_fall_rate')}")
        print(f"  hard threshold         : fall>={cc.get('hard_fall_rate')}")
        print(f"  step_up / step_down    : {cc.get('step_up')} / {cc.get('step_down')}")
        print(f"  mix_prob_current       : {cc.get('mix_prob_current')}")
        print(f"  friction_easy          : {cc.get('friction_easy')}")
        print(f"  kp_easy / kd_easy      : {cc.get('kp_easy')} / {cc.get('kd_easy')}")
        print(f"  mass_shift_easy        : {cc.get('mass_shift_easy')}")
        print(f"  push_start / easy_int  : {cc.get('push_start')} / {cc.get('push_interval_easy_s')}")

    # Commands
    print("-" * 70)
    print(f"  Commands:")
    print(f"    lin_vel_x            : {command_cfg['lin_vel_x_range']}")
    print(f"    lin_vel_y            : {command_cfg['lin_vel_y_range']}")
    print(f"    ang_vel              : {command_cfg['ang_vel_range']}")
    print(f"    cmd_curriculum       : {command_cfg.get('cmd_curriculum', False)}")
    print(f"    start_frac           : {command_cfg.get('cmd_curriculum_start_frac', 'N/A')}")
    print(f"    standing_envs        : {command_cfg.get('rel_standing_envs', 0.0)}")

    # Rewards
    print("-" * 70)
    print(f"  Rewards (pre-dt scaling):")
    for name, scale in reward_cfg["reward_scales"].items():
        marker = "NEW" if name in ["feet_air_time", "energy"] else "   "
        marker2 = "FIX" if name in ["tracking_lin_vel", "tracking_ang_vel", "lin_vel_z",
                                      "action_rate", "similar_to_default", "orientation_penalty",
                                      "dof_acc", "dof_vel", "stand_still"] else "   "
        m = marker if marker.strip() else marker2
        print(f"    [{m}] {name:25s}: {scale}")
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