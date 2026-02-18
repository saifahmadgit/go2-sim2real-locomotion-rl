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

from go2_env_Omni_walk_16output import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.003,  # FIX: was 0.01, IsaacLab standard
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,  # FIX: was 1e-4, standard is 1e-3
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
        "num_steps_per_env": 24,  # FIX: was 36, standard is 24
        "save_interval": 1000,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict


def get_cfgs():
    # ================================================================
    # PER-LEG STIFFNESS (PLS) — variable stiffness control
    # Paper: "Variable Stiffness for Robust Locomotion through RL"
    # Policy outputs 4 extra actions (one Kp per leg).
    # Kd = 0.2 * sqrt(Kp) automatically (Eq. 6 in paper).
    # Leg mapping: FR(joints 0-2), FL(3-5), RR(6-8), RL(9-11)
    # ================================================================

    pls_enable = True
    pls_kp_range = [10.0, 70.0]  # paper range (not 80)
    pls_kp_default = 40.0
    pls_kp_action_scale = 20.0

    num_pos_actions = 12
    num_stiffness_actions = 4 if pls_enable else 0
    num_actions_total = num_pos_actions + num_stiffness_actions

    # ================================================================
    # Go2 joint torque limits (from URDF <effort> field)
    # Critical for sim2real: prevents commanding impossible torques
    # GO-M8010-6 motor: ~23.7 Nm nominal
    # Largest Go2 joint (calf): ~45 Nm peak
    # ================================================================
    # Per-joint torque limits [FR_hip, FR_thigh, FR_calf, FL_hip, ...]
    # If your URDF has specific values, use those instead.
    torque_limits = [23.7, 23.7, 45.0] * 4  # hip, thigh, calf × 4 legs

    # ================================================================
    # DR maxima (HARD ceiling — curriculum ramps from easy to these)
    # ================================================================

    friction_enable = True
    friction_range = [0.3, 1.25]  # paper Table I

    kp_kd_factor_enable = True
    kp_factor_range = [0.8, 1.2]  # paper Table I
    kd_factor_range = [0.8, 1.2]  # paper Table I

    # Nominal Kp/Kd — used only when PLS is DISABLED
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
    init_pos_z_range = [0.38, 0.45]
    init_euler_range = [-5.0, 5.0]

    # Mass DR — paper: U(-1.0, 3.0) kg trunk
    mass_enable = True
    mass_shift_range = [-1.0, 3.0]
    com_shift_range = [-0.03, 0.03]

    # Per-leg (hip link) mass randomisation — paper Table I
    leg_mass_enable = True
    leg_mass_shift_range = [-0.5, 0.5]

    # FIX: dynamic payload DISABLED — it's global (not per-env) and
    # paper doesn't use it. Creates massive nonstationarity.
    dynamic_payload_enable = False

    simulate_action_latency = True

    # Gravity offset DR — paper Table I
    gravity_offset_enable = True
    gravity_offset_range = [-1.0, 1.0]

    # Motor strength DR — paper Table I
    motor_strength_enable = True
    motor_strength_range = [0.9, 1.1]

    # System delay — paper Table I: U(0, 15ms) → 0-1 steps at dt=0.02
    delay_enable = True
    min_delay_steps = 0
    max_delay_steps = 1

    # ================================================================
    # Curriculum (metric-gated)
    # ================================================================
    curriculum_enable = True
    curriculum_cfg = {
        "enabled": curriculum_enable,
        "level_init": 0.10,
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
        # Gravity offset easy
        "gravity_offset_easy": [-0.2, 0.2],
        # Motor strength easy
        "motor_strength_easy": [0.97, 1.03],
        # Push curriculum
        "push_start": 0.0,
        "push_interval_easy_s": 10.0,
        # Delay curriculum
        "delay_easy_max_steps": 0,
        # FIX: friction/mass DR update frequency
        # Only re-randomize global DR every N resets to reduce nonstationarity
        "global_dr_update_interval": 200,
    }

    # ================================================================
    # Environment config
    # ================================================================
    env_cfg = {
        "num_actions": num_actions_total,
        "num_pos_actions": num_pos_actions,
        # PLS (Per-Leg Stiffness)
        "pls_enable": pls_enable,
        "pls_kp_range": pls_kp_range,
        "pls_kp_default": pls_kp_default,
        "pls_kp_action_scale": pls_kp_action_scale,
        # Nominal PD (used when PLS is disabled)
        "kp": kp_nominal,
        "kd": kd_nominal,
        # FIX: torque limits for sim2real safety
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
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 5.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,  # NOTE: kept at 100 (standard), torque clamp is the real safety net
        "curriculum": curriculum_cfg,
    }

    # Apply DR flags
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
    # dynamic_payload_enable is False — not added to env_cfg
    if gravity_offset_enable:
        env_cfg["gravity_offset_range"] = gravity_offset_range
    if motor_strength_enable:
        env_cfg["motor_strength_range"] = motor_strength_range
    if delay_enable:
        env_cfg["min_delay_steps"] = min_delay_steps
        env_cfg["max_delay_steps"] = max_delay_steps

    # ================================================================
    # Observation config
    # ================================================================
    # Actor obs: ang_vel(3) + gravity(3) + commands(3) + dof_pos(12)
    #            + dof_vel(12) + actions(num_actions_total)
    num_obs = 3 + 3 + 3 + 12 + 12 + num_actions_total

    # Privileged critic obs = actor_obs + hidden DR params + ground truth
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
    # Reward config
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "feet_air_time_target": 0.1,
        "reward_scales": {
            # --- Tracking (paper weights) ---
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.8,  # FIX: was 0.75, paper uses 0.8
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
            "feet_air_time": 0.2,
            "foot_slip": -0.1,
            "foot_clearance": -0.1,
            "joint_tracking": -0.1,
            # --- Energy / torque ---
            "energy": 0.0,  ###-2e-5,
            "torque_load": 0.0,  ###########-0.001,
            # --- Standing (active when commands ≈ 0) ---
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
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-v6")
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
    print("  TRAINING CONFIG (v6: 5-bug-fix + sim2real)")
    print("=" * 70)

    pls = env_cfg.get("pls_enable", False)
    print(
        f"  {'PLS (Per-Leg Stiffness)':30s}: {'ON   Kp=' + str(env_cfg['pls_kp_range']) + '  scale=' + str(env_cfg['pls_kp_action_scale']) if pls else 'OFF'}"
    )
    print(f"  {'Action space':30s}: {env_cfg['num_actions']} ({'12 pos + 4 stiffness' if pls else '12 pos'})")
    print(f"  {'Torque limits':30s}: {env_cfg.get('torque_limits', 'NONE — DANGEROUS')}")
    print(f"  {'Actor obs':30s}: {obs_cfg['num_obs']}")
    print(f"  {'Privileged critic obs':30s}: {obs_cfg['num_privileged_obs']}")

    # PPO params
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

    # Curriculum
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

    # Rewards
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
