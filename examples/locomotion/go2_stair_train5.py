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

from go2_env_stair5 import Go2Env, DEFAULT_TERRAIN_CFG


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
    # Height scan for privileged critic
    # ================================================================
    height_scan_cfg = {
        "num_x": 11,
        "num_y": 7,
        "x_range": [-0.5, 0.5],
        "y_range": [-0.3, 0.3],
    }
    num_height_scan = height_scan_cfg["num_x"] * height_scan_cfg["num_y"]  # 77

    # ================================================================
    # TERRAIN — UP-ONLY STAIR SPECIALIST
    #
    # CHANGED from v4:
    #   - up_only=True: monotonically ascending stairs, no descents
    #   - step_height_min: 0.10m (10cm, skip trivial bumps)
    #   - step_height_max: 0.20m (20cm, max residential riser)
    #   - step_depth_m: 0.28m (28cm tread, per user spec)
    #   - num_flights: 3 (3 ascending flights per row)
    #   - No flat_gap (not needed for up-only)
    # ================================================================
    terrain_cfg = dict(DEFAULT_TERRAIN_CFG)  # from go2_env_stair5.py
    terrain_cfg["height_scan"] = height_scan_cfg

    # ================================================================
    # DR — FIXED MODERATE (no two-phase schedule)
    #
    # CHANGED from v4:
    #   - Removed dr_schedule entirely (no two-phase)
    #   - Walking policy already knows robustness — just maintain it
    #   - No pushes (destabilise stair climbing disproportionately)
    #   - Fixed moderate ranges, not curriculum-coupled
    # ================================================================
    friction_range = [0.4, 1.0]         # Stairs can be slippery
    kp_factor_range = [0.9, 1.1]        # Mild
    kd_factor_range = [0.9, 1.1]        # Mild
    kp_nominal = 60.0
    kd_nominal = 2.0
    kp_range = [50.0, 70.0]
    kd_range = [1.0, 4.0]

    obs_noise = {
        "ang_vel": 0.2,
        "gravity": 0.05,
        "dof_pos": 0.01,
        "dof_vel": 1.5,
    }

    mass_shift_range = [-0.5, 1.5]      # Moderate (half of v4's hard range)
    com_shift_range = [-0.02, 0.02]     # Moderate
    leg_mass_shift_range = [-0.3, 0.3]  # Moderate
    gravity_offset_range = [-0.5, 0.5]  # Moderate
    motor_strength_range = [0.92, 1.08] # Mild

    # NO push forces — they destabilise stair climbing
    # Push can be added in a later hardening phase
    # push_force_range = None  (omitted from env_cfg entirely)

    # ================================================================
    # CURRICULUM — terrain difficulty only
    #
    # CHANGED from v4:
    #   - No DR coupling (DR is fixed, only terrain advances)
    #   - Slower advancement for harder stairs
    #   - Lower tracking bar (stairs slow the robot)
    #   - forward_progress now counts for curriculum tracking metric
    # ================================================================
    curriculum_cfg = {
        "enabled": True,

        "level_init": 0.0,              # Start at easiest (10cm steps)
        "level_min": 0.0,
        "level_max": 1.0,

        "ema_alpha": 0.03,

        "ready_timeout_rate": 0.55,     # 55% survive to timeout
        "ready_tracking": 0.30,         # Low bar (forward_progress is slower)
        "ready_fall_rate": 0.40,        # Up to 40% falls OK
        "ready_streak": 5,              # 5 consecutive checks

        "hard_fall_rate": 0.50,         # Retreat at 50% falls
        "hard_streak": 2,

        "step_up": 0.01,               # Very slow advance (1% per step)
        "step_down": 0.03,             # Fast retreat when struggling
        "cooldown_updates": 5,

        "update_every_episodes": 4096,

        "mix_prob_current": 0.75,       # 75% on current frontier
        "mix_level_low": 0.00,
        "mix_level_high": 0.50,

        # DR easy/hard are same since DR is fixed (not curriculum-coupled)
        "friction_easy": [0.4, 1.0],
        "kp_easy": [50.0, 70.0],
        "kd_easy": [1.0, 4.0],
        "kp_factor_easy": [0.9, 1.1],
        "kd_factor_easy": [0.9, 1.1],
        "mass_shift_easy": [-0.5, 1.5],
        "com_shift_easy": [-0.02, 0.02],
        "leg_mass_shift_easy": [-0.3, 0.3],
        "gravity_offset_easy": [-0.5, 0.5],
        "motor_strength_easy": [0.92, 1.08],
        "push_start": 999.0,           # Effectively never (pushes disabled)
        "push_interval_easy_s": 999.0,
        "delay_easy_max_steps": 1,      # Match v4 (0-1 random delay)
        "global_dr_update_interval": 200,
    }

    # ================================================================
    # ENVIRONMENT CONFIG
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
        "episode_length_s": 30.0,          # 30s for multi-flight stair traversal
        "resampling_time_s": 10.0,         # Resample every 10s (less frequent)
        "action_scale": 0.25,
        "clip_actions": 100.0,

        # --- Terrain ---
        "terrain": terrain_cfg,

        # --- Curriculum (terrain-only, DR is fixed) ---
        "curriculum": curriculum_cfg,

        # --- DR: fixed moderate (NO dr_schedule = no two-phase) ---
        "friction_range": friction_range,
        "kp_factor_range": kp_factor_range,
        "kd_factor_range": kd_factor_range,
        "kp_range": kp_range,
        "kd_range": kd_range,
        "obs_noise": obs_noise,
        "obs_noise_level": 1.0,            # Same as v4 (full noise, already learned)
        "action_noise_std": 0.1,            # Same as v4 (full motor noise)
        # NO push_force_range — pushes disabled for stair training
        # NO push_interval_s — pushes disabled
        # NO push_duration_s — pushes disabled
        "init_pos_z_range": [0.38, 0.45],
        "init_euler_range": [-3.0, 3.0],    # Tighter init (less wild spawns)
        "mass_shift_range": mass_shift_range,
        "com_shift_range": com_shift_range,
        "leg_mass_shift_range": leg_mass_shift_range,
        "gravity_offset_range": gravity_offset_range,
        "motor_strength_range": motor_strength_range,
        "min_delay_steps": 0,               # Same as v4 (0-1 step random delay)
        "max_delay_steps": 1,               # Same as v4
    }

    # ================================================================
    # OBSERVATION CONFIG
    # ================================================================
    num_obs = 3 + 3 + 3 + 12 + 12 + num_actions_total

    num_privileged_extra = (3 + 1 + 12 + 12 + 12 + 1 + 3 + 4 + 3 + 3 + 1
                           + 1                     # terrain_row
                           + num_height_scan)       # height scan (77)
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
    # REWARDS — STAIR-UP SPECIALIST
    #
    # Design philosophy:
    #   1. forward_progress is THE primary reward (not velocity tracking)
    #   2. foot_clearance_barrier is the anti-face-collision reward
    #   3. feet_air_time encourages arc-shaped foot trajectories
    #   4. pitch_excess allows natural climbing tilt, only penalises extremes
    #   5. base_height keeps body high (terrain-relative)
    #   6. Minimal regularisation — don't fight stair-climbing postures
    #
    # CHANGED from v4:
    #   - tracking_lin_vel: 1.5 → 0.0 (dropped, replaced by forward_progress)
    #   - tracking_ang_vel: 0.8 → 0.0 (no turning on corridor terrain)
    #   - forward_progress: 0.4 → 1.0 (primary driver)
    #   - foot_clearance_barrier: NEW, -1.5 (one-sided barrier, the key reward)
    #   - foot_clearance: -0.5 → 0.0 (replaced by barrier version)
    #   - pitch_excess: NEW, -3.0 (deadzone allows climbing tilt)
    #   - orientation_roll_only: -5.0 → -5.0 (unchanged, roll always bad)
    #   - feet_air_time: 0.2 → 0.5 (STRONG, encourages arc foot trajectory)
    #   - base_height: -0.1 → -0.5 (STRONGER, prevents face collision)
    #   - lin_vel_z: -1.0 → -0.5 (more relaxed + deadzone handles stairs)
    #   - similar_to_default: -0.05 → 0.0 (dropped, stair posture differs)
    #   - stand_still/stand_still_vel/feet_stance: kept for zero-command
    # ================================================================
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.30,         # Target body height above local terrain
        "feet_height_target": 0.12,          # 12cm foot clearance target
        "feet_air_time_target": 0.15,        # 0.15s target (deliberate stepping for stairs)
        "lin_vel_z_deadzone": 0.20,          # Allow 0.20 m/s z-vel without penalty
        "pitch_allowed_min_deg": 0.0,        # Allow 0° to 40° forward pitch
        "pitch_allowed_max_deg": 40.0,       # Penalise backward (<0°) and extreme forward (>40°)

        "reward_scales": {
            # --- PRIMARY: forward displacement (stair climbing incentive) ---
            "forward_progress": 1.0,

            # --- TRACKING: kept at 0 for now, can be enabled later ---
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.0,

            # --- GAIT: encourage high-arc deliberate stepping ---
            "feet_air_time": 0.5,            # Strong: forces deliberate swing phase
            "foot_clearance_barrier": -1.5,  # Strong: one-sided barrier (feet must clear!)
            "foot_slip": -0.2,               # Stable foot contact on treads

            # --- STABILITY ---
            "orientation_roll_only": -5.0,   # Roll is always bad
            "pitch_excess": -3.0,            # Penalise pitch outside [0°, 40°]
            "base_height": -0.5,             # Keep body high above terrain
            "lin_vel_z": -0.5,               # Gentle + deadzone for stair climbing
            "ang_vel_xy": -0.05,             # Smooth body rotation

            # --- SMOOTHNESS ---
            "action_rate": -0.01,
            "dof_acc": -2.5e-7,
            "dof_vel": -5e-4,
            "joint_tracking": -0.05,         # Mild: follow commanded positions

            # --- STANDING (zero-command behaviour) ---
            "stand_still": -0.3,
            "stand_still_vel": -1.5,
            "feet_stance": -0.2,
        },
    }

    # ================================================================
    # COMMANDS — forward only, no turning
    #
    # CHANGED from v4:
    #   - lin_vel_x: [0.3, 0.8] → [0.3, 0.6] (slower, more careful climbing)
    #   - lin_vel_y: [0.0, 0.0] (unchanged, corridor)
    #   - ang_vel: [0.0, 0.0] (unchanged, corridor)
    #   - rel_standing_envs: 0.05 → 0.05 (keep 5% standing)
    # ================================================================
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.6],     # Moderate forward speed
        "lin_vel_y_range": [0.0, 0.0],      # No lateral
        "ang_vel_range": [0.0, 0.0],        # No turning
        "cmd_curriculum": False,
        "compound_commands": True,
        "rel_standing_envs": 0.05,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairs-v5-up")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, resume_path=args.resume)

    if args.resume is None and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ================================================================
    # Config summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  STAIR-UP SPECIALIST v5 — up-only terrain + barrier clearance")
    print("=" * 70)

    tc = env_cfg.get("terrain", {})
    if tc.get("enabled"):
        print(f"  Terrain mode       : {'UP-ONLY' if tc.get('up_only') else 'UP-DOWN'}")
        print(f"  Difficulty rows    : {tc['num_difficulty_rows']}")
        print(f"  Step height range  : {tc['step_height_min']*100:.0f}cm → {tc['step_height_max']*100:.0f}cm")
        print(f"  Steps per flight   : {tc['num_steps']}")
        print(f"  Tread depth        : {tc['step_depth_m']*100:.0f}cm")
        print(f"  Flights per row    : {tc['num_flights']}")
        max_elev = tc['num_flights'] * tc['num_steps'] * tc['step_height_max']
        print(f"  Max total elevation: {max_elev:.1f}m (hardest row)")

    print(f"\n  DR strategy        : FIXED MODERATE (no curriculum coupling)")
    print(f"  Push forces        : DISABLED (add in hardening phase)")
    print(f"  Obs noise level    : {env_cfg.get('obs_noise_level', 0)} (same as v4)")
    print(f"  Action noise std   : {env_cfg.get('action_noise_std', 0)} (same as v4)")
    print(f"  Action delay       : {env_cfg['min_delay_steps']}-{env_cfg['max_delay_steps']} steps (same as v4)")
    print(f"  Friction range     : {env_cfg['friction_range']}")

    print(f"\n  Primary reward     : forward_progress = {reward_cfg['reward_scales']['forward_progress']}")
    print(f"  Key anti-collision : foot_clearance_barrier = {reward_cfg['reward_scales']['foot_clearance_barrier']}")
    print(f"  Pitch allowed      : [{reward_cfg['pitch_allowed_min_deg']}°, {reward_cfg['pitch_allowed_max_deg']}°]")
    print(f"  Airtime target     : {reward_cfg['feet_air_time_target']}s (deliberate stepping)")
    print(f"  Foot clearance     : {reward_cfg['feet_height_target']}m target (terrain-relative)")
    print(f"  Tracking rewards   : lin_vel={reward_cfg['reward_scales']['tracking_lin_vel']}, "
          f"ang_vel={reward_cfg['reward_scales']['tracking_ang_vel']} (kept at 0)")

    print(f"\n  KEPT at 0          : tracking_lin_vel, tracking_ang_vel (can enable later)")
    print(f"  DROPPED            : similar_to_default (stair posture differs)")
    print(f"  REPLACED           : foot_clearance → foot_clearance_barrier (one-sided)")
    print(f"  NEW                : pitch_excess [0°,40°] allowed, forward_progress (primary)")

    cc = env_cfg.get("curriculum", {})
    if cc.get("enabled"):
        print(f"\n  Curriculum         : terrain-only (DR fixed)")
        print(f"  level_init         : {cc.get('level_init')}")
        print(f"  step_up / step_down: {cc.get('step_up')} / {cc.get('step_down')}")

    if args.resume:
        print(f"\n  Resuming from      : {args.resume}")

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