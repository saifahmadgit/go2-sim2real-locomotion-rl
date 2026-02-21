import argparse
import math
import os
import pickle
from collections import deque
from importlib import metadata

import torch

# -------- rsl-rl version guard --------
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e

import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

# =====================================================================

FIXED_COMMAND = torch.tensor([[0.7, 0.0, 0.0]])

# -------------------- 1. PD GAINS -----------------------------------
KP = 60.0
KD = 2.0

# -------------------- 2. GROUND / ROBOT FRICTION --------------------
GROUND_FRICTION = 0.7
ROBOT_FRICTION = 0.7

# -------------------- 3. EXTERNAL PUSHES ----------------------------
PUSH_ENABLE = False
PUSH_FORCE_RANGE = (150.0, 150.0)
PUSH_DURATION_S = 0.15
PUSH_INTERVAL_S = 2.0
PUSH_Z_SCALE = 0.0

# -------------------- 4. OBSERVATION NOISE --------------------------
# Additive Gaussian noise on observation vector before the policy sees it.
# Models IMU drift, encoder noise, joint-velocity estimation error, etc.
# OBS_NOISE_LEVEL scales ALL std values together (0 = off, 1 = nominal).
OBS_NOISE_ENABLE = True
OBS_NOISE_LEVEL = 0.1

# Per-component std at OBS_NOISE_LEVEL = 1.0
# Layout: [ang_vel(3), gravity(3), commands(3),
#          dof_pos(12), dof_vel(12), last_actions(12)]  = 45 dim
OBS_NOISE_STD = {
    "ang_vel": 0.2,  # rad/s
    "gravity": 0.05,
    "commands": 0.0,  # no noise on commands
    "dof_pos": 0.02,  # rad
    "dof_vel": 1.0,  # rad/s
    "last_actions": 0.0,  # no noise on prev actions
}

# -------------------- 5. ACTION NOISE / MOTOR NOISE -----------------
# Additive Gaussian on actions AFTER policy, BEFORE PD controller.
# Simulates actuator backlash, electrical noise, quantisation.
ACTION_NOISE_ENABLE = True
ACTION_NOISE_STD = 0.14  # rad

# -------------------- 6. ACTION LATENCY / DELAY --------------------
# Buffers actions by N steps to simulate communication delay.
# Real Go2 has ~10-20ms latency; at 50Hz that is ~1 step.
ACTION_DELAY_ENABLE = False
ACTION_DELAY_STEPS = 1

# -------------------- 7. PAYLOAD / ADDED MASS ----------------------
# Constant downward force on base link: F = mass * 9.81
# Simulates backpack, sensor rig, or mass estimation error.
PAYLOAD_ENABLE = False
PAYLOAD_MASS = 0.0  # kg

# -------------------- 8. GRAVITY PERTURBATION ----------------------
# Constant lateral force on base = approx_mass * accel.
# Simulates slope or IMU calibration error.
GRAVITY_PERTURB_ENABLE = False
GRAVITY_PERTURB_X = 0.0  # m/s^2  (forward/back)
GRAVITY_PERTURB_Y = 0.0  # m/s^2  (left/right)
GO2_APPROX_MASS = 12.0  # kg (used to convert accel to force)

# -------------------- 9. MOTOR STRENGTH SCALING --------------------
# Multiplier on Kp -> simulates weak/strong motors.
# e.g. 0.8 = 80% nominal strength (voltage sag, wear).
MOTOR_STRENGTH_ENABLE = False
MOTOR_STRENGTH_SCALE = 0.8

# -------------------- 10. INITIAL POSE PERTURBATION ----------------
# Override the starting base position, orientation, and/or joint angles.
# Useful to test: can the policy recover from a non-nominal start?
#
# Base pos/orientation and joint angles are INDEPENDENT (floating base):
#   - base pos/orient = where the torso is in world space
#   - joint angles    = leg configuration relative to the torso
#
# Set *_ENABLE = False to use the defaults from training config.

# Base position [x, y, z] in metres.  Default is [0, 0, 0.42].
# Try z=0.6 (drop from height) or z=0.25 (start crouched/collapsed).
INIT_POS_ENABLE = False
INIT_POS = [0.0, 0.0, 0.45]

# Base orientation in degrees.  Default is (0, 0).
# Examples:  (15, 0)  = 15° roll (right side down)
#            (0, 15)  = 15° pitch (nose down)
#            (10, -5) = combo tilt
INIT_ORIENT_ENABLE = False
INIT_ROLL_DEG = 25.0  # degrees, positive = right side down
INIT_PITCH_DEG = 0.0  # degrees, positive = nose down

# Joint angles [rad].  Only the joints you list here get overridden;
# the rest keep their training defaults.
# Set to {} to skip, or override specific joints, e.g.:
#   {"FL_thigh_joint": 1.2, "FR_thigh_joint": 1.2}   (legs more bent)
#   or override all 12 to test a completely different stance.
INIT_JOINTS_ENABLE = False
INIT_JOINTS_OVERRIDE = {
    # "FL_hip_joint":   0.0,
    # "FR_hip_joint":   0.0,
    # "RL_hip_joint":   0.0,
    # "RR_hip_joint":   0.0,
    # "FL_thigh_joint": 0.8,
    # "FR_thigh_joint": 0.8,
    # "RL_thigh_joint": 1.0,
    # "RR_thigh_joint": 1.0,
    # "FL_calf_joint": -1.5,
    # "FR_calf_joint": -1.5,
    # "RL_calf_joint": -1.5,
    # "RR_calf_joint": -1.5,
}

# -------------------- DEBUG ----------------------------------------
DEBUG_PRINT_INTERVAL = 200  # print debug info every N steps (0=off)

# =====================================================================


def rand_float(lower, upper, shape):
    return (upper - lower) * torch.rand(shape, device=gs.device) + lower


def euler_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg=0.0):
    """Convert roll/pitch/yaw in degrees to quaternion [w, x, y, z]."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, yy, z]


def build_obs_noise_vec(obs_dim, noise_cfg, level):
    """Build 1-D std vector matching the obs layout."""
    components = [
        ("ang_vel", 3),
        ("gravity", 3),
        ("commands", 3),
        ("dof_pos", 12),
        ("dof_vel", 12),
        ("last_actions", 12),
    ]

    parts = []
    for name, size in components:
        std = noise_cfg.get(name, 0.0)
        parts.append(torch.full((size,), std * level, device=gs.device))

    noise_vec = torch.cat(parts)

    if obs_dim > len(noise_vec):
        noise_vec = torch.cat(
            [
                noise_vec,
                torch.zeros(obs_dim - len(noise_vec), device=gs.device),
            ]
        )
    elif obs_dim < len(noise_vec):
        noise_vec = noise_vec[:obs_dim]

    return noise_vec.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # ---- PD gains ----
    effective_kp = KP * (MOTOR_STRENGTH_SCALE if MOTOR_STRENGTH_ENABLE else 1.0)
    effective_kd = KD
    env_cfg["kp"] = effective_kp
    env_cfg["kd"] = effective_kd

    # ---- Disable termination ----
    env_cfg["termination_if_roll_greater_than"] = 1e9
    env_cfg["termination_if_pitch_greater_than"] = 1e9
    env_cfg["termination_if_z_vel_greater_than"] = 1e9
    env_cfg["termination_if_y_vel_greater_than"] = 1e9

    reward_cfg["reward_scales"] = {}

    # ---- Initial pose overrides ----
    if INIT_POS_ENABLE:
        env_cfg["base_init_pos"] = list(INIT_POS)
        print(f"  Init pos override: {INIT_POS}")

    if INIT_ORIENT_ENABLE:
        quat = euler_deg_to_quat_wxyz(INIT_ROLL_DEG, INIT_PITCH_DEG)
        env_cfg["base_init_quat"] = quat
        print(
            f"  Init orient override: roll={INIT_ROLL_DEG}° pitch={INIT_PITCH_DEG}°  ->  quat={[round(q, 4) for q in quat]}"
        )

    if INIT_JOINTS_ENABLE and INIT_JOINTS_OVERRIDE:
        for joint_name, angle in INIT_JOINTS_OVERRIDE.items():
            if joint_name in env_cfg["default_joint_angles"]:
                env_cfg["default_joint_angles"][joint_name] = angle
            else:
                print(f"  WARNING: joint '{joint_name}' not found in config, skipping")
        print(f"  Init joints override: {INIT_JOINTS_OVERRIDE}")

    # -------- Create Environment --------
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # -------- Set Friction --------
    ground_entity = None
    for ent in env.scene.entities:
        try:
            if hasattr(ent, "morph") and hasattr(ent.morph, "file"):
                if "plane.urdf" in ent.morph.file:
                    ground_entity = ent
                    break
        except Exception:
            pass

    if ground_entity is None:
        raise RuntimeError("Ground entity not found")

    ground_entity.set_friction(GROUND_FRICTION)
    env.robot.set_friction(ROBOT_FRICTION)

    print("\nFriction set:")
    print("  Ground:", GROUND_FRICTION)
    print("  Robot :", ROBOT_FRICTION)

    # -------- Load Policy --------
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # -------- Push Setup --------
    base_link_idx = env.robot.links[1].idx

    push_steps = int(PUSH_DURATION_S / env.dt)
    push_interval_steps = int(PUSH_INTERVAL_S / env.dt)

    push_timer = 0
    cached_push = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)

    # -------- Precompute constant forces (payload + gravity perturb) --------
    constant_force = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)
    has_constant_force = False

    if PAYLOAD_ENABLE and PAYLOAD_MASS > 0:
        constant_force[0, 2] -= PAYLOAD_MASS * 9.81
        has_constant_force = True
        print(f"\n  Payload force: {constant_force[0].tolist()} N")

    if GRAVITY_PERTURB_ENABLE:
        constant_force[0, 0] += GRAVITY_PERTURB_X * GO2_APPROX_MASS
        constant_force[0, 1] += GRAVITY_PERTURB_Y * GO2_APPROX_MASS
        has_constant_force = True
        print(f"  Gravity perturb force: {constant_force[0].tolist()} N")

    # -------- Obs noise vector (built lazily) --------
    obs_noise_vec = None

    # -------- Action delay buffer --------
    action_buffer = deque(maxlen=ACTION_DELAY_STEPS + 1)

    # -------- Reset --------
    obs, _ = env.reset()
    target_cmd = FIXED_COMMAND.to(gs.device)

    # Seed action buffer with zeros
    zero_action = torch.zeros((1, env.num_actions), device=gs.device)
    for _ in range(ACTION_DELAY_STEPS + 1):
        action_buffer.append(zero_action.clone())

    step = 0

    # -------- Config Summary --------
    print("\n" + "=" * 60)
    print("  DOMAIN RANDOMISATION EVAL CONFIG")
    print("=" * 60)
    print(f"  Command         : {FIXED_COMMAND.tolist()}")
    print(f"  Kp / Kd         : {KP} / {KD}")
    if MOTOR_STRENGTH_ENABLE:
        print(
            f"  Motor strength  : {MOTOR_STRENGTH_SCALE:.2f}x  ->  Kp = {effective_kp:.1f}"
        )
    print(f"  Friction        : ground={GROUND_FRICTION}, robot={ROBOT_FRICTION}")
    print(
        f"  Pushes          : {'ON' if PUSH_ENABLE else 'OFF'}"
        f"  range={PUSH_FORCE_RANGE}, interval={PUSH_INTERVAL_S}s"
    )
    print(
        f"  Obs noise       : {'ON' if OBS_NOISE_ENABLE else 'OFF'}"
        f"  level={OBS_NOISE_LEVEL}"
    )
    print(
        f"  Action noise    : {'ON' if ACTION_NOISE_ENABLE else 'OFF'}"
        f"  std={ACTION_NOISE_STD}"
    )
    print(
        f"  Action delay    : {'ON' if ACTION_DELAY_ENABLE else 'OFF'}"
        f"  steps={ACTION_DELAY_STEPS}"
    )
    print(
        f"  Payload         : {'ON' if PAYLOAD_ENABLE else 'OFF'}"
        f"  mass={PAYLOAD_MASS} kg"
        f"  force_z={-PAYLOAD_MASS * 9.81 if PAYLOAD_ENABLE else 0:.1f} N"
    )
    print(
        f"  Gravity perturb : {'ON' if GRAVITY_PERTURB_ENABLE else 'OFF'}"
        f"  ({GRAVITY_PERTURB_X}, {GRAVITY_PERTURB_Y}) m/s^2"
    )
    if INIT_POS_ENABLE:
        print(f"  Init base pos   : {INIT_POS}")
    if INIT_ORIENT_ENABLE:
        print(f"  Init orientation: roll={INIT_ROLL_DEG}° pitch={INIT_PITCH_DEG}°")
    if INIT_JOINTS_ENABLE and INIT_JOINTS_OVERRIDE:
        print(f"  Init joints     : {INIT_JOINTS_OVERRIDE}")
    print("=" * 60 + "\n")
    print("Running evaluation...\n")

    with torch.no_grad():
        while True:
            step += 1
            env.commands[:] = target_cmd

            # ==========================================================
            #  FORCES - combine everything into ONE tensor, ONE call
            #  (Genesis overwrites on multiple apply calls per step)
            # ==========================================================
            total_force = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)
            apply_force = False

            # --- Push ---
            if PUSH_ENABLE and step % push_interval_steps == 0 and push_timer == 0:
                mag = rand_float(*PUSH_FORCE_RANGE, (1, 1)).to(gs.tc_float)
                theta = rand_float(0.0, 2.0 * math.pi, (1, 1))

                cached_push[0, 0] = torch.cos(theta) * mag
                cached_push[0, 1] = torch.sin(theta) * mag
                cached_push[0, 2] = PUSH_Z_SCALE * mag
                push_timer = push_steps

                print(f"[Push] force={cached_push[0].tolist()}")

            if PUSH_ENABLE and push_timer > 0:
                total_force += cached_push
                apply_force = True
                push_timer -= 1

            # --- Constant forces (payload + gravity perturb) ---
            if has_constant_force:
                total_force += constant_force
                apply_force = True

            # --- Single apply call ---
            if apply_force:
                env.scene.sim.rigid_solver.apply_links_external_force(
                    force=total_force,
                    links_idx=[base_link_idx],
                    envs_idx=[0],
                )

            # ==========================================================
            #  OBSERVATION NOISE
            # ==========================================================
            policy_obs = obs

            if OBS_NOISE_ENABLE:
                if obs_noise_vec is None:
                    obs_noise_vec = build_obs_noise_vec(
                        obs.shape[-1], OBS_NOISE_STD, OBS_NOISE_LEVEL
                    )
                    print(
                        f"[ObsNoise] obs_dim={obs.shape[-1]}, "
                        f"noise_std range=[{obs_noise_vec.min().item():.4f}, "
                        f"{obs_noise_vec.max().item():.4f}]"
                    )

                noise = torch.randn_like(obs) * obs_noise_vec
                policy_obs = obs + noise

            # ==========================================================
            #  POLICY INFERENCE
            # ==========================================================
            raw_actions = policy(policy_obs)

            # ==========================================================
            #  ACTION NOISE
            # ==========================================================
            if ACTION_NOISE_ENABLE:
                action_noise = torch.randn_like(raw_actions) * ACTION_NOISE_STD
                raw_actions = raw_actions + action_noise

            # ==========================================================
            #  ACTION DELAY
            # ==========================================================
            if ACTION_DELAY_ENABLE:
                action_buffer.append(raw_actions.clone())
                actions_to_apply = action_buffer[0]
            else:
                actions_to_apply = raw_actions

            # ==========================================================
            #  DEBUG PRINTS
            # ==========================================================
            if DEBUG_PRINT_INTERVAL > 0 and step % DEBUG_PRINT_INTERVAL == 0:
                parts = []

                if apply_force:
                    parts.append(f"force={total_force[0].tolist()}")

                if OBS_NOISE_ENABLE:
                    # Compare: what would policy output with CLEAN obs?
                    clean_act = policy(obs)
                    act_diff = (raw_actions - clean_act).abs().mean().item()
                    obs_diff = (policy_obs - obs).abs().mean().item()
                    parts.append(
                        f"obs_noise_mean={obs_diff:.4f}, "
                        f"action_diff_from_clean={act_diff:.4f}"
                    )

                if ACTION_NOISE_ENABLE:
                    parts.append(f"act_noise_std={ACTION_NOISE_STD}")

                if ACTION_DELAY_ENABLE:
                    delay_diff = (actions_to_apply - raw_actions).abs().mean().item()
                    parts.append(f"delay_act_diff={delay_diff:.4f}")

                if parts:
                    print(f"[step {step}] " + " | ".join(parts))

            # ==========================================================
            #  STEP
            # ==========================================================
            obs, _, _, _ = env.step(actions_to_apply)


if __name__ == "__main__":
    main()
