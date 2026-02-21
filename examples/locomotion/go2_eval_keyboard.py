import argparse
import math
import os
import pickle
import threading
from collections import deque
from importlib import metadata
from typing import Optional

import torch
from pynput import keyboard

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
from go2_env_v5 import Go2Env
from rsl_rl.runners import OnPolicyRunner

# =====================================================================
#  INFERENCE CONFIG — all eval-time knobs at the top
# =====================================================================

# ============================================================
# PREDEFINED VELOCITIES (edit these to change speed)
# ============================================================
FORWARD_VX = 1.0  # W key: forward speed
BACKWARD_VX = 0.5  # S key: backward speed
LEFT_VY = 0.7  # A key: negative vy (left)
RIGHT_VY = 0.7  # D key: positive vy (right)
YAW_CW_WZ = 0.7  # Q key: clockwise yaw
YAW_CCW_WZ = 0.7  # R key: counter-clockwise yaw

# -------------------- 1. PD GAINS / STIFFNESS -----------------------
KP = 60.0
KD = 2.0

# -------------------- 2. GROUND / ROBOT FRICTION --------------------
GROUND_FRICTION = 0.7
ROBOT_FRICTION = 0.7

# -------------------- 3. EXTERNAL PUSHES ----------------------------
PUSH_ENABLE = False
PUSH_FORCE_RANGE = (200.0, 200.0)
PUSH_DURATION_S = [0.05, 0.15]
PUSH_INTERVAL_S = 2.0
PUSH_Z_SCALE = 0.0

# -------------------- 4. OBSERVATION NOISE --------------------------
OBS_NOISE_ENABLE = True
OBS_NOISE_LEVEL = 0.1

OBS_NOISE_STD = {
    "ang_vel": 0.2,
    "gravity": 0.05,
    "commands": 0.0,
    "dof_pos": 0.01,
    "dof_vel": 1.5,
    "actions": 0.0,
}

# -------------------- 5. ACTION NOISE / MOTOR NOISE -----------------
ACTION_NOISE_ENABLE = True
ACTION_NOISE_STD = 0.1

# -------------------- 6. ACTION LATENCY / DELAY --------------------
ACTION_DELAY_ENABLE = True
ACTION_DELAY_STEPS = 1

# -------------------- 7. PAYLOAD / ADDED MASS ----------------------
PAYLOAD_ENABLE = True
PAYLOAD_MASS = 0.0

# -------------------- 8. GRAVITY PERTURBATION ----------------------
GRAVITY_PERTURB_ENABLE = False
GRAVITY_PERTURB_X = 0.0
GRAVITY_PERTURB_Y = 0.0
GO2_APPROX_MASS = 15.0

# -------------------- 9. MOTOR STRENGTH SCALING --------------------
MOTOR_STRENGTH_ENABLE = False
MOTOR_STRENGTH_SCALE = 0.8

# -------------------- 10. GRAVITY OBSERVATION OFFSET ----------------
GRAVITY_OBS_OFFSET_ENABLE = False
GRAVITY_OBS_OFFSET = [0.0, 0.0, 0.0]

# -------------------- 11. INITIAL POSE PERTURBATION ----------------
INIT_POS_ENABLE = False
INIT_POS = [0.0, 0.0, 0.42]

INIT_ORIENT_ENABLE = False
INIT_ROLL_DEG = 25.0
INIT_PITCH_DEG = 0.0

INIT_JOINTS_ENABLE = False
INIT_JOINTS_OVERRIDE = {}

# -------------------- 12. DYNAMIC PAYLOAD ---------------------------
DYNAMIC_PAYLOAD_ENABLE = False
DYNAMIC_PAYLOAD_STEP = 50
DYNAMIC_PAYLOAD_MASS = 3.0

# -------------------- DEBUG ----------------------------------------
DEBUG_PRINT_INTERVAL = 200

# =====================================================================
# KEYBOARD INPUT STATE (shared)
# =====================================================================

command_state = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
}

_state_lock = threading.Lock()


def make_command_tensor():
    # Read under lock (avoid partial reads while listener updates)
    with _state_lock:
        vx = command_state["vx"]
        vy = command_state["vy"]
        wz = command_state["wz"]
    return torch.tensor([[vx, vy, wz]])


def handle_key(key):
    """Process a keypress and update command_state. Returns False to quit."""
    if key is None:
        return True

    if isinstance(key, str) and len(key) == 1:
        key = key.lower()

    if key in ("x", "CTRL_C", "ESC"):
        return False

    with _state_lock:
        if key == "p":
            command_state["vx"] = FORWARD_VX
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key == "m":
            command_state["vx"] = -BACKWARD_VX
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key == "k":
            command_state["vx"] = 0.0
            command_state["vy"] = RIGHT_VY
            command_state["wz"] = 0.0
        elif key == "j":
            command_state["vx"] = 0.0
            command_state["vy"] = -LEFT_VY
            command_state["wz"] = 0.0
        elif key == "u":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = -YAW_CW_WZ
        elif key == "p":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = YAW_CCW_WZ
        elif key == " ":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0

    return True


def print_controls():
    print("\n============ KEYBOARD CONTROLS ============")
    print(f"  W           : forward   (vx={FORWARD_VX:.2f})")
    print(f"  S           : backward  (vx={-BACKWARD_VX:.2f})")
    print(f"  D           : right     (vy={RIGHT_VY:.2f})")
    print(f"  A           : left      (vy={-LEFT_VY:.2f})")
    print(f"  Q           : yaw CW    (wz={-YAW_CW_WZ:.2f})")
    print(f"  R           : yaw CCW   (wz={YAW_CCW_WZ:.2f})")
    print("  SPACE       : zero velocity (stop in place)")
    print("  X / ESC     : quit")
    print("  CTRL+C      : quit")
    print("============================================\n")


# =====================================================================
# GLOBAL KEYBOARD LISTENER (works even when viewer is focused)
# =====================================================================

_quit_event = threading.Event()
_pressed_mods = set()


def _normalize_key(k) -> Optional[str]:
    # Special keys
    if k == keyboard.Key.space:
        return " "
    if k == keyboard.Key.esc:
        return "ESC"
    # Normal character keys
    try:
        ch = k.char
        if ch is None:
            return None
        return ch.lower()
    except AttributeError:
        return None


def _on_press(k):
    # Track Ctrl for Ctrl+C detection
    if k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        _pressed_mods.add("ctrl")
        return

    # Ctrl+C -> quit
    try:
        ch = k.char
        if ch is not None and ch.lower() == "c" and ("ctrl" in _pressed_mods):
            _quit_event.set()
            return False
    except AttributeError:
        pass

    key_str = _normalize_key(k)
    if key_str is None:
        return

    keep_running = handle_key(key_str)
    if not keep_running:
        _quit_event.set()
        return False  # stop listener


def _on_release(k):
    if k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        _pressed_mods.discard("ctrl")


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()
    return listener


# =====================================================================
# Helper functions
# =====================================================================


def rand_float(lower, upper, shape):
    return (upper - lower) * torch.rand(shape, device=gs.device) + lower


def rand_int(lower, upper, shape):
    return torch.randint(lower, upper + 1, size=shape, device=gs.device)


def euler_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg=0.0):
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
    fixed_size = 3 + 3 + 3 + 12 + 12
    action_size = obs_dim - fixed_size

    components = [
        ("ang_vel", 3),
        ("gravity", 3),
        ("commands", 3),
        ("dof_pos", 12),
        ("dof_vel", 12),
        ("actions", action_size),
    ]

    parts = []
    for name, size in components:
        std = noise_cfg.get(name, 0.0)
        parts.append(torch.full((size,), std * level, device=gs.device))

    noise_vec = torch.cat(parts)[:obs_dim]
    return noise_vec.unsqueeze(0)


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-v5")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # ================================================================
    # Override env config for evaluation
    # ================================================================
    pls_enabled = env_cfg.get("pls_enable", False)

    if not pls_enabled:
        effective_kp = KP * (MOTOR_STRENGTH_SCALE if MOTOR_STRENGTH_ENABLE else 1.0)
        env_cfg["kp"] = effective_kp
        env_cfg["kd"] = KD

    # Disable training-time DR and curriculum
    for key in [
        "friction_range",
        "kp_range",
        "kd_range",
        "kp_factor_range",
        "kd_factor_range",
        "obs_noise",
        "obs_noise_level",
        "action_noise_std",
        "push_interval_s",
        "push_force_range",
        "push_duration_s",
        "mass_shift_range",
        "com_shift_range",
        "leg_mass_shift_range",
        "dynamic_payload_range",
        "dynamic_payload_interval_s",
        "dynamic_payload_prob",
        "gravity_offset_range",
        "motor_strength_range",
        "init_pos_z_range",
        "init_euler_range",
        "min_delay_steps",
        "max_delay_steps",
    ]:
        env_cfg.pop(key, None)

    if "curriculum" in env_cfg:
        env_cfg["curriculum"]["enabled"] = False

    # Disable termination (let it run forever for eval)
    env_cfg["termination_if_roll_greater_than"] = 1e9
    env_cfg["termination_if_pitch_greater_than"] = 1e9
    env_cfg["termination_if_z_vel_greater_than"] = 1e9
    env_cfg["termination_if_y_vel_greater_than"] = 1e9

    reward_cfg["reward_scales"] = {}

    # Initial pose overrides
    if INIT_POS_ENABLE:
        env_cfg["base_init_pos"] = list(INIT_POS)
    if INIT_ORIENT_ENABLE:
        quat = euler_deg_to_quat_wxyz(INIT_ROLL_DEG, INIT_PITCH_DEG)
        env_cfg["base_init_quat"] = quat
    if INIT_JOINTS_ENABLE and INIT_JOINTS_OVERRIDE:
        for joint_name, angle in INIT_JOINTS_OVERRIDE.items():
            if joint_name in env_cfg["default_joint_angles"]:
                env_cfg["default_joint_angles"][joint_name] = angle

    # ================================================================
    # Create Environment
    # ================================================================
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Set Friction
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

    if MOTOR_STRENGTH_ENABLE and pls_enabled:
        env._motor_strength[:] = MOTOR_STRENGTH_SCALE
        env._kp_factors[:] = 1.0
        env._kd_factors[:] = 1.0

    # ================================================================
    # Load Policy
    # ================================================================
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # ================================================================
    # Push Setup
    # ================================================================
    base_link_idx = env.robot.links[1].idx
    dt = env.dt

    push_interval_steps = int(PUSH_INTERVAL_S / dt)
    push_dur_steps_range = [
        max(1, int(PUSH_DURATION_S[0] / dt)),
        max(1, int(PUSH_DURATION_S[1] / dt)),
    ]

    push_remaining = 0
    cached_push = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)

    # Precompute constant forces
    constant_force = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)
    has_constant_force = False

    if PAYLOAD_ENABLE and PAYLOAD_MASS > 0:
        constant_force[0, 2] -= PAYLOAD_MASS * 9.81
        has_constant_force = True

    if GRAVITY_PERTURB_ENABLE:
        constant_force[0, 0] += GRAVITY_PERTURB_X * GO2_APPROX_MASS
        constant_force[0, 1] += GRAVITY_PERTURB_Y * GO2_APPROX_MASS
        has_constant_force = True

    gravity_obs_offset = None
    if GRAVITY_OBS_OFFSET_ENABLE:
        gravity_obs_offset = torch.tensor(
            GRAVITY_OBS_OFFSET, device=gs.device, dtype=gs.tc_float
        ).unsqueeze(0)

    obs_noise_vec = None
    action_buffer = deque(maxlen=ACTION_DELAY_STEPS + 1)

    # ================================================================
    # Reset
    # ================================================================
    obs, _ = env.reset()

    zero_action = torch.zeros((1, env.num_actions), device=gs.device)
    for _ in range(ACTION_DELAY_STEPS + 1):
        action_buffer.append(zero_action.clone())

    step = 0

    # ================================================================
    # Config Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  INFERENCE CONFIG (v5) — KEYBOARD CONTROL (GLOBAL LISTENER)")
    print("=" * 60)
    print(f"  PLS enabled     : {pls_enabled}")
    print(
        f"  Action space    : {env.num_actions} ({'12 pos + 4 stiffness' if pls_enabled else '12 pos'})"
    )
    print(f"  Actor obs dim   : {obs_cfg['num_obs']}")
    if not pls_enabled:
        print(f"  Kp / Kd         : {env_cfg['kp']} / {env_cfg['kd']}")
    else:
        print(
            f"  Kp              : policy-controlled [{env_cfg['pls_kp_range'][0]}, {env_cfg['pls_kp_range'][1]}]"
        )
    print(f"  Friction        : ground={GROUND_FRICTION}, robot={ROBOT_FRICTION}")
    print(
        f"  Obs noise       : {'ON' if OBS_NOISE_ENABLE else 'OFF'} level={OBS_NOISE_LEVEL}"
    )
    print(
        f"  Action noise    : {'ON' if ACTION_NOISE_ENABLE else 'OFF'} std={ACTION_NOISE_STD}"
    )
    print(
        f"  Action delay    : {'ON' if ACTION_DELAY_ENABLE else 'OFF'} steps={ACTION_DELAY_STEPS}"
    )
    print("=" * 60)

    print_controls()
    print("Running evaluation with keyboard control...\n")

    # ================================================================
    # Main Loop — with global keyboard control
    # ================================================================
    listener = start_keyboard_listener()

    try:
        with torch.no_grad():
            while True:
                step += 1

                if _quit_event.is_set():
                    break

                # --- Set command from keyboard state ---
                target_cmd = make_command_tensor().to(gs.device)
                env.commands[:] = target_cmd

                # ==========================================================
                #  DYNAMIC PAYLOAD
                # ==========================================================
                if DYNAMIC_PAYLOAD_ENABLE and step == DYNAMIC_PAYLOAD_STEP:
                    old_mass = PAYLOAD_MASS if PAYLOAD_ENABLE else 0.0
                    constant_force[0, 2] += old_mass * 9.81
                    constant_force[0, 2] -= DYNAMIC_PAYLOAD_MASS * 9.81
                    has_constant_force = True
                    try:
                        env.robot.set_mass_shift([DYNAMIC_PAYLOAD_MASS], [0])
                    except Exception:
                        pass

                # ==========================================================
                #  FORCES
                # ==========================================================
                total_force = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)
                apply_force = False

                if (
                    PUSH_ENABLE
                    and step % push_interval_steps == 0
                    and push_remaining == 0
                ):
                    mag = rand_float(*PUSH_FORCE_RANGE, (1, 1)).to(gs.tc_float)
                    theta = rand_float(0.0, 2.0 * math.pi, (1, 1))
                    cached_push[0, 0] = torch.cos(theta) * mag
                    cached_push[0, 1] = torch.sin(theta) * mag
                    cached_push[0, 2] = PUSH_Z_SCALE * mag
                    push_remaining = int(
                        rand_int(
                            push_dur_steps_range[0], push_dur_steps_range[1], (1,)
                        ).item()
                    )

                if PUSH_ENABLE and push_remaining > 0:
                    total_force += cached_push
                    apply_force = True
                    push_remaining -= 1

                if has_constant_force:
                    total_force += constant_force
                    apply_force = True

                if apply_force:
                    env.scene.sim.rigid_solver.apply_links_external_force(
                        force=total_force,
                        links_idx=[base_link_idx],
                        envs_idx=[0],
                    )

                # ==========================================================
                #  OBSERVATION NOISE + GRAVITY OFFSET
                # ==========================================================
                policy_obs = obs.clone()

                if gravity_obs_offset is not None:
                    policy_obs[:, 3:6] += gravity_obs_offset

                if OBS_NOISE_ENABLE:
                    if obs_noise_vec is None:
                        obs_noise_vec = build_obs_noise_vec(
                            obs.shape[-1], OBS_NOISE_STD, OBS_NOISE_LEVEL
                        )
                    noise = torch.randn_like(obs) * obs_noise_vec
                    policy_obs = policy_obs + noise

                # ==========================================================
                #  POLICY INFERENCE
                # ==========================================================
                raw_actions = policy(policy_obs)

                # ==========================================================
                #  ACTION NOISE
                # ==========================================================
                if ACTION_NOISE_ENABLE:
                    num_pos = env.num_pos_actions
                    pos_noise = (
                        torch.randn(1, num_pos, device=gs.device) * ACTION_NOISE_STD
                    )
                    action_noise = torch.zeros_like(raw_actions)
                    action_noise[:, :num_pos] = pos_noise
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
                #  DEBUG PRINTS (includes current command)
                # ==========================================================
                if DEBUG_PRINT_INTERVAL > 0 and step % DEBUG_PRINT_INTERVAL == 0:
                    parts = []

                    cmd = target_cmd[0]
                    parts.append(
                        f"cmd=[{cmd[0].item():+.2f},{cmd[1].item():+.2f},{cmd[2].item():+.2f}]"
                    )

                    pos = env.base_pos[0]
                    vel = env.base_lin_vel[0]
                    parts.append(
                        f"z={pos[2].item():.3f}m, "
                        f"vel=[{vel[0].item():.2f},{vel[1].item():.2f},{vel[2].item():.2f}]"
                    )

                    if pls_enabled:
                        stiffness_actions = actions_to_apply[0, env.num_pos_actions :]
                        kp_per_leg = (
                            env.pls_kp_default
                            + stiffness_actions * env.pls_kp_action_scale
                        )
                        kp_per_leg = torch.clamp(
                            kp_per_leg, env.pls_kp_range[0], env.pls_kp_range[1]
                        )
                        parts.append(
                            f"Kp=[{kp_per_leg[0].item():.1f},{kp_per_leg[1].item():.1f},"
                            f"{kp_per_leg[2].item():.1f},{kp_per_leg[3].item():.1f}]"
                        )

                    print(
                        f"\r[step {step}] " + " | ".join(parts) + "    ",
                        end="",
                        flush=True,
                    )

                # ==========================================================
                #  STEP
                # ==========================================================
                obs, _, _, _ = env.step(actions_to_apply)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            listener.stop()
        except Exception:
            pass

    print("\n\nEvaluation ended.")


if __name__ == "__main__":
    main()
