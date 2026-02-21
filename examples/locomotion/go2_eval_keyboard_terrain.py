import argparse
import math
import os
import pickle
import threading
from collections import deque
from importlib import metadata
from typing import Optional

import numpy as np
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
from rsl_rl.runners import OnPolicyRunner

from go2_env_terrain import Go2Env

# =====================================================================
#  INFERENCE CONFIG — all eval-time knobs at the top
# =====================================================================

# ============================================================
# PREDEFINED VELOCITIES (edit these to change speed)
# ============================================================
FORWARD_VX = 0.6  # P key: forward speed
BACKWARD_VX = 0.5  # M key: backward speed
LEFT_VY = 0.7  # J key: negative vy (left)
RIGHT_VY = 0.7  # K key: positive vy (right)
YAW_CW_WZ = 0.7  # U key: clockwise yaw
YAW_CCW_WZ = 0.7  # O key: counter-clockwise yaw

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

# -------------------- 13. TERRAIN ----------------------------------
# When TERRAIN_ENABLE = True, a custom numpy heightmap is generated
# with EXACT control over bump heights. Change MIN_HEIGHT / MAX_HEIGHT
# to control difficulty. Heights are in METERS — what you set is what
# you get. A new random terrain is generated every run.
#
# Set TERRAIN_ENABLE = False for a flat plane (same as old behaviour).

TERRAIN_ENABLE = True

TERRAIN_SIZE_X = 15.0  # terrain width in meters
TERRAIN_SIZE_Y = 15.0  # terrain depth in meters
TERRAIN_HORIZONTAL_SCALE = 0.05  # meters per heightmap pixel (smaller = finer detail)

# =====================================================================
# BUMP HEIGHT — this is the ONLY knob you need to change difficulty
# Heights are in METERS, directly. No vertical_scale confusion.
# =====================================================================
MIN_HEIGHT = -0.00  # meters (negative = dips)
MAX_HEIGHT = 0.00  # meters (positive = bumps)
# Examples:
#   ±0.00  = perfectly flat
#   ±0.02  = gentle (carpet, smooth pavement)
#   ±0.04  = training match (mats, grass, uneven pavement)
#   ±0.08  = hard (rough outdoor)
#   ±0.15  = very hard (rocky, likely beyond policy)

TERRAIN_SMOOTHING = 5  # uniform_filter size (1 = no smoothing, 3-5 = natural bumps)

# Starting tile (0-indexed). Press 1-6 to change row, [ ] to change col.
TERRAIN_SELECTED_ROW = 0
TERRAIN_SELECTED_COL = 0

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

terrain_state = {
    "row": TERRAIN_SELECTED_ROW,
    "col": TERRAIN_SELECTED_COL,
    "respawn_requested": False,
}

_state_lock = threading.Lock()


def make_command_tensor():
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
        # ---- Movement ----
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
        elif key == "o":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = YAW_CCW_WZ
        elif key == " ":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0

        # ---- Terrain row selection: 1-9 ----
        elif key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            row_idx = int(key) - 1
            if row_idx < len(_TERRAIN_PRESETS):
                preset = _TERRAIN_PRESETS[row_idx]
                terrain_state["row"] = row_idx
                terrain_state["respawn_requested"] = True
                print(f"\n  >> Selected preset {row_idx + 1}: {preset['label']}")

        # ---- R: respawn on current tile ----
        elif key == "r":
            terrain_state["respawn_requested"] = True

    return True


# =====================================================================
# TERRAIN PRESETS — press 1-5 to switch difficulty during eval
# =====================================================================
_TERRAIN_PRESETS = [
    {"label": "Flat", "min_h": 0.0, "max_h": 0.0},
    {"label": "Gentle (±2cm)", "min_h": -0.02, "max_h": 0.02},
    {"label": "Medium (±4cm)", "min_h": -0.04, "max_h": 0.04},
    {"label": "Hard (±8cm)", "min_h": -0.08, "max_h": 0.08},
    {"label": "Extreme (±15cm)", "min_h": -0.15, "max_h": 0.15},
]


def generate_heightmap(min_h, max_h, size_x, size_y, h_scale, smoothing=3):
    """Generate a numpy heightmap with exact height range in meters."""
    res_x = int(size_x / h_scale)
    res_y = int(size_y / h_scale)

    if min_h == 0.0 and max_h == 0.0:
        # Perfectly flat
        hf = np.zeros((res_y, res_x), dtype=np.float32)
    else:
        hf = np.random.uniform(min_h, max_h, (res_y, res_x)).astype(np.float32)
        if smoothing > 1:
            try:
                from scipy.ndimage import uniform_filter

                hf = uniform_filter(hf, size=smoothing).astype(np.float32)
            except ImportError:
                pass  # no scipy, use raw noise

    return hf


def print_controls():
    print("\n============ KEYBOARD CONTROLS ============")
    print(f"  P           : forward   (vx={FORWARD_VX:.2f})")
    print(f"  M           : backward  (vx={-BACKWARD_VX:.2f})")
    print(f"  K           : right     (vy={RIGHT_VY:.2f})")
    print(f"  J           : left      (vy={-LEFT_VY:.2f})")
    print(f"  U           : yaw CW    (wz={-YAW_CW_WZ:.2f})")
    print(f"  O           : yaw CCW   (wz={YAW_CCW_WZ:.2f})")
    print("  SPACE       : zero velocity (stop)")
    if TERRAIN_ENABLE:
        print("  ─────────── TERRAIN PRESETS ───────────")
        for i, p in enumerate(_TERRAIN_PRESETS):
            print(f"  {i + 1}           : {p['label']}")
        print("  R           : respawn (new random terrain at same difficulty)")
    print("  ─────────── OTHER ─────────────")
    print("  X / ESC     : quit")
    print("  CTRL+C      : quit")
    print("============================================\n")


# =====================================================================
# GLOBAL KEYBOARD LISTENER (works even when viewer is focused)
# =====================================================================

_quit_event = threading.Event()
_pressed_mods = set()


def _normalize_key(k) -> Optional[str]:
    if k == keyboard.Key.space:
        return " "
    if k == keyboard.Key.esc:
        return "ESC"
    try:
        ch = k.char
        if ch is None:
            return None
        return ch.lower()
    except AttributeError:
        return None


def _on_press(k):
    if k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        _pressed_mods.add("ctrl")
        return

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
        return False


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
# Terrain helpers
# =====================================================================


def respawn_on_tile(env, row, col):
    """Teleport the single eval robot to the center of the terrain or flat ground."""
    terrain = env.terrain

    if terrain is None or not terrain.enabled:
        # Flat ground — just reset to default pos
        env.robot.set_dofs_position(
            position=env.default_dof_pos.unsqueeze(0),
            dofs_idx_local=env.motors_dof_idx,
            zero_velocity=True,
            envs_idx=[0],
        )
        env.robot.set_pos(
            env.base_init_pos.unsqueeze(0), zero_velocity=True, envs_idx=[0]
        )
        env.robot.set_quat(
            env.base_init_quat.unsqueeze(0), zero_velocity=True, envs_idx=[0]
        )
        env.robot.zero_all_dofs_velocity([0])
        print("\n  >> Respawned on flat ground")
        return

    # ---- Tile-based terrain (subterrain grid) ----
    if hasattr(terrain, "_tile_centers") and terrain._tile_centers is not None:
        r = min(row, terrain.n_rows - 1)
        c = min(col, terrain.n_cols - 1)

        center = terrain._tile_centers[r, c].clone()
        ground_z = terrain._query_height_batch(center[0:1], center[1:2])
        base_z = float(env.base_init_pos[2])
        spawn_z = ground_z[0].item() + base_z + terrain.spawn_height_offset

        spawn_pos = torch.tensor(
            [[center[0].item(), center[1].item(), spawn_z]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        terrain._env_tile_row[0] = r
        terrain._env_tile_col[0] = c
        terrain._env_spawn_pos[0] = spawn_pos[0]

        terrain_type = terrain.subterrain_types[r][c]
        label = f"Row {r}, Col {c}: {terrain_type}"

    # ---- Custom heightmap (no tile grid) ----
    else:
        base_z = float(env.base_init_pos[2])
        spawn_offset = getattr(terrain, "spawn_height_offset", 0.05)

        # Query ground height at center of heightmap
        if hasattr(terrain, "_query_height_batch"):
            cx = torch.tensor([0.0], device=gs.device)
            cy = torch.tensor([0.0], device=gs.device)
            ground_z = terrain._query_height_batch(cx, cy)
            spawn_z = ground_z[0].item() + base_z + spawn_offset
        else:
            spawn_z = base_z + spawn_offset

        spawn_pos = torch.tensor(
            [[0.0, 0.0, spawn_z]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        if hasattr(terrain, "_env_spawn_pos"):
            terrain._env_spawn_pos[0] = spawn_pos[0]

        label = "custom heightmap"

    # Reset robot
    env.robot.set_dofs_position(
        position=env.default_dof_pos.unsqueeze(0),
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=[0],
    )
    env.robot.set_pos(spawn_pos, zero_velocity=True, envs_idx=[0])
    env.robot.set_quat(
        env.base_init_quat.unsqueeze(0), zero_velocity=True, envs_idx=[0]
    )
    env.robot.zero_all_dofs_velocity([0])

    # Reset env buffers
    env.last_actions[0] = 0.0
    env.last_dof_vel[0] = 0.0
    env._applied_actions[0] = 0.0
    env._action_history[0] = 0.0
    env.episode_length_buf[0] = 0

    print(
        f"\n  >> Spawned on {label} "
        f"at ({spawn_pos[0, 0]:.1f}, {spawn_pos[0, 1]:.1f}, {spawn_pos[0, 2]:.2f})"
    )


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-terrain-v8")
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

    # Disable privileged obs — not needed for inference (only actor runs,
    # not critic), and avoids dimension mismatch with old checkpoints
    obs_cfg["num_privileged_obs"] = None

    # ================================================================
    # Terrain config — custom numpy heightmap for EXACT height control
    # ================================================================
    if TERRAIN_ENABLE:
        hf = generate_heightmap(
            MIN_HEIGHT,
            MAX_HEIGHT,
            TERRAIN_SIZE_X,
            TERRAIN_SIZE_Y,
            TERRAIN_HORIZONTAL_SCALE,
            TERRAIN_SMOOTHING,
        )

        print("\n  Custom heightmap generated:")
        print(f"    shape       : {hf.shape}")
        print(f"    height range: [{hf.min():.4f}, {hf.max():.4f}] m")
        print(f"    h_scale     : {TERRAIN_HORIZONTAL_SCALE} m/px")
        print(f"    covers      : {TERRAIN_SIZE_X} x {TERRAIN_SIZE_Y} m")

        env_cfg["terrain"] = {
            "enabled": True,
            "height_field": hf,
            "horizontal_scale": TERRAIN_HORIZONTAL_SCALE,
            "vertical_scale": 1.0,  # heights already in meters — no extra scaling
            "spawn_height_offset": 0.05,
            "boundary_margin": 1.0,
            "terrain_name": None,  # don't cache — new terrain every run
        }
    else:
        env_cfg["terrain"] = {"enabled": False}

    # Disable termination (let it run forever for eval)
    env_cfg["termination_if_roll_greater_than"] = 1e9
    env_cfg["termination_if_pitch_greater_than"] = 1e9
    env_cfg["termination_if_z_vel_greater_than"] = 1e9
    env_cfg["termination_if_y_vel_greater_than"] = 1e9

    reward_cfg["reward_scales"] = {}

    # Initial pose overrides (only meaningful on flat terrain)
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

    # Set Friction (env.ground works for both terrain and flat plane)
    env.ground.set_friction(GROUND_FRICTION)
    env.robot.set_friction(ROBOT_FRICTION)

    if MOTOR_STRENGTH_ENABLE and pls_enabled:
        env._motor_strength[:] = MOTOR_STRENGTH_SCALE
        env._kp_factors[:] = 1.0
        env._kd_factors[:] = 1.0

    # ================================================================
    # Load Policy
    # ================================================================
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Load checkpoint, skipping critic weights that don't match shape
    # (old checkpoint has critic for 104 privileged obs, we only need actor)
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    loaded = torch.load(ckpt_path, map_location=gs.device, weights_only=True)

    model_dict = runner.alg.actor_critic.state_dict()
    pretrained_dict = {}
    skipped = []
    for k, v in loaded["model_state_dict"].items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
        else:
            skipped.append(k)
    model_dict.update(pretrained_dict)
    runner.alg.actor_critic.load_state_dict(model_dict)

    if skipped:
        print(
            f"  Skipped {len(skipped)} weight(s) with shape mismatch (critic): {skipped}"
        )
    print(f"  Loaded actor weights from: {ckpt_path}")

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
    # Reset & initial terrain spawn
    # ================================================================
    obs, _ = env.reset()

    zero_action = torch.zeros((1, env.num_actions), device=gs.device)
    for _ in range(ACTION_DELAY_STEPS + 1):
        action_buffer.append(zero_action.clone())

    # Spawn on terrain
    if TERRAIN_ENABLE:
        respawn_on_tile(env, TERRAIN_SELECTED_ROW, TERRAIN_SELECTED_COL)
        obs, _ = env.get_observations()

    step = 0

    # ================================================================
    # Config Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  INFERENCE CONFIG (v9) — CUSTOM HEIGHTMAP + KEYBOARD")
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

    # Terrain summary
    print("  ─── Terrain ───")
    print(
        f"  Terrain         : {'CUSTOM HEIGHTMAP' if TERRAIN_ENABLE else 'OFF (flat plane)'}"
    )
    if TERRAIN_ENABLE:
        print(f"  Bump range      : [{MIN_HEIGHT:.3f}, {MAX_HEIGHT:.3f}] m")
        print(f"  Smoothing       : {TERRAIN_SMOOTHING}")
        print(f"  Difficulty presets (press 1-{len(_TERRAIN_PRESETS)}):")
        for i, p in enumerate(_TERRAIN_PRESETS):
            marker = " <-- current" if i == 0 else ""
            print(f"    [{i + 1}] {p['label']}{marker}")

    print("=" * 60)

    print_controls()
    print("Running evaluation...\n")

    # ================================================================
    # Main Loop
    # ================================================================
    listener = start_keyboard_listener()

    try:
        with torch.no_grad():
            while True:
                step += 1

                if _quit_event.is_set():
                    break

                # --- Check for terrain respawn request ---
                do_respawn = False
                with _state_lock:
                    if terrain_state["respawn_requested"]:
                        terrain_state["respawn_requested"] = False
                        do_respawn = True
                        respawn_row = terrain_state["row"]

                if do_respawn and TERRAIN_ENABLE:
                    # Generate new heightmap from preset
                    preset = _TERRAIN_PRESETS[
                        min(respawn_row, len(_TERRAIN_PRESETS) - 1)
                    ]
                    new_hf = generate_heightmap(
                        preset["min_h"],
                        preset["max_h"],
                        TERRAIN_SIZE_X,
                        TERRAIN_SIZE_Y,
                        TERRAIN_HORIZONTAL_SCALE,
                        TERRAIN_SMOOTHING,
                    )
                    print(
                        f"\n  New terrain: {preset['label']} "
                        f"[{new_hf.min():.4f}, {new_hf.max():.4f}] m"
                    )

                    # Rebuild terrain in the scene
                    # NOTE: This regenerates the heightmap. If your Go2Terrain
                    # wrapper supports update_heightfield(), use that instead.
                    # Otherwise we just respawn on the existing terrain.
                    if hasattr(env, "terrain") and env.terrain is not None:
                        if hasattr(env.terrain, "_height_field_np"):
                            env.terrain._height_field_np = new_hf
                        if hasattr(env.terrain, "_update_heightfield"):
                            env.terrain._update_heightfield(new_hf)

                    respawn_on_tile(env, 0, 0)
                    obs, _ = env.get_observations()
                    action_buffer.clear()
                    for _ in range(ACTION_DELAY_STEPS + 1):
                        action_buffer.append(zero_action.clone())
                    continue
                elif do_respawn:
                    respawn_on_tile(env, 0, 0)
                    obs, _ = env.get_observations()
                    action_buffer.clear()
                    for _ in range(ACTION_DELAY_STEPS + 1):
                        action_buffer.append(zero_action.clone())
                    continue

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
                #  DEBUG PRINTS
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

                    # Terrain info in debug line
                    if TERRAIN_ENABLE and hasattr(env, "_ground_height"):
                        with _state_lock:
                            pr = terrain_state["row"]
                        preset_label = _TERRAIN_PRESETS[
                            min(pr, len(_TERRAIN_PRESETS) - 1)
                        ]["label"]
                        ground_z = env._ground_height[0].item()
                        h_above = pos[2].item() - ground_z
                        parts.append(f"[{preset_label} h={h_above:.2f}m]")

                    print(
                        f"\r[step {step}] " + " | ".join(parts) + "    ",
                        end="",
                        flush=True,
                    )

                # ==========================================================
                #  STEP
                # ==========================================================
                obs, _, rsets, _ = env.step(actions_to_apply)

                # Auto-respawn if boundary reset triggered
                if rsets[0].item():
                    respawn_on_tile(env, 0, 0)
                    obs, _ = env.get_observations()
                    action_buffer.clear()
                    for _ in range(ACTION_DELAY_STEPS + 1):
                        action_buffer.append(zero_action.clone())

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
