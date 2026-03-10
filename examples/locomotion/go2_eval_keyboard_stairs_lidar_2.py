"""
eval_sim_info.py
=================
Sim evaluation of Go2 stair climbing — mirrors real deployment.

Unlike training with 13 difficulty rows, this builds a SINGLE terrain
with user-specified stair height and depth (just like on the real robot).

Usage:
  python eval_sim_info.py -e go2-stairs-v5-info --ckpt 5000 --height 0.15 --depth 0.25
  python eval_sim_info.py -e go2-stairs-v5-info --ckpt 5000 --height 0.0   # flat ground

Controls (same as real deployment):
  W/S     : forward/backward
  A/D     : strafe left/right
  Q/E     : yaw CW/CCW
  SPACE   : zero velocity
  F       : respawn at start
  X/ESC   : quit
"""

import argparse
import os
import pickle
import threading
from collections import deque
from importlib import metadata

import torch
from pynput import keyboard

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
from go2_env_stair_lidar_2 import Go2Env
from rsl_rl.runners import OnPolicyRunner

####################################

STAIR_HEIGHT = 0.20  # riser height in meters
STAIR_DEPTH = 0.25  # tread depth in meters
NUM_FLIGHTS = 2  # number of up-down cycles
NUM_STEPS_PER_FLIGHT = 10  # steps per flight

# ============================================================
# VELOCITY PRESETS
# ============================================================
FORWARD_VX = 0.5
BACKWARD_VX = 0.5
LEFT_VY = 0.7
RIGHT_VY = 0.7
YAW_CW_WZ = 0.7
YAW_CCW_WZ = 0.7

# ============================================================
# EVAL-TIME DR SETTINGS
# ============================================================
KP = 60.0
KD = 2.0
GROUND_FRICTION = 0.6
ROBOT_FRICTION = 0.6

OBS_NOISE_ENABLE = True
OBS_NOISE_LEVEL = 0.1
OBS_NOISE_STD = {
    "ang_vel": 0.2,
    "gravity": 0.05,
    "commands": 0.0,
    "dof_pos": 0.01,
    "dof_vel": 1.5,
    "actions": 0.0,
    "stair_info": 0.0,  # stair info noise handled by env
}

ACTION_NOISE_ENABLE = True
ACTION_NOISE_STD = 0.1

ACTION_DELAY_ENABLE = True
ACTION_DELAY_STEPS = 1

MOTOR_STRENGTH_ENABLE = False
MOTOR_STRENGTH_SCALE = 0.8

DEBUG_PRINT_INTERVAL = 100

# ============================================================
# KEYBOARD STATE
# ============================================================
command_state = {"vx": FORWARD_VX, "vy": 0.0, "wz": 0.0}
control_state = {"respawn_requested": False}
_state_lock = threading.Lock()
_quit_event = threading.Event()
_pressed_mods = set()


def make_command_tensor():
    with _state_lock:
        return torch.tensor(
            [[command_state["vx"], command_state["vy"], command_state["wz"]]]
        )


def handle_key(key):
    if key is None:
        return True
    if isinstance(key, str) and len(key) == 1:
        key = key.lower()
    if key in ("x", "CTRL_C", "ESC"):
        return False

    with _state_lock:
        if key == "w":
            command_state.update({"vx": FORWARD_VX, "vy": 0.0, "wz": 0.0})
        elif key == "s":
            command_state.update({"vx": -BACKWARD_VX, "vy": 0.0, "wz": 0.0})
        elif key == "d":
            command_state.update({"vx": 0.0, "vy": RIGHT_VY, "wz": 0.0})
        elif key == "a":
            command_state.update({"vx": 0.0, "vy": -LEFT_VY, "wz": 0.0})
        elif key == "q":
            command_state.update({"vx": 0.0, "vy": 0.0, "wz": -YAW_CW_WZ})
        elif key == "e":
            command_state.update({"vx": 0.0, "vy": 0.0, "wz": YAW_CCW_WZ})
        elif key == " ":
            command_state.update({"vx": 0.0, "vy": 0.0, "wz": 0.0})
        elif key == "f":
            control_state["respawn_requested"] = True
    return True


def _normalize_key(k):
    if k == keyboard.Key.space:
        return " "
    if k == keyboard.Key.esc:
        return "ESC"
    try:
        ch = k.char
        return ch.lower() if ch else None
    except AttributeError:
        return None


def _on_press(k):
    if k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        _pressed_mods.add("ctrl")
        return
    try:
        ch = k.char
        if ch and ch.lower() == "c" and ("ctrl" in _pressed_mods):
            _quit_event.set()
            return False
    except AttributeError:
        pass
    key_str = _normalize_key(k)
    if key_str and not handle_key(key_str):
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


def print_controls():
    print("\n============ KEYBOARD CONTROLS ============")
    print(f"  W : forward   (vx={FORWARD_VX:.2f})")
    print(f"  S : backward  (vx={-BACKWARD_VX:.2f})")
    print("  A : left      D : right")
    print("  Q : yaw CW    E : yaw CCW")
    print("  SPACE : stop")
    print("  F     : respawn at start")
    print("  X/ESC : quit")
    print("============================================\n")


# ============================================================
# HELPERS
# ============================================================


def rand_float(lower, upper, shape):
    return (upper - lower) * torch.rand(shape, device=gs.device) + lower


def build_obs_noise_vec(obs_dim, noise_cfg, level, num_stair_info=4):
    proprio_size = 3 + 3 + 3 + 12 + 12  # 33
    action_size = obs_dim - proprio_size - num_stair_info

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

    if num_stair_info > 0:
        parts.append(
            torch.full(
                (num_stair_info,),
                noise_cfg.get("stair_info", 0.0) * level,
                device=gs.device,
            )
        )

    return torch.cat(parts)[:obs_dim].unsqueeze(0)


def respawn_at_start(env):
    """Teleport env 0 back to spawn point."""
    envs_idx = torch.tensor([0], device=gs.device, dtype=torch.long)

    if env._use_terrain:
        row = int(env._env_terrain_row[0].item())
        cx, cy, cz = env._terrain_row_centers[row]
        spawn = torch.tensor(
            [[cx, cy, cz + env.base_init_pos[2].item()]],
            device=gs.device,
            dtype=gs.tc_float,
        )
    else:
        spawn = env.base_init_pos.unsqueeze(0).clone()

    env.dof_pos[0] = env.default_dof_pos
    env.dof_vel[0] = 0.0
    env.robot.set_dofs_position(
        position=env.dof_pos[envs_idx],
        dofs_idx_local=env.motors_dof_idx,
        zero_velocity=True,
        envs_idx=envs_idx,
    )

    env.base_pos[0] = spawn[0]
    env.base_quat[0] = env.base_init_quat
    env.robot.set_pos(spawn, zero_velocity=False, envs_idx=envs_idx)
    env.robot.set_quat(
        env.base_init_quat.unsqueeze(0), zero_velocity=False, envs_idx=envs_idx
    )
    env.robot.zero_all_dofs_velocity(envs_idx)

    env.last_actions[0] = 0.0
    env.last_dof_vel[0] = 0.0
    env._applied_actions[0] = 0.0
    env._action_history[0] = 0.0
    env.base_lin_vel[0] = 0.0
    env.base_ang_vel[0] = 0.0
    if hasattr(env, "_last_base_pos_x"):
        env._last_base_pos_x[0] = spawn[0, 0].item()
    if env._stair_cfg.enabled:
        env._stair_info_buf[0] = 0.0
        env._stair_info_prev[0] = 0.0
        env._stair_info_clean[0] = 0.0


# ============================================================
# MODEL LOADING
# ============================================================


def load_model_compat(runner, ckpt_path, env):
    ckpt = torch.load(ckpt_path, map_location=gs.device, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)

    # Check actor dim
    actor_key = None
    for k in model_state:
        if "actor" in k.lower() and "weight" in k.lower():
            actor_key = k
            break

    if actor_key is not None:
        saved = model_state[actor_key].shape[1]
        expected = env.num_obs
        if saved != expected:
            raise RuntimeError(
                f"Actor dim mismatch: checkpoint has {saved}, env expects {expected}. "
                f"Need a checkpoint from train_stair5_info.py (53-dim actor obs)."
            )

    # Check critic dim (mismatch is OK, critic re-initialises)
    critic_mismatch = 0
    for k in model_state:
        if "critic" in k.lower() and "weight" in k.lower():
            saved_c = model_state[k].shape[1]
            expected_c = (
                env.num_privileged_obs if env.num_privileged_obs else env.num_obs
            )
            if saved_c != expected_c:
                critic_mismatch = expected_c - saved_c
                print(
                    f"[COMPAT] Critic dim mismatch: saved={saved_c}, expected={expected_c}. "
                    f"Actor OK, critic re-initialised."
                )
            break

    try:
        runner.load(ckpt_path)
    except RuntimeError as e:
        if "size mismatch" in str(e).lower():
            print("[COMPAT] Partial load due to critic mismatch...")
            current = runner.alg.actor_critic.state_dict()
            loaded = 0
            for k, v in model_state.items():
                if k in current and current[k].shape == v.shape:
                    current[k] = v
                    loaded += 1
            runner.alg.actor_critic.load_state_dict(current, strict=False)
            print(f"[COMPAT] Loaded {loaded} matching tensors")
        else:
            raise

    return critic_mismatch


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sim evaluation — mirrors real deployment"
    )
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairs-v5-info")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument(
        "--height",
        type=float,
        default=STAIR_HEIGHT,
        help="Stair riser height in meters (0 = flat ground)",
    )
    parser.add_argument(
        "--depth", type=float, default=STAIR_DEPTH, help="Stair tread depth in meters"
    )
    parser.add_argument(
        "--flights",
        type=int,
        default=NUM_FLIGHTS,
        help="Number of up-down flight cycles",
    )
    parser.add_argument(
        "--steps", type=int, default=NUM_STEPS_PER_FLIGHT, help="Steps per flight"
    )
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # ================================================================
    # Build single-row terrain with user-specified dimensions
    # ================================================================
    use_stairs = args.height > 0.001

    if use_stairs:
        env_cfg["terrain"] = {
            "enabled": True,
            "horizontal_scale": 0.05,
            "vertical_scale": 0.005,
            "num_difficulty_rows": 1,  # SINGLE ROW
            "row_width_m": 6.0,
            "step_depth_range": [args.depth, args.depth],  # exact depth
            "num_steps": args.steps,
            "num_flights": args.flights,
            "step_height_min": args.height,  # exact height
            "step_height_max": args.height,
            "flat_before_m": 4.0,
            "flat_top_m": 1.5,
            "flat_gap_m": 1.5,
            "flat_after_m": 2.0,
        }
    else:
        if "terrain" in env_cfg:
            env_cfg["terrain"]["enabled"] = False

    # ================================================================
    # Disable training-time DR and curriculum
    # ================================================================
    pls_enabled = env_cfg.get("pls_enable", False)

    if not pls_enabled:
        effective_kp = KP * (MOTOR_STRENGTH_SCALE if MOTOR_STRENGTH_ENABLE else 1.0)
        env_cfg["kp"] = effective_kp
        env_cfg["kd"] = KD

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
        "gravity_offset_range",
        "motor_strength_range",
        "init_pos_z_range",
        "init_euler_range",
    ]:
        env_cfg.pop(key, None)

    env_cfg["min_delay_steps"] = 0
    env_cfg["max_delay_steps"] = 0
    env_cfg.pop("dr_schedule", None)

    if "curriculum" in env_cfg:
        env_cfg["curriculum"]["enabled"] = False

    # Zero out stair info noise for clean eval
    si_cfg = env_cfg.get("stair_info", {})
    if isinstance(si_cfg, dict) and si_cfg.get("enabled"):
        noise = si_cfg.get("noise", {})
        for k in list(noise.keys()):
            noise[k] = 0.0
        si_cfg["noise"] = noise

    # Disable termination
    env_cfg["termination_if_roll_greater_than"] = 1e9
    env_cfg["termination_if_pitch_greater_than"] = 1e9
    env_cfg["termination_if_z_vel_greater_than"] = 1e9
    env_cfg["termination_if_y_vel_greater_than"] = 1e9

    reward_cfg["reward_scales"] = {}

    # ================================================================
    # Create env
    # ================================================================
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Set friction
    if env._use_terrain:
        env.ground.set_friction(GROUND_FRICTION)
    else:
        for ent in env.scene.entities:
            try:
                if hasattr(ent, "morph") and hasattr(ent.morph, "file"):
                    if "plane.urdf" in ent.morph.file:
                        ent.set_friction(GROUND_FRICTION)
                        break
            except Exception:
                pass
    env.robot.set_friction(ROBOT_FRICTION)

    if MOTOR_STRENGTH_ENABLE and pls_enabled:
        env._motor_strength[:] = MOTOR_STRENGTH_SCALE

    # ================================================================
    # Load policy
    # ================================================================
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    load_model_compat(runner, ckpt_path, env)
    policy = runner.get_inference_policy(device=gs.device)

    # ================================================================
    # Initial spawn
    # ================================================================
    obs, _ = env.reset()
    env._lock_terrain_rows = True

    zero_action = torch.zeros((1, env.num_actions), device=gs.device)
    action_buffer = deque(maxlen=ACTION_DELAY_STEPS + 1)
    for _ in range(ACTION_DELAY_STEPS + 1):
        action_buffer.append(zero_action.clone())

    obs_noise_vec = None
    step = 0

    # ================================================================
    # Print config
    # ================================================================
    stair_enabled = env._stair_cfg.enabled if hasattr(env, "_stair_cfg") else False

    print("\n" + "=" * 64)
    print("  SIM EVALUATION — mirrors real deployment")
    print("=" * 64)
    if use_stairs:
        print(f"  Stair height   : {args.height * 100:.1f}cm")
        print(f"  Stair depth    : {args.depth * 100:.1f}cm")
        print(f"  Flights        : {args.flights}  ({args.steps} steps each)")
    else:
        print("  Terrain        : FLAT (no stairs)")
    print(f"  Stair info     : {'ENABLED' if stair_enabled else 'DISABLED'} (4 values)")
    print(f"  Actor obs      : {obs_cfg['num_obs']}")
    print(f"  PLS            : {'ON' if pls_enabled else 'OFF'}")
    print(
        f"  Obs noise      : {'ON' if OBS_NOISE_ENABLE else 'OFF'} level={OBS_NOISE_LEVEL}"
    )
    print(
        f"  Action noise   : {'ON' if ACTION_NOISE_ENABLE else 'OFF'} std={ACTION_NOISE_STD}"
    )
    print(
        f"  Action delay   : {'ON' if ACTION_DELAY_ENABLE else 'OFF'} steps={ACTION_DELAY_STEPS}"
    )
    print("=" * 64)

    print_controls()
    print("Running sim evaluation...\n")

    # ================================================================
    # Main loop
    # ================================================================
    listener = start_keyboard_listener()

    try:
        with torch.no_grad():
            while True:
                step += 1

                if _quit_event.is_set():
                    break

                # Check respawn
                with _state_lock:
                    if control_state["respawn_requested"]:
                        control_state["respawn_requested"] = False
                        print("\n[Respawn] Back to start")
                        respawn_at_start(env)
                        action_buffer.clear()
                        for _ in range(ACTION_DELAY_STEPS + 1):
                            action_buffer.append(zero_action.clone())
                        obs, _ = env.get_observations()

                # Set commands
                env.commands[:] = make_command_tensor().to(gs.device)

                # Observation noise
                policy_obs = obs.clone()
                if OBS_NOISE_ENABLE:
                    if obs_noise_vec is None:
                        obs_noise_vec = build_obs_noise_vec(
                            obs.shape[-1],
                            OBS_NOISE_STD,
                            OBS_NOISE_LEVEL,
                            num_stair_info=4,
                        )
                    policy_obs = policy_obs + torch.randn_like(obs) * obs_noise_vec

                # Policy inference
                raw_actions = policy(policy_obs)

                # Action noise
                if ACTION_NOISE_ENABLE:
                    num_pos = env.num_pos_actions
                    noise = torch.zeros_like(raw_actions)
                    noise[:, :num_pos] = (
                        torch.randn(1, num_pos, device=gs.device) * ACTION_NOISE_STD
                    )
                    raw_actions = raw_actions + noise

                # Action delay
                if ACTION_DELAY_ENABLE:
                    action_buffer.append(raw_actions.clone())
                    actions_to_apply = action_buffer[0]
                else:
                    actions_to_apply = raw_actions

                # Debug print
                if DEBUG_PRINT_INTERVAL > 0 and step % DEBUG_PRINT_INTERVAL == 0:
                    parts = []

                    cmd = env.commands[0]
                    parts.append(
                        f"cmd=[{cmd[0].item():+.2f},{cmd[1].item():+.2f},{cmd[2].item():+.2f}]"
                    )

                    pos = env.base_pos[0]
                    vel = env.base_lin_vel[0]

                    if env._use_terrain:
                        terrain_z = env._get_terrain_height(pos[0:1], pos[1:2]).item()
                        hag = pos[2].item() - terrain_z
                        parts.append(f"hag={hag:.3f}m")

                    # Stair info
                    if stair_enabled and hasattr(env, "_stair_info_clean"):
                        info = env._stair_info_clean[0]
                        edge_raw = info[0].item() / env._stair_cfg.edge_dist_scale
                        dir_raw = info[3].item() / env._stair_cfg.direction_scale
                        dir_sym = {1.0: "↑", -1.0: "↓"}.get(dir_raw, "—")
                        parts.append(f"edge={edge_raw:.2f}m {dir_sym}")

                    parts.append(
                        f"vel=[{vel[0].item():.2f},{vel[1].item():.2f},{vel[2].item():.2f}]"
                    )

                    if pls_enabled:
                        sa = actions_to_apply[0, env.num_pos_actions :]
                        kp_per_leg = env.pls_kp_default + sa * env.pls_kp_action_scale
                        kp_per_leg = torch.clamp(
                            kp_per_leg, env.pls_kp_range[0], env.pls_kp_range[1]
                        )
                        parts.append(
                            f"Kp=[{kp_per_leg[0].item():.0f},{kp_per_leg[1].item():.0f},"
                            f"{kp_per_leg[2].item():.0f},{kp_per_leg[3].item():.0f}]"
                        )

                    print(
                        f"\r[step {step}] " + " | ".join(parts) + "    ",
                        end="",
                        flush=True,
                    )

                # Step
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
