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
from go2_env_stair6_Omni import Go2Env
from rsl_rl.runners import OnPolicyRunner

# ============================================================
# PREDEFINED VELOCITIES
# ============================================================
FORWARD_VX = 0.5
BACKWARD_VX = 0.5
LEFT_VY = 0.7
RIGHT_VY = 0.7
YAW_CW_WZ = 0.7
YAW_CCW_WZ = 0.7

# ============================================================
# TERRAIN — set to match training or override
# ============================================================
STAIR_HEIGHT = 0.135  # override step height (0 = use training cfg)
USE_FLAT_GROUND = False  # True = flat ground

# ============================================================
# SPAWN
# ============================================================
# Which well difficulty to spawn in (0 = easiest, 8 = hardest, -1 = use training cfg)
SPAWN_WELL_LEVEL = 8
SPAWN_LOCATION = "bottom"  # "bottom" only for stairwells

# -------------------- 1. PD GAINS -----------------------
KP = 60.0
KD = 2.0

# -------------------- 2. FRICTION --------------------
GROUND_FRICTION = 0.6
ROBOT_FRICTION = 0.6

# -------------------- 3. PUSHES ----------------------------
PUSH_ENABLE = False
PUSH_FORCE_RANGE = (200.0, 200.0)
PUSH_DURATION_S = [0.05, 0.15]
PUSH_INTERVAL_S = 2.0
PUSH_Z_SCALE = 0.0

# -------------------- 4. OBS NOISE --------------------------
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

# -------------------- 5. ACTION NOISE -----------------
ACTION_NOISE_ENABLE = True
ACTION_NOISE_STD = 0.01

# -------------------- 6. ACTION DELAY --------------------
ACTION_DELAY_ENABLE = True
ACTION_DELAY_STEPS = 1

# -------------------- 7. PAYLOAD ----------------------
PAYLOAD_ENABLE = False
PAYLOAD_MASS = 0.0

# -------------------- 8. GRAVITY PERTURB ----------------------
GRAVITY_PERTURB_ENABLE = True
GRAVITY_PERTURB_X = 0.0
GRAVITY_PERTURB_Y = 0.0
GO2_APPROX_MASS = 15.0

# -------------------- 9. MOTOR STRENGTH --------------------
MOTOR_STRENGTH_ENABLE = False
MOTOR_STRENGTH_SCALE = 0.8

# -------------------- 10. GRAVITY OBS OFFSET ----------------
GRAVITY_OBS_OFFSET_ENABLE = False
GRAVITY_OBS_OFFSET = [0.0, 0.0, 0.0]

# -------------------- DEBUG ----------------------------------------
DEBUG_PRINT_INTERVAL = 200

# =====================================================================
# KEYBOARD STATE
# =====================================================================
command_state = {"vx": FORWARD_VX, "vy": 0.0, "wz": 0.0}
control_state = {"respawn_requested": False}
_state_lock = threading.Lock()


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
        # WASD
        if key == "w":
            command_state["vx"] = FORWARD_VX
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key == "s":
            command_state["vx"] = -BACKWARD_VX
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key == "d":
            command_state["vx"] = 0.0
            command_state["vy"] = RIGHT_VY
            command_state["wz"] = 0.0
        elif key == "a":
            command_state["vx"] = 0.0
            command_state["vy"] = -LEFT_VY
            command_state["wz"] = 0.0
        elif key == "q":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = -YAW_CW_WZ
        elif key == "e":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = YAW_CCW_WZ
        elif key == " ":
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        # Numpad / number keys
        elif key in ("NUM_8", "8"):
            command_state["vx"] = FORWARD_VX
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key in ("NUM_2", "2"):
            command_state["vx"] = -BACKWARD_VX
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key in ("NUM_4", "4"):
            command_state["vx"] = 0.0
            command_state["vy"] = -LEFT_VY
            command_state["wz"] = 0.0
        elif key in ("NUM_6", "6"):
            command_state["vx"] = 0.0
            command_state["vy"] = RIGHT_VY
            command_state["wz"] = 0.0
        elif key in ("NUM_7", "7"):
            command_state["vx"] = FORWARD_VX * 0.7
            command_state["vy"] = -LEFT_VY * 0.7
            command_state["wz"] = 0.0
        elif key in ("NUM_9", "9"):
            command_state["vx"] = FORWARD_VX * 0.7
            command_state["vy"] = RIGHT_VY * 0.7
            command_state["wz"] = 0.0
        elif key in ("NUM_1", "1"):
            command_state["vx"] = -BACKWARD_VX * 0.7
            command_state["vy"] = -LEFT_VY * 0.7
            command_state["wz"] = 0.0
        elif key in ("NUM_3", "3"):
            command_state["vx"] = -BACKWARD_VX * 0.7
            command_state["vy"] = RIGHT_VY * 0.7
            command_state["wz"] = 0.0
        elif key in ("NUM_5", "5"):
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = 0.0
        elif key in ("NUM_PLUS", "+"):
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = YAW_CCW_WZ
        elif key in ("NUM_MINUS", "-"):
            command_state["vx"] = 0.0
            command_state["vy"] = 0.0
            command_state["wz"] = -YAW_CW_WZ
        elif key == "f":
            control_state["respawn_requested"] = True
    return True


def print_controls():
    print("\n============ KEYBOARD CONTROLS ============")
    print("  WASD + QE:")
    print("    W/8 : forward    S/2 : backward")
    print("    A/4 : left       D/6 : right")
    print("    Q/- : yaw CW     E/+ : yaw CCW")
    print("    SPACE/5 : stop")
    print("  Numpad diagonals: 7(↖) 9(↗) 1(↙) 3(↘)")
    print("  F : respawn    X/ESC : quit")
    print("============================================\n")


_quit_event = threading.Event()
_pressed_mods = set()


def _normalize_key(k) -> Optional[str]:
    if k == keyboard.Key.space:
        return " "
    if k == keyboard.Key.esc:
        return "ESC"
    try:
        vk = getattr(k, "vk", None)
        if vk is not None:
            vk_map = {
                96: "NUM_0",
                97: "NUM_1",
                98: "NUM_2",
                99: "NUM_3",
                100: "NUM_4",
                101: "NUM_5",
                102: "NUM_6",
                103: "NUM_7",
                104: "NUM_8",
                105: "NUM_9",
                107: "NUM_PLUS",
                109: "NUM_MINUS",
                65456: "NUM_0",
                65457: "NUM_1",
                65458: "NUM_2",
                65459: "NUM_3",
                65460: "NUM_4",
                65461: "NUM_5",
                65462: "NUM_6",
                65463: "NUM_7",
                65464: "NUM_8",
                65465: "NUM_9",
                65451: "NUM_PLUS",
                65453: "NUM_MINUS",
            }
            if vk in vk_map:
                return vk_map[vk]
    except Exception:
        pass
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
        if ch and ch.lower() == "c" and "ctrl" in _pressed_mods:
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


# =====================================================================
# Helpers
# =====================================================================


def rand_float(lo, hi, shape):
    return (hi - lo) * torch.rand(shape, device=gs.device) + lo


def rand_int(lo, hi, shape):
    return torch.randint(lo, hi + 1, size=shape, device=gs.device)


def build_obs_noise_vec(obs_dim, noise_cfg, level):
    fixed = 3 + 3 + 3 + 12 + 12
    action_size = obs_dim - fixed
    parts = []
    for name, size in [
        ("ang_vel", 3),
        ("gravity", 3),
        ("commands", 3),
        ("dof_pos", 12),
        ("dof_vel", 12),
        ("actions", action_size),
    ]:
        parts.append(
            torch.full((size,), noise_cfg.get(name, 0.0) * level, device=gs.device)
        )
    return torch.cat(parts)[:obs_dim].unsqueeze(0)


def respawn(env, well_level=-1):
    """Teleport env 0 to bottom of specified well."""
    import random

    envs_idx = torch.tensor([0], device=gs.device, dtype=torch.long)

    if env._use_terrain:
        if well_level < 0:
            well_level = 0
        well_level = min(well_level, env._num_difficulty_levels - 1)
        info = env._well_spawns[well_level]
        cx, cy, cz = info["center"]
        dx = random.uniform(-0.3, 0.3)
        dy = random.uniform(-0.3, 0.3)
        spawn = torch.tensor(
            [[cx + dx, cy + dy, cz + env.base_init_pos[2].item()]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        print(
            f"\n[Respawn] Well {well_level} ({info['step_height_m'] * 100:.1f}cm) → "
            f"({cx + dx:.1f}, {cy + dy:.1f}, z={cz:.3f})"
        )
    else:
        spawn = env.base_init_pos.unsqueeze(0).clone()
        print("\n[Respawn] Flat ground")

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


def load_model_compat(runner, ckpt_path, env):
    ckpt = torch.load(ckpt_path, map_location=gs.device, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)

    critic_key = None
    for k in model_state:
        if "critic" in k.lower() and "weight" in k.lower():
            critic_key = k
            break

    mismatch = 0
    if critic_key is not None:
        saved_in = model_state[critic_key].shape[1]
        expected_in = env.num_privileged_obs if env.num_privileged_obs else env.num_obs
        if saved_in != expected_in:
            mismatch = expected_in - saved_in
            print(
                f"\n[COMPAT] Critic dim mismatch: saved={saved_in}, expected={expected_in}, diff={mismatch}"
            )

    try:
        runner.load(ckpt_path)
    except RuntimeError as e:
        if "size mismatch" in str(e).lower():
            print("[COMPAT] Partial load...")
            current = runner.alg.actor_critic.state_dict()
            loaded = 0
            for k, v in model_state.items():
                if k in current and current[k].shape == v.shape:
                    current[k] = v
                    loaded += 1
            runner.alg.actor_critic.load_state_dict(current, strict=False)
            print(f"[COMPAT] Loaded {loaded} tensors")
        else:
            raise
    return mismatch


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-stairwell-omni-v1")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Override terrain if needed
    if USE_FLAT_GROUND:
        if "terrain" in env_cfg:
            env_cfg["terrain"]["enabled"] = False
    elif STAIR_HEIGHT > 0.001 and "terrain" in env_cfg:
        env_cfg["terrain"]["step_height_max"] = STAIR_HEIGHT

    pls_enabled = env_cfg.get("pls_enable", False)
    if not pls_enabled:
        env_cfg["kp"] = KP * (MOTOR_STRENGTH_SCALE if MOTOR_STRENGTH_ENABLE else 1.0)
        env_cfg["kd"] = KD

    # Disable training DR
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
        "min_delay_steps",
        "max_delay_steps",
    ]:
        env_cfg.pop(key, None)

    if "curriculum" in env_cfg:
        env_cfg["curriculum"]["enabled"] = False

    env_cfg["termination_if_roll_greater_than"] = 1e9
    env_cfg["termination_if_pitch_greater_than"] = 1e9
    env_cfg["termination_if_z_vel_greater_than"] = 1e9
    env_cfg["termination_if_y_vel_greater_than"] = 1e9
    reward_cfg["reward_scales"] = {}

    # Create env
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    if env._use_terrain:
        env.ground.set_friction(GROUND_FRICTION)
    env.robot.set_friction(ROBOT_FRICTION)

    if MOTOR_STRENGTH_ENABLE and pls_enabled:
        env._motor_strength[:] = MOTOR_STRENGTH_SCALE

    # Load model
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    priv_mismatch = load_model_compat(runner, ckpt_path, env)
    policy = runner.get_inference_policy(device=gs.device)

    # Setup
    base_link_idx = env.robot.links[1].idx
    dt = env.dt
    push_interval_steps = int(PUSH_INTERVAL_S / dt)
    push_dur_range = [
        max(1, int(PUSH_DURATION_S[0] / dt)),
        max(1, int(PUSH_DURATION_S[1] / dt)),
    ]
    push_remaining = 0
    cached_push = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)

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

    # Reset & spawn
    obs, _ = env.reset()
    env._lock_terrain_rows = True
    respawn(env, SPAWN_WELL_LEVEL)
    obs, _ = env.get_observations()

    zero_action = torch.zeros((1, env.num_actions), device=gs.device)
    for _ in range(ACTION_DELAY_STEPS + 1):
        action_buffer.append(zero_action.clone())

    step = 0

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  STAIRWELL EVAL — {args.exp_name} @ ckpt {args.ckpt}")
    print(f"{'=' * 60}")
    if env._use_terrain:
        ti = env._terrain_info
        print(f"  Wells          : {ti['num_difficulty_levels']}")
        print(
            f"  Step heights   : {[f'{h * 100:.1f}cm' for h in ti['step_heights_m']]}"
        )
        print(f"  Spawn well     : {SPAWN_WELL_LEVEL}")
    else:
        print("  Terrain        : FLAT")
    print(f"  PLS            : {pls_enabled}")
    print(
        f"  Obs/Critic     : {obs_cfg['num_obs']}/{obs_cfg.get('num_privileged_obs', 'N/A')}"
    )
    if priv_mismatch:
        print(f"  Compat         : mismatch={priv_mismatch}")
    print(f"{'=' * 60}")
    print_controls()

    # Main loop
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()

    try:
        with torch.no_grad():
            while not _quit_event.is_set():
                step += 1

                with _state_lock:
                    if control_state["respawn_requested"]:
                        control_state["respawn_requested"] = False
                        respawn(env, SPAWN_WELL_LEVEL)
                        action_buffer.clear()
                        for _ in range(ACTION_DELAY_STEPS + 1):
                            action_buffer.append(zero_action.clone())
                        obs, _ = env.get_observations()

                env.commands[:] = make_command_tensor().to(gs.device)

                # Forces
                total_force = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)
                apply_force = False
                if (
                    PUSH_ENABLE
                    and step % push_interval_steps == 0
                    and push_remaining == 0
                ):
                    mag = rand_float(*PUSH_FORCE_RANGE, (1, 1)).to(gs.tc_float)
                    theta = rand_float(0.0, 2 * math.pi, (1, 1))
                    cached_push[0, 0] = torch.cos(theta) * mag
                    cached_push[0, 1] = torch.sin(theta) * mag
                    cached_push[0, 2] = PUSH_Z_SCALE * mag
                    push_remaining = int(
                        rand_int(push_dur_range[0], push_dur_range[1], (1,)).item()
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
                        force=total_force, links_idx=[base_link_idx], envs_idx=[0]
                    )

                # Obs processing
                policy_obs = obs.clone()
                if gravity_obs_offset is not None:
                    policy_obs[:, 3:6] += gravity_obs_offset
                if OBS_NOISE_ENABLE:
                    if obs_noise_vec is None:
                        obs_noise_vec = build_obs_noise_vec(
                            obs.shape[-1], OBS_NOISE_STD, OBS_NOISE_LEVEL
                        )
                    policy_obs += torch.randn_like(obs) * obs_noise_vec

                raw_actions = policy(policy_obs)

                if ACTION_NOISE_ENABLE:
                    noise = torch.zeros_like(raw_actions)
                    noise[:, : env.num_pos_actions] = (
                        torch.randn(1, env.num_pos_actions, device=gs.device)
                        * ACTION_NOISE_STD
                    )
                    raw_actions += noise

                if ACTION_DELAY_ENABLE:
                    action_buffer.append(raw_actions.clone())
                    actions = action_buffer[0]
                else:
                    actions = raw_actions

                if DEBUG_PRINT_INTERVAL > 0 and step % DEBUG_PRINT_INTERVAL == 0:
                    cmd = env.commands[0]
                    pos = env.base_pos[0]
                    vel = env.base_lin_vel[0]
                    parts = [
                        f"cmd=[{cmd[0].item():+.2f},{cmd[1].item():+.2f},{cmd[2].item():+.2f}]",
                        f"z={pos[2].item():.3f}m, vel=[{vel[0].item():.2f},{vel[1].item():.2f},{vel[2].item():.2f}]",
                    ]
                    if env._use_terrain and hasattr(env, "_get_terrain_height"):
                        tz = env._get_terrain_height(pos[0:1], pos[1:2]).item()
                        parts.append(f"hag={pos[2].item() - tz:.3f}m")
                    if pls_enabled:
                        sa = actions[0, env.num_pos_actions :]
                        kp = torch.clamp(
                            env.pls_kp_default + sa * env.pls_kp_action_scale,
                            env.pls_kp_range[0],
                            env.pls_kp_range[1],
                        )
                        parts.append(
                            f"Kp=[{kp[0]:.1f},{kp[1]:.1f},{kp[2]:.1f},{kp[3]:.1f}]"
                        )
                    print(
                        f"\r[step {step}] " + " | ".join(parts) + "    ",
                        end="",
                        flush=True,
                    )

                obs, _, _, _ = env.step(actions)

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    print("\n\nEvaluation ended.")


if __name__ == "__main__":
    main()
