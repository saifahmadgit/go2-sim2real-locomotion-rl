import argparse
import math
import os
import pickle
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

# ================= USER PARAM BLOCK =================

FIXED_COMMAND = torch.tensor([[1.0, 0.0, 0.0]])

KP = 60.0
KD = 2.0

GROUND_FRICTION = 0.7
ROBOT_FRICTION = 0.7

PUSH_ENABLE = True
PUSH_FORCE_RANGE = (0.0, 0.0)
PUSH_DURATION_S = 0.15
PUSH_INTERVAL_S = 2.0
PUSH_Z_SCALE = 0.0

# ====================================================


def rand_float(lower, upper, shape):
    return (upper - lower) * torch.rand(shape, device=gs.device) + lower


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

    # ---- PD ----
    env_cfg["kp"] = KP
    env_cfg["kd"] = KD

    # ---- Disable termination ----
    env_cfg["termination_if_roll_greater_than"] = 1e9
    env_cfg["termination_if_pitch_greater_than"] = 1e9
    env_cfg["termination_if_z_vel_greater_than"] = 1e9
    env_cfg["termination_if_y_vel_greater_than"] = 1e9

    reward_cfg["reward_scales"] = {}

    # -------- Create Environment --------
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # FRICTION SET HERE (NO ENV EDIT REQUIRED)

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
    print("  Effective μ:", max(GROUND_FRICTION, ROBOT_FRICTION))
    # =================================================

    # -------- Load Policy --------
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # -------- Push Setup --------
    base_link_idx = env.robot.links[1].idx

    push_steps = int(PUSH_DURATION_S / env.dt)
    push_interval_steps = int(PUSH_INTERVAL_S / env.dt)

    push_timer = 0
    cached_force = torch.zeros((1, 3), device=gs.device, dtype=gs.tc_float)

    # -------- Reset --------
    obs, _ = env.reset()
    target_cmd = FIXED_COMMAND.to(gs.device)

    step = 0

    print("\nRunning evaluation...\n")

    with torch.no_grad():
        while True:
            step += 1

            env.commands[:] = target_cmd

            if PUSH_ENABLE and step % push_interval_steps == 0 and push_timer == 0:
                mag = rand_float(*PUSH_FORCE_RANGE, (1, 1)).to(gs.tc_float)
                theta = rand_float(0.0, 2.0 * math.pi, (1, 1))

                force_dir = torch.zeros((1, 3), device=gs.device)
                force_dir[0, 0] = torch.cos(theta)
                force_dir[0, 1] = torch.sin(theta)
                force_dir[0, 2] = PUSH_Z_SCALE

                cached_force[:] = force_dir * mag
                push_timer = push_steps

                print("[Push]", cached_force[0].tolist())

            if PUSH_ENABLE and push_timer > 0:
                env.scene.sim.rigid_solver.apply_links_external_force(
                    force=cached_force,
                    links_idx=[base_link_idx],
                    envs_idx=[0],
                )
                push_timer -= 1

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
