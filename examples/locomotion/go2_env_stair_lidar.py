"""
go2_env_stair4_lidar.py
========================
Go2 stair-climbing RL environment with SIMULATED LIDAR for both actor & critic.

Sim-to-Real LIDAR Design (Minimal 5×3 Grid)
----------------------------------------------
The Unitree Go2's L1 LIDAR (~21,600 pts/s) is projected into a compact
body-frame height grid for the RL policy:

1. **Actor terrain scan** (5×3 = 15 points, deployed on robot):
   - 5 forward-facing rows at x = [0.0, 0.15, 0.30, 0.50, 0.80] m
   - 3 lateral columns at y = [-0.15, 0.0, 0.15] m
   - Non-uniform x-spacing: denser near feet, sparser far ahead
   - 15cm cell size → ~8-10 LIDAR points per cell (reliable estimates)
   - Designed to match real L1 sensor physics constraints

2. **Critic height scan** (dense surround, privileged, training only)

Sim-to-Real Robustness:
- 2cm Gaussian noise (matches L1 accuracy spec)
- 15% per-cell dropout (sparse scan regions / occlusions)
- 5% full-scan blackout (forces proprioceptive fallback)
- 10% latency (stale data from previous step)
- 1cm vertical offset (mounting vibration)

The aggressive noise model forces the policy to treat terrain info as a
helpful hint rather than ground truth, falling back to proprioception
(gravity vector, joint positions, foot contacts) when data is unreliable.
"""

import math
import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


# ======================================================================
# Utility functions (unchanged from base)
# ======================================================================

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_rand_int(lower, upper, shape, device):
    return torch.randint(lower, upper + 1, size=shape, device=device)


def euler_to_quat_wxyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _lerp(a: float, b: float, t: float) -> float:
    t = _clamp01(t)
    return a + (b - a) * t


def _lerp_range(a, b, t: float):
    return [_lerp(a[0], b[0], t), _lerp(a[1], b[1], t)]


# ======================================================================
# Terrain heightfield generation (unchanged from base)
# ======================================================================

def build_stair_terrain(terrain_cfg: dict) -> tuple:
    h_scale = terrain_cfg.get("horizontal_scale", 0.05)
    v_scale = terrain_cfg.get("vertical_scale", 0.005)

    num_rows = terrain_cfg.get("num_difficulty_rows", 10)
    row_width_m = terrain_cfg.get("row_width_m", 6.0)

    step_depth_m = terrain_cfg.get("step_depth_m", 0.30)
    num_steps = terrain_cfg.get("num_steps", 6)
    num_flights = terrain_cfg.get("num_flights", 4)
    flat_before_m = terrain_cfg.get("flat_before_m", 2.0)
    flat_top_m = terrain_cfg.get("flat_top_m", 1.5)
    flat_gap_m = terrain_cfg.get("flat_gap_m", 1.5)
    flat_after_m = terrain_cfg.get("flat_after_m", 2.0)

    step_height_min = terrain_cfg.get("step_height_min", 0.02)
    step_height_max = terrain_cfg.get("step_height_max", 0.15)

    step_depth_cells = max(1, int(round(step_depth_m / h_scale)))
    flat_before_cells = max(1, int(round(flat_before_m / h_scale)))
    flat_top_cells = max(1, int(round(flat_top_m / h_scale)))
    flat_gap_cells = max(1, int(round(flat_gap_m / h_scale)))
    flat_after_cells = max(1, int(round(flat_after_m / h_scale)))
    stair_section_cells = num_steps * step_depth_cells
    row_width_cells = max(1, int(round(row_width_m / h_scale)))

    flight_cells = stair_section_cells + flat_top_cells + stair_section_cells + flat_gap_cells
    total_x_cells = flat_before_cells + num_flights * flight_cells + flat_after_cells
    total_y_cells = num_rows * row_width_cells

    step_heights_m = np.linspace(step_height_min, step_height_max, num_rows)
    step_heights_units = np.round(step_heights_m / v_scale).astype(np.int16)

    hf = np.zeros((total_x_cells, total_y_cells), dtype=np.int16)

    row_centers = []

    for row_idx in range(num_rows):
        y_start = row_idx * row_width_cells
        y_end = (row_idx + 1) * row_width_cells
        sh = step_heights_units[row_idx]

        x = 0
        x += flat_before_cells

        for _flight in range(num_flights):
            for s in range(num_steps):
                height = (s + 1) * sh
                hf[x:x + step_depth_cells, y_start:y_end] = height
                x += step_depth_cells

            top_height = num_steps * sh
            hf[x:x + flat_top_cells, y_start:y_end] = top_height
            x += flat_top_cells

            for s in range(num_steps):
                height = top_height - (s + 1) * sh
                hf[x:x + step_depth_cells, y_start:y_end] = height
                x += step_depth_cells

            x += flat_gap_cells

        spawn_x_m = flat_before_cells * h_scale * 0.5
        spawn_y_m = (y_start + row_width_cells / 2.0) * h_scale
        row_centers.append((spawn_x_m, spawn_y_m, 0.0))

    total_x_m = total_x_cells * h_scale
    total_y_m = total_y_cells * h_scale

    terrain_origin = (0.0, -total_y_m / 2.0, 0.0)

    adjusted_centers = []
    for cx, cy, cz in row_centers:
        adjusted_centers.append((
            cx + terrain_origin[0],
            cy + terrain_origin[1],
            cz + terrain_origin[2],
        ))

    stair_m_per_flight = 2 * stair_section_cells * h_scale
    total_stair_m = stair_m_per_flight * num_flights
    stair_pct = total_stair_m / total_x_m * 100

    terrain_info = {
        "heightfield": hf,
        "horizontal_scale": h_scale,
        "vertical_scale": v_scale,
        "terrain_origin": terrain_origin,
        "num_difficulty_rows": num_rows,
        "row_centers": adjusted_centers,
        "step_heights_m": step_heights_m.tolist(),
        "total_x_m": total_x_m,
        "total_y_m": total_y_m,
        "num_flights": num_flights,
        "num_steps": num_steps,
        "step_depth_m": step_depth_m,
    }

    print(f"\n[Terrain] Built stair terrain heightfield:")
    print(f"  Shape         : {hf.shape} cells  ({total_x_m:.1f}m × {total_y_m:.1f}m)")
    print(f"  Difficulty rows: {num_rows}")
    print(f"  Flights/row   : {num_flights} (up-down cycles)")
    print(f"  Steps/flight  : {num_steps} × {step_depth_m:.2f}m tread")
    print(f"  Step heights  : {[f'{h:.3f}m' for h in step_heights_m]}")
    print(f"  Row length    : {total_x_m:.1f}m  ({stair_pct:.0f}% stairs / {100-stair_pct:.0f}% flat)")
    print(f"  Terrain origin: {terrain_origin}")
    for i, (cx, cy, cz) in enumerate(adjusted_centers):
        print(f"  Row {i} (step={step_heights_m[i]*100:.1f}cm): spawn=({cx:.2f}, {cy:.2f}, {cz:.2f})")
    print()

    return hf, terrain_info


# ======================================================================
# Curriculum Manager (unchanged from base)
# ======================================================================

class CurriculumManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))

        self.level = float(self.cfg.get("level_init", 0.0))
        self.level_min = float(self.cfg.get("level_min", 0.0))
        self.level_max = float(self.cfg.get("level_max", 1.0))

        self.ema_alpha = float(self.cfg.get("ema_alpha", 0.05))

        self.ready_timeout_rate = float(self.cfg.get("ready_timeout_rate", 0.7))
        self.ready_tracking = float(self.cfg.get("ready_tracking", 0.6))
        self.ready_fall_rate = float(self.cfg.get("ready_fall_rate", 0.30))
        self.ready_streak_needed = int(self.cfg.get("ready_streak", 3))

        self.hard_fall_rate = float(self.cfg.get("hard_fall_rate", 0.55))
        self.hard_streak_needed = int(self.cfg.get("hard_streak", 2))

        self.step_up = float(self.cfg.get("step_up", 0.02))
        self.step_down = float(self.cfg.get("step_down", 0.01))
        self.cooldown_updates = int(self.cfg.get("cooldown_updates", 1))
        self._cooldown = 0

        self.mix_prob_current = float(self.cfg.get("mix_prob_current", 0.80))
        self.mix_level_low = float(self.cfg.get("mix_level_low", 0.0))
        self.mix_level_high = float(self.cfg.get("mix_level_high", 0.6))

        self.timeout_rate_ema = None
        self.fall_rate_ema = None
        self.tracking_ema = None

        self._ready_streak = 0
        self._hard_streak = 0

    def sample_level(self) -> float:
        if not self.enabled:
            return 1.0
        if torch.rand(()) < self.mix_prob_current:
            return _clamp01(self.level)
        hi = min(self.level, self.mix_level_high)
        lo = min(self.mix_level_low, hi)
        t = float(lo + (hi - lo) * torch.rand(()).item())
        return _clamp01(t)

    def _ema_update(self, old, x):
        if old is None:
            return float(x)
        a = self.ema_alpha
        return float((1.0 - a) * old + a * float(x))

    def update(self, timeout_rate: float, tracking_per_sec: float, fall_rate: float) -> bool:
        if not self.enabled:
            return False

        self.timeout_rate_ema = self._ema_update(self.timeout_rate_ema, timeout_rate)
        self.tracking_ema = self._ema_update(self.tracking_ema, tracking_per_sec)
        self.fall_rate_ema = self._ema_update(self.fall_rate_ema, fall_rate)

        if self._cooldown > 0:
            self._cooldown -= 1

        ready = (
            self.timeout_rate_ema >= self.ready_timeout_rate
            and self.tracking_ema >= self.ready_tracking
            and self.fall_rate_ema <= self.ready_fall_rate
        )
        hard = (self.fall_rate_ema >= self.hard_fall_rate)

        if ready:
            self._ready_streak += 1
        else:
            self._ready_streak = 0

        if hard:
            self._hard_streak += 1
        else:
            self._hard_streak = 0

        old_level = self.level

        if self._hard_streak >= self.hard_streak_needed:
            self.level = max(self.level_min, self.level - self.step_down)
            self._hard_streak = 0
            self._ready_streak = 0
            self._cooldown = self.cooldown_updates
        elif (self._ready_streak >= self.ready_streak_needed) and (self._cooldown == 0):
            self.level = min(self.level_max, self.level + self.step_up)
            self._ready_streak = 0
            self._cooldown = self.cooldown_updates

        self.level = _clamp01(self.level)
        return (self.level != old_level)

    def state_dict(self):
        return {
            "enabled": self.enabled,
            "level": float(self.level),
            "timeout_rate_ema": None if self.timeout_rate_ema is None else float(self.timeout_rate_ema),
            "tracking_ema": None if self.tracking_ema is None else float(self.tracking_ema),
            "fall_rate_ema": None if self.fall_rate_ema is None else float(self.fall_rate_ema),
        }


# ======================================================================
# LIDAR Scan Configuration
# ======================================================================

class LidarScanConfig:
    """
    Configuration for simulated LIDAR terrain scan.

    Designed to match what you'd extract from the Go2's L1/L2 LIDAR
    on the real robot. The L1 has 360°×90° FOV with ~21,600 pts/s.

    For RL locomotion, we don't need the full point cloud — we project
    it into a compact terrain profile that the policy can digest.

    Two scan types:
    1. actor_scan:  Forward-biased sparse grid (deployed on robot)
    2. critic_scan: Dense surround grid (privileged, training only)

    Real-robot pipeline:
      L1 point cloud → voxel filter → project to body-frame grid →
      compute height per cell → subtract base z → clip → scale

    This class pre-computes the body-frame sample points so the env
    just does fast heightfield lookups at runtime.
    """

    def __init__(self, cfg: dict, device):
        self.device = device
        self.enabled = bool(cfg.get("enabled", True))

        if not self.enabled:
            self.num_actor_scan = 0
            self.num_critic_scan = 0
            return

        # ---- Actor scan: minimal forward-biased grid ----
        # 5×3 = 15 points optimized for stair climbing + sim-to-real.
        # Larger cells (15cm) → more LIDAR points per cell → more reliable.
        # Supports EITHER explicit x_points/y_points lists OR num_x/num_y + range.
        actor_cfg = cfg.get("actor_scan", {})

        if "x_points" in actor_cfg:
            # Explicit sample positions (preferred for non-uniform spacing)
            ax = torch.tensor(actor_cfg["x_points"], dtype=torch.float32, device=device)
            ay = torch.tensor(actor_cfg["y_points"], dtype=torch.float32, device=device)
            self._actor_nx = len(actor_cfg["x_points"])
            self._actor_ny = len(actor_cfg["y_points"])
        else:
            # Uniform linspace fallback
            self._actor_nx = int(actor_cfg.get("num_x", 5))
            self._actor_ny = int(actor_cfg.get("num_y", 3))
            actor_x_range = actor_cfg.get("x_range", [0.0, 0.8])
            actor_y_range = actor_cfg.get("y_range", [-0.15, 0.15])
            ax = torch.linspace(float(actor_x_range[0]), float(actor_x_range[1]),
                                self._actor_nx, device=device)
            ay = torch.linspace(float(actor_y_range[0]), float(actor_y_range[1]),
                                self._actor_ny, device=device)

        self.num_actor_scan = self._actor_nx * self._actor_ny     # 15 points default

        agx, agy = torch.meshgrid(ax, ay, indexing="ij")
        self.actor_local_x = agx.reshape(-1)   # (num_actor_scan,)
        self.actor_local_y = agy.reshape(-1)   # (num_actor_scan,)

        # ---- Critic scan: dense surround grid (privileged) ----
        critic_cfg = cfg.get("critic_scan", {})
        self._critic_nx = int(critic_cfg.get("num_x", 9))
        self._critic_ny = int(critic_cfg.get("num_y", 5))
        critic_x_range = critic_cfg.get("x_range", [-0.4, 0.8])
        critic_y_range = critic_cfg.get("y_range", [-0.3, 0.3])

        self.num_critic_scan = self._critic_nx * self._critic_ny  # 45 points default

        cx = torch.linspace(float(critic_x_range[0]), float(critic_x_range[1]),
                            self._critic_nx, device=device)
        cy = torch.linspace(float(critic_y_range[0]), float(critic_y_range[1]),
                            self._critic_ny, device=device)
        cgx, cgy = torch.meshgrid(cx, cy, indexing="ij")
        self.critic_local_x = cgx.reshape(-1)
        self.critic_local_y = cgy.reshape(-1)

        # ---- Noise / DR config (AGGRESSIVE for sim-to-real gap) ----
        noise_cfg = cfg.get("noise", {})
        self.height_noise_std = float(noise_cfg.get("height_noise_std", 0.02))      # 2cm noise (L1 accuracy ±2cm)
        self.dropout_prob = float(noise_cfg.get("dropout_prob", 0.15))               # 15% per-cell dropout
        self.full_blackout_prob = float(noise_cfg.get("full_blackout_prob", 0.05))   # 5% entire scan goes dead
        self.latency_prob = float(noise_cfg.get("latency_prob", 0.1))                # 10% chance of stale data
        self.vertical_offset_std = float(noise_cfg.get("vertical_offset_std", 0.01)) # Mounting vibration

        # ---- Clipping ----
        self.height_clip = float(cfg.get("height_clip", 1.0))   # Clip heights to ±1.0m
        self.height_scale = float(cfg.get("height_scale", 1.0))  # Scale factor for obs

        print(f"\n[LidarScan] Configuration:")
        print(f"  Actor scan  : {self._actor_nx}×{self._actor_ny} = {self.num_actor_scan} points")
        print(f"    x_points  : {ax.tolist()}")
        print(f"    y_points  : {ay.tolist()}")
        print(f"  Critic scan : {self._critic_nx}×{self._critic_ny} = {self.num_critic_scan} points")
        print(f"    x_range   : {critic_x_range}m")
        print(f"    y_range   : {critic_y_range}m")
        print(f"  Noise (sim2real):")
        print(f"    height_std     : {self.height_noise_std}m")
        print(f"    cell dropout   : {self.dropout_prob}")
        print(f"    full blackout  : {self.full_blackout_prob}")
        print(f"    latency        : {self.latency_prob}")
        print(f"  Height clip : ±{self.height_clip}m, scale={self.height_scale}")
        print()


# ======================================================================
# Go2Env with LIDAR
# ======================================================================

class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg.get("num_privileged_obs", None)
        self.num_actions = env_cfg["num_actions"]
        self.num_pos_actions = env_cfg.get("num_pos_actions", 12)
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ----------------------------------------------------------
        # PLS config
        # ----------------------------------------------------------
        self.pls_enable = bool(env_cfg.get("pls_enable", False))
        self.num_stiffness_actions = 4 if self.pls_enable else 0

        if self.pls_enable:
            self.pls_kp_range = env_cfg["pls_kp_range"]
            self.pls_kp_default = float(env_cfg["pls_kp_default"])
            self.pls_kp_action_scale = float(env_cfg["pls_kp_action_scale"])
            self.leg_joint_map = torch.tensor(
                [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                device=gs.device, dtype=torch.long,
            )

        # ----------------------------------------------------------
        # Torque limits
        # ----------------------------------------------------------
        torque_lim_cfg = env_cfg.get("torque_limits", None)
        if torque_lim_cfg is not None:
            self._torque_limits = torch.tensor(
                torque_lim_cfg, device=gs.device, dtype=gs.tc_float,
            )
        else:
            self._torque_limits = torch.full(
                (self.num_pos_actions,), 23.7,
                device=gs.device, dtype=gs.tc_float,
            )

        # ----------------------------------------------------------
        # LIDAR scan setup  ← NEW
        # ----------------------------------------------------------
        lidar_cfg = env_cfg.get("lidar", {})
        self._lidar = LidarScanConfig(lidar_cfg, gs.device)

        # ----------------------------------------------------------
        # Terrain setup
        # ----------------------------------------------------------
        self._terrain_cfg = env_cfg.get("terrain", None)
        self._use_terrain = self._terrain_cfg is not None and self._terrain_cfg.get("enabled", False)

        if self._use_terrain:
            self._hf, self._terrain_info = build_stair_terrain(self._terrain_cfg)
            self._num_terrain_rows = self._terrain_info["num_difficulty_rows"]
            self._terrain_row_centers = self._terrain_info["row_centers"]
            self._terrain_step_heights = self._terrain_info["step_heights_m"]

            self._terrain_origin = self._terrain_info["terrain_origin"]
            self._h_scale = self._terrain_info["horizontal_scale"]
            self._v_scale = self._terrain_info["vertical_scale"]
            self._hf_heights = torch.tensor(
                self._hf, device=gs.device, dtype=gs.tc_float,
            ) * self._v_scale

            # Legacy height scan config (kept for backward compat, but now
            # the LidarScanConfig handles both actor and critic scans)
            hs_cfg = self._terrain_cfg.get("height_scan", {})
            self._scan_nx = int(hs_cfg.get("num_x", 11))
            self._scan_ny = int(hs_cfg.get("num_y", 7))
            self._scan_n = self._scan_nx * self._scan_ny
            x_range = hs_cfg.get("x_range", [-0.5, 0.5])
            y_range = hs_cfg.get("y_range", [-0.3, 0.3])
            scan_x = torch.linspace(float(x_range[0]), float(x_range[1]),
                                    self._scan_nx, device=gs.device)
            scan_y = torch.linspace(float(y_range[0]), float(y_range[1]),
                                    self._scan_ny, device=gs.device)
            gx, gy = torch.meshgrid(scan_x, scan_y, indexing="ij")
            self._scan_local_x = gx.reshape(-1)
            self._scan_local_y = gy.reshape(-1)
        else:
            self._num_terrain_rows = 1
            self._terrain_row_centers = [(0.0, 0.0, 0.0)]
            self._scan_n = 0

        self._env_terrain_row = torch.zeros(num_envs, device=gs.device, dtype=torch.long)
        self._lock_terrain_rows = False

        # ----------------------------------------------------------
        # Scene
        # ----------------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
        )

        if self._use_terrain:
            origin = self._terrain_info["terrain_origin"]
            self.ground = self.scene.add_entity(
                gs.morphs.Terrain(
                    height_field=self._hf,
                    horizontal_scale=self._terrain_info["horizontal_scale"],
                    vertical_scale=self._terrain_info["vertical_scale"],
                    pos=origin,
                ),
            )
        else:
            self.ground = self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
            )

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_pos_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_pos_actions, self.motors_dof_idx)

        # ----------------------------------------------------------
        # Rewards
        # ----------------------------------------------------------
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            fn_name = "_reward_" + name
            if not hasattr(self, fn_name):
                raise AttributeError(
                    f"Reward function '{fn_name}' not found in Go2Env. "
                    f"You have reward_scales['{name}'] but no method '{fn_name}'. "
                    f"Either implement it or remove '{name}' from reward_scales."
                )
            self.reward_functions[name] = getattr(self, fn_name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # ----------------------------------------------------------
        # Foot contact tracking
        # ----------------------------------------------------------
        self._foot_reward_names = ["feet_air_time", "foot_slip", "foot_clearance", "feet_stance"]
        self._has_foot_contact_rewards = any(
            name in self.reward_scales and self.reward_scales[name] != 0.0
            for name in self._foot_reward_names
        )
        if self._has_foot_contact_rewards:
            self._init_foot_contact_tracking()

        # ----------------------------------------------------------
        # Per-leg (hip) mass DR
        # ----------------------------------------------------------
        self._has_leg_mass_dr = "leg_mass_shift_range" in self.env_cfg
        self._hip_link_indices = []
        if self._has_leg_mass_dr:
            hip_joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
            _link_name_map = {}
            for i, link in enumerate(self.robot.links):
                name = getattr(link, "name", None)
                if name is not None:
                    _link_name_map[name] = i
            if not _link_name_map and hasattr(self.robot, "link_names"):
                for i, name in enumerate(self.robot.link_names):
                    _link_name_map[name] = i

            hip_link_patterns = [
                ["FR_hip", "FL_hip", "RR_hip", "RL_hip"],
                ["FR_HIP", "FL_HIP", "RR_HIP", "RL_HIP"],
                ["FR_hip_link", "FL_hip_link", "RR_hip_link", "RL_hip_link"],
            ]
            found = False
            for pattern in hip_link_patterns:
                if all(p in _link_name_map for p in pattern):
                    self._hip_link_indices = [_link_name_map[p] for p in pattern]
                    found = True
                    break

            if not found:
                for jname in hip_joint_names:
                    try:
                        joint = self.robot.get_joint(jname)
                        link_idx = getattr(joint, "child_link_idx", None)
                        if link_idx is None:
                            link_idx = getattr(joint, "link_idx", None)
                        if link_idx is not None:
                            self._hip_link_indices.append(int(link_idx))
                    except Exception:
                        pass

            if len(self._hip_link_indices) == 4:
                print(f"[Go2Env] Per-leg mass DR ENABLED: hip_link_indices={self._hip_link_indices}")
            else:
                self._has_leg_mass_dr = False

        # ----------------------------------------------------------
        # DR base config
        # ----------------------------------------------------------
        self.obs_noise_components = self.env_cfg.get("obs_noise", None)
        self.obs_noise_level_max = float(self.env_cfg.get("obs_noise_level", 0.0)) if self.obs_noise_components else 0.0
        self.action_noise_std_max = float(self.env_cfg.get("action_noise_std", 0.0)) if ("action_noise_std" in self.env_cfg) else 0.0

        self.push_interval_s_hard = float(self.env_cfg.get("push_interval_s", 5.0))
        self.push_force_range_hard = self.env_cfg.get("push_force_range", None)

        self._push_counter = 0
        if self.push_force_range_hard is not None:
            self._all_envs_idx = torch.arange(self.num_envs, device=gs.device, dtype=gs.tc_int)
            try:
                self._base_link_idx = int(self.robot.links[1].idx)
            except Exception:
                self._base_link_idx = int(self.robot.links[0].idx)

            dur_cfg = self.env_cfg.get("push_duration_s", [0.05, 0.15])
            self._push_duration_steps = [
                max(1, int(dur_cfg[0] / self.dt)),
                max(1, int(dur_cfg[1] / self.dt)),
            ]
            self._push_stored_force = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
            self._push_remaining = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)

        # ----------------------------------------------------------
        # DR level decoupling
        # ----------------------------------------------------------
        dr_cfg = self.env_cfg.get("dr_schedule", None)
        if dr_cfg is not None:
            self._dr_phase1_level = float(dr_cfg.get("phase1_level", 0.15))
            self._dr_terrain_gate = float(dr_cfg.get("terrain_gate", 0.85))
            self._dr_enabled = True
        else:
            self._dr_enabled = False

        # ----------------------------------------------------------
        # Command config
        # ----------------------------------------------------------
        self._cmd_curriculum = bool(command_cfg.get("cmd_curriculum", False))
        self._cmd_start_frac = float(command_cfg.get("cmd_curriculum_start_frac", 0.1))
        self._rel_standing_envs = float(command_cfg.get("rel_standing_envs", 0.0))
        self._compound_commands = bool(command_cfg.get("compound_commands", True))

        self._cmd_full_ranges = {
            "lin_vel_x": list(command_cfg["lin_vel_x_range"]),
            "lin_vel_y": list(command_cfg["lin_vel_y_range"]),
            "ang_vel": list(command_cfg["ang_vel_range"]),
        }
        self._cmd_cur_ranges = {k: list(v) for k, v in self._cmd_full_ranges.items()}

        n_standing = int(self._rel_standing_envs * self.num_envs)
        self._standing_env_indices = torch.arange(n_standing, device=gs.device) if n_standing > 0 else None

        # ----------------------------------------------------------
        # Action delay buffer
        # ----------------------------------------------------------
        self._min_delay_steps = int(self.env_cfg.get("min_delay_steps", 0))
        self._max_delay_steps = int(self.env_cfg.get("max_delay_steps", 2))
        self._delay_buf_size = self._max_delay_steps + 1
        self._action_history = torch.zeros(
            (self.num_envs, self._delay_buf_size, self.num_actions),
            device=gs.device, dtype=gs.tc_float,
        )
        self._action_history_write_idx = 0
        self._delay_steps = torch.ones((self.num_envs,), device=gs.device, dtype=torch.long)

        self._applied_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float,
        )

        # ----------------------------------------------------------
        # Per-env DR buffers
        # ----------------------------------------------------------
        self._kp_factors = torch.ones((self.num_envs, self.num_pos_actions), device=gs.device, dtype=gs.tc_float)
        self._kd_factors = torch.ones((self.num_envs, self.num_pos_actions), device=gs.device, dtype=gs.tc_float)
        self._motor_strength = torch.ones((self.num_envs, self.num_pos_actions), device=gs.device, dtype=gs.tc_float)
        self._gravity_offset = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self._current_push_force = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self._current_friction = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self._current_mass_shift = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self._current_com_shift = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self._current_leg_mass_shifts = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self._effective_kp = torch.full(
            (self.num_envs, self.num_pos_actions), float(self.env_cfg["kp"]),
            device=gs.device, dtype=gs.tc_float,
        )
        self._effective_kd = torch.full(
            (self.num_envs, self.num_pos_actions), float(self.env_cfg["kd"]),
            device=gs.device, dtype=gs.tc_float,
        )

        self._target_dof_pos = torch.zeros(
            (self.num_envs, self.num_pos_actions), device=gs.device, dtype=gs.tc_float,
        )

        # ----------------------------------------------------------
        # Main buffers
        # ----------------------------------------------------------
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.num_privileged_obs if self.num_privileged_obs else self.num_obs),
            device=gs.device, dtype=gs.tc_float,
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device, dtype=gs.tc_float,
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.dof_pos = torch.zeros((self.num_envs, self.num_pos_actions), device=gs.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_pos)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device, dtype=gs.tc_float,
        )

        self._last_base_pos_x = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

        # ----------------------------------------------------------
        # LIDAR scan buffers  ← NEW
        # ----------------------------------------------------------
        if self._lidar.enabled:
            # Actor scan buffer (included in actor obs)
            self._actor_scan_buf = torch.zeros(
                (self.num_envs, self._lidar.num_actor_scan),
                device=gs.device, dtype=gs.tc_float,
            )
            # Previous step's actor scan (for latency simulation)
            self._actor_scan_prev = torch.zeros_like(self._actor_scan_buf)

            # Critic scan buffer (included in privileged obs)
            self._critic_scan_buf = torch.zeros(
                (self.num_envs, self._lidar.num_critic_scan),
                device=gs.device, dtype=gs.tc_float,
            )
            print(f"[Go2Env] LIDAR scan buffers allocated:")
            print(f"  Actor scan  : ({self.num_envs}, {self._lidar.num_actor_scan})")
            print(f"  Critic scan : ({self.num_envs}, {self._lidar.num_critic_scan})")
        else:
            self._actor_scan_buf = torch.zeros(
                (self.num_envs, 0), device=gs.device, dtype=gs.tc_float)
            self._actor_scan_prev = torch.zeros_like(self._actor_scan_buf)
            self._critic_scan_buf = torch.zeros(
                (self.num_envs, 0), device=gs.device, dtype=gs.tc_float)

        self.extras = {"observations": {}}

        # ----------------------------------------------------------
        # Curriculum
        # ----------------------------------------------------------
        self.curr_cfg = self.env_cfg.get("curriculum", {}) or {}
        self.curriculum = CurriculumManager(self.curr_cfg)

        self._curr_ep_total = 0
        self._curr_timeout_total = 0.0
        self._curr_tracking_sum = 0.0
        self._curr_tracking_n = 0

        self.curr_update_every_episodes = int(self.curr_cfg.get("update_every_episodes", 2048))

        self.friction_easy = self.curr_cfg.get("friction_easy", [0.6, 0.9])
        self.friction_hard = self.env_cfg.get("friction_range", self.friction_easy)

        kp_nom = float(self.env_cfg.get("kp", 60.0))
        kd_nom = float(self.env_cfg.get("kd", 2.0))
        self.kp_easy = self.curr_cfg.get("kp_easy", [0.9 * kp_nom, 1.1 * kp_nom])
        self.kd_easy = self.curr_cfg.get("kd_easy", [0.75 * kd_nom, 1.25 * kd_nom])
        self.kp_hard = self.env_cfg.get("kp_range", self.kp_easy)
        self.kd_hard = self.env_cfg.get("kd_range", self.kd_easy)

        self.kp_factor_easy = self.curr_cfg.get("kp_factor_easy", [0.95, 1.05])
        self.kp_factor_hard = self.env_cfg.get("kp_factor_range", self.kp_factor_easy)
        self.kd_factor_easy = self.curr_cfg.get("kd_factor_easy", [0.85, 1.15])
        self.kd_factor_hard = self.env_cfg.get("kd_factor_range", self.kd_factor_easy)

        self.mass_easy = self.curr_cfg.get("mass_shift_easy", [-0.2, 0.5])
        self.mass_hard = self.env_cfg.get("mass_shift_range", self.mass_easy)
        self.com_easy = self.curr_cfg.get("com_shift_easy", [-0.005, 0.005])
        self.com_hard = self.env_cfg.get("com_shift_range", self.com_easy)

        self.gravity_offset_easy = self.curr_cfg.get("gravity_offset_easy", [-0.2, 0.2])
        self.gravity_offset_hard = self.env_cfg.get("gravity_offset_range", self.gravity_offset_easy)

        self.motor_strength_easy = self.curr_cfg.get("motor_strength_easy", [0.97, 1.03])
        self.motor_strength_hard = self.env_cfg.get("motor_strength_range", self.motor_strength_easy)

        self.leg_mass_easy = self.curr_cfg.get("leg_mass_shift_easy", [-0.1, 0.1])
        self.leg_mass_hard = self.env_cfg.get("leg_mass_shift_range", self.leg_mass_easy)

        self.delay_easy_max = int(self.curr_cfg.get("delay_easy_max_steps", 1))

        self.push_start = float(self.curr_cfg.get("push_start", 0.30))
        self.push_interval_s_easy = float(self.curr_cfg.get("push_interval_easy_s", 6.0))

        self._global_dr_update_interval = int(self.curr_cfg.get("global_dr_update_interval", 200))
        self._global_dr_reset_counter = 0

        self._obs_noise_level_cur = 0.0
        self._action_noise_std_cur = 0.0
        self._push_enable_cur = False
        self._push_interval = int(self.push_interval_s_easy / self.dt) if self.push_force_range_hard is not None else 1
        self._push_force_range_cur = [0.0, 0.0]
        self._delay_max_cur = self.delay_easy_max

        # LIDAR noise level (scales with DR level)
        self._lidar_noise_level = 0.0

        self.obs_noise_vec = None
        self._apply_curriculum_level(force=True)

        # ----------------------------------------------------------
        # Manual PD detection
        # ----------------------------------------------------------
        self._use_manual_pd = self.pls_enable or ("kp_factor_range" in self.env_cfg)
        if self._use_manual_pd:
            self.robot.set_dofs_kp([0.0] * self.num_pos_actions, self.motors_dof_idx)
            self.robot.set_dofs_kv([0.0] * self.num_pos_actions, self.motors_dof_idx)
            print(f"[Go2Env] Manual PD torque mode ENABLED")

    # ==========================================================
    # Terrain: height lookup
    # ==========================================================

    def _get_terrain_height(self, x, y):
        if not self._use_terrain:
            return torch.zeros_like(x)
        ox, oy, _ = self._terrain_origin
        col = ((x - ox) / self._h_scale).long().clamp(0, self._hf_heights.shape[0] - 1)
        row = ((y - oy) / self._h_scale).long().clamp(0, self._hf_heights.shape[1] - 1)
        return self._hf_heights[col, row]

    # ==========================================================
    # LIDAR: scan computation  ← NEW
    # ==========================================================

    def _compute_lidar_scan(self, local_x, local_y, num_points):
        """
        Compute body-frame terrain height scan.

        For each robot, rotates local sample points by robot yaw,
        translates to world frame, looks up terrain height, and
        returns heights relative to the robot base.

        This is the core of the simulated LIDAR — it produces the
        same data you'd compute from the real L1/L2 point cloud.

        Args:
            local_x: (num_points,) body-frame x coordinates of sample points
            local_y: (num_points,) body-frame y coordinates
            num_points: number of scan points

        Returns:
            (num_envs, num_points) terrain height relative to robot base
        """
        if not self._use_terrain or num_points == 0:
            return torch.zeros(self.num_envs, max(num_points, 1),
                               device=self.device, dtype=gs.tc_float)

        # Extract yaw from quaternion (w, x, y, z)
        qw = self.base_quat[:, 0]
        qx = self.base_quat[:, 1]
        qy = self.base_quat[:, 2]
        qz = self.base_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
        cos_y = torch.cos(yaw).unsqueeze(1)   # (N, 1)
        sin_y = torch.sin(yaw).unsqueeze(1)   # (N, 1)

        # Rotate local points by yaw and translate to world frame
        lx = local_x.unsqueeze(0)  # (1, num_points)
        ly = local_y.unsqueeze(0)  # (1, num_points)
        wx = self.base_pos[:, 0:1] + cos_y * lx - sin_y * ly   # (N, num_points)
        wy = self.base_pos[:, 1:2] + sin_y * lx + cos_y * ly   # (N, num_points)

        terrain_h = self._get_terrain_height(wx, wy)             # (N, num_points)
        base_z = self.base_pos[:, 2:3]                            # (N, 1)

        return terrain_h - base_z   # Heights relative to robot base

    def _update_actor_lidar_scan(self):
        """
        Compute actor LIDAR scan with aggressive sim-to-real domain randomization.

        This produces the observation that the policy will receive both
        in simulation AND on the real robot. The noise model is deliberately
        aggressive to prevent the policy from overfitting to simulation's
        perfect heightfield lookups:

        - ±2cm measurement accuracy (Gaussian noise)
        - 15% per-cell dropout (sparse regions / occlusions)
        - 5% full-scan blackout (entire scan dead → proprioception only)
        - Processing latency (stale data from previous step)
        - Mounting vibration (global vertical offset per env)

        The policy must maintain a stable gait from proprioception alone
        and treat the terrain scan as a helpful hint, not a requirement.
        """
        if not self._lidar.enabled or self._lidar.num_actor_scan == 0:
            return

        # Save previous scan for latency simulation
        self._actor_scan_prev[:] = self._actor_scan_buf[:]

        # Fresh scan
        raw_scan = self._compute_lidar_scan(
            self._lidar.actor_local_x,
            self._lidar.actor_local_y,
            self._lidar.num_actor_scan,
        )

        # --- Domain Randomization (scales with curriculum / DR level) ---
        noise_scale = self._lidar_noise_level

        # 1. Gaussian height noise (L1 accuracy is ±2cm)
        if self._lidar.height_noise_std > 0 and noise_scale > 0:
            noise = torch.randn_like(raw_scan) * self._lidar.height_noise_std * noise_scale
            raw_scan = raw_scan + noise

        # 2. Random point dropout (mimics occlusion, sparse scan regions)
        if self._lidar.dropout_prob > 0 and noise_scale > 0:
            dropout_mask = torch.rand_like(raw_scan) < (self._lidar.dropout_prob * noise_scale)
            raw_scan[dropout_mask] = 0.0   # Occluded points read as base height

        # 3. Per-env vertical offset (mounting vibration / calibration)
        if self._lidar.vertical_offset_std > 0 and noise_scale > 0:
            v_offset = torch.randn(self.num_envs, 1, device=self.device, dtype=gs.tc_float)
            v_offset = v_offset * self._lidar.vertical_offset_std * noise_scale
            raw_scan = raw_scan + v_offset

        # 4. Latency: some envs get previous step's data
        if self._lidar.latency_prob > 0 and noise_scale > 0:
            latency_mask = torch.rand(self.num_envs, device=self.device) < (self._lidar.latency_prob * noise_scale)
            raw_scan[latency_mask] = self._actor_scan_prev[latency_mask]

        # 5. Full blackout: entire scan goes dead for some envs
        #    Forces policy to be robust with proprioception alone
        if self._lidar.full_blackout_prob > 0 and noise_scale > 0:
            blackout_mask = torch.rand(self.num_envs, device=self.device) < (self._lidar.full_blackout_prob * noise_scale)
            raw_scan[blackout_mask] = 0.0

        # Clip and scale
        raw_scan = torch.clamp(raw_scan, -self._lidar.height_clip, self._lidar.height_clip)
        self._actor_scan_buf[:] = raw_scan * self._lidar.height_scale

    def _update_critic_lidar_scan(self):
        """Compute clean critic LIDAR scan (privileged, no noise)."""
        if not self._lidar.enabled or self._lidar.num_critic_scan == 0:
            return

        raw_scan = self._compute_lidar_scan(
            self._lidar.critic_local_x,
            self._lidar.critic_local_y,
            self._lidar.num_critic_scan,
        )
        raw_scan = torch.clamp(raw_scan, -self._lidar.height_clip, self._lidar.height_clip)
        self._critic_scan_buf[:] = raw_scan * self._lidar.height_scale

    # ==========================================================
    # Legacy height scan (kept for backward compat)
    # ==========================================================

    def _compute_height_scan(self):
        if not self._use_terrain or self._scan_n == 0:
            return torch.zeros(self.num_envs, max(self._scan_n, 1),
                               device=self.device, dtype=gs.tc_float)

        qw = self.base_quat[:, 0]
        qx = self.base_quat[:, 1]
        qy = self.base_quat[:, 2]
        qz = self.base_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
        cos_y = torch.cos(yaw).unsqueeze(1)
        sin_y = torch.sin(yaw).unsqueeze(1)

        lx = self._scan_local_x.unsqueeze(0)
        ly = self._scan_local_y.unsqueeze(0)
        wx = self.base_pos[:, 0:1] + cos_y * lx - sin_y * ly
        wy = self.base_pos[:, 1:2] + sin_y * lx + cos_y * ly

        terrain_h = self._get_terrain_height(wx, wy)
        base_z = self.base_pos[:, 2:3]

        return terrain_h - base_z

    # ==========================================================
    # Terrain: assign envs to difficulty rows
    # ==========================================================

    def _assign_terrain_rows(self, envs_idx, curriculum_level: float):
        if not self._use_terrain or self._num_terrain_rows <= 1:
            return

        n = len(envs_idx)
        if n == 0:
            return

        max_row = int(curriculum_level * (self._num_terrain_rows - 1))
        max_row = max(0, min(max_row, self._num_terrain_rows - 1))

        rows = torch.zeros(n, device=gs.device, dtype=torch.long)

        n_frontier = int(n * 0.40)
        rows[:n_frontier] = max_row

        n_near = int(n * 0.30)
        if max_row >= 2:
            near_min = max(0, max_row - 2)
            near_max = max(0, max_row - 1)
            rows[n_frontier:n_frontier + n_near] = gs_rand_int(
                near_min, near_max, (n_near,), gs.device)
        else:
            rows[n_frontier:n_frontier + n_near] = max_row

        n_easy = n - n_frontier - n_near
        if max_row >= 3:
            easy_max = max_row - 3
        else:
            easy_max = 0
        rows[n_frontier + n_near:] = gs_rand_int(0, easy_max, (n_easy,), gs.device)

        perm = torch.randperm(n, device=gs.device)
        self._env_terrain_row[envs_idx] = rows[perm]

    def _get_terrain_spawn_pos(self, envs_idx):
        if not self._use_terrain:
            return self.base_init_pos.unsqueeze(0).expand(len(envs_idx), -1).clone()

        rows = self._env_terrain_row[envs_idx]
        spawn_pos = torch.zeros((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float)

        for i in range(len(envs_idx)):
            row = int(rows[i].item())
            cx, cy, cz = self._terrain_row_centers[row]
            spawn_pos[i, 0] = cx
            spawn_pos[i, 1] = cy
            spawn_pos[i, 2] = cz + self.base_init_pos[2]

        return spawn_pos

    # ==========================================================
    # Foot contact tracking init
    # ==========================================================

    def _init_foot_contact_tracking(self):
        foot_names = self.env_cfg.get("foot_names", [])
        if not foot_names:
            raise ValueError("env_cfg['foot_names'] is required.")

        link_name_map = {}
        for i, link in enumerate(self.robot.links):
            name = getattr(link, "name", None)
            if name is not None:
                link_name_map[name] = i
        if not link_name_map and hasattr(self.robot, "link_names"):
            for i, name in enumerate(self.robot.link_names):
                link_name_map[name] = i

        if not link_name_map:
            raise RuntimeError("Could not enumerate robot link names.")

        self._foot_link_indices = []
        for fname in foot_names:
            if fname not in link_name_map:
                raise RuntimeError(f"Foot link '{fname}' NOT FOUND.")
            self._foot_link_indices.append(link_name_map[fname])

        cf = self.robot.get_links_net_contact_force()
        if cf is None:
            raise RuntimeError("get_links_net_contact_force() returned None.")

        cf_shape = cf.shape
        n_links_in_cf = cf_shape[1] if len(cf_shape) >= 2 else 0
        for fname, lidx in zip(foot_names, self._foot_link_indices):
            if lidx >= n_links_in_cf:
                raise RuntimeError(f"Foot link '{fname}' idx {lidx} out of range.")

        self._has_link_vel = False
        self._has_link_pos = False
        try:
            lv = self.robot.get_links_vel()
            if lv is not None and lv.shape[0] == self.num_envs:
                self._has_link_vel = True
        except Exception:
            pass
        try:
            lp = self.robot.get_links_pos()
            if lp is not None and lp.shape[0] == self.num_envs:
                self._has_link_pos = True
        except Exception:
            pass

        n_feet = len(foot_names)
        self._n_feet = n_feet
        self._foot_contact = torch.zeros(self.num_envs, n_feet, device=self.device, dtype=torch.bool)
        self._last_foot_contact = torch.zeros_like(self._foot_contact)
        self._feet_air_time = torch.zeros(self.num_envs, n_feet, device=self.device, dtype=gs.tc_float)
        self._foot_contact_threshold = float(self.env_cfg.get("foot_contact_threshold", 1.0))

    def _update_foot_contacts(self):
        cf = self.robot.get_links_net_contact_force()
        self._last_foot_contact[:] = self._foot_contact[:]
        for i, link_idx in enumerate(self._foot_link_indices):
            foot_force_z = cf[:, link_idx, 2]
            self._foot_contact[:, i] = torch.abs(foot_force_z) > self._foot_contact_threshold

    # ==========================================================
    # Curriculum application
    # ==========================================================

    def _rebuild_obs_noise_vec(self, level: float):
        if self.obs_noise_components is None or self.obs_noise_level_max <= 0.0:
            self.obs_noise_vec = None
            return

        noise_level = _lerp(0.0, self.obs_noise_level_max, level)
        nc = self.obs_noise_components

        # Proprioceptive noise (same dimension layout as before, but now
        # the obs vector is longer because of the LIDAR scan at the end)
        num_proprio = 3 + 3 + 3 + 12 + 12 + self.num_actions  # base proprioceptive obs
        nv = torch.zeros(self.num_obs, device=gs.device, dtype=gs.tc_float)

        nv[0:3] = nc.get("ang_vel", 0.0) * self.obs_scales["ang_vel"] * noise_level
        nv[3:6] = nc.get("gravity", 0.0) * noise_level
        nv[6:9] = 0.0
        nv[9:21] = nc.get("dof_pos", 0.0) * self.obs_scales["dof_pos"] * noise_level
        nv[21:33] = nc.get("dof_vel", 0.0) * self.obs_scales["dof_vel"] * noise_level
        nv[33:num_proprio] = 0.0

        # LIDAR scan noise is handled separately in _update_actor_lidar_scan()
        # so we leave the LIDAR portion of obs_noise_vec as zero here
        if self._lidar.enabled and self._lidar.num_actor_scan > 0:
            nv[num_proprio:] = 0.0  # LIDAR noise applied in scan computation

        self.obs_noise_vec = nv

    def _get_dr_level(self, terrain_level: float) -> float:
        if not self._dr_enabled:
            return terrain_level
        if terrain_level < self._dr_terrain_gate:
            return self._dr_phase1_level
        else:
            progress = (terrain_level - self._dr_terrain_gate) / max(
                1e-6, 1.0 - self._dr_terrain_gate)
            progress = _clamp01(progress)
            return _lerp(self._dr_phase1_level, 1.0, progress)

    def _apply_curriculum_level(self, force: bool = False):
        if not self.curriculum.enabled and not force:
            return

        lvl = float(self.curriculum.level) if self.curriculum.enabled else 1.0
        dr_lvl = self._get_dr_level(lvl)

        self._obs_noise_level_cur = _lerp(0.0, self.obs_noise_level_max, dr_lvl)
        self._action_noise_std_cur = _lerp(0.0, self.action_noise_std_max, dr_lvl)
        self._rebuild_obs_noise_vec(dr_lvl)

        # LIDAR noise ramps with DR level
        self._lidar_noise_level = dr_lvl

        if self.push_force_range_hard is None:
            self._push_enable_cur = False
            self._push_force_range_cur = [0.0, 0.0]
            self._push_interval = 10**9
        else:
            if dr_lvl < self.push_start:
                self._push_enable_cur = False
                self._push_force_range_cur = [0.0, 0.0]
                self._push_interval = int(self.push_interval_s_easy / self.dt)
            else:
                s = (dr_lvl - self.push_start) / max(1e-6, (1.0 - self.push_start))
                s = _clamp01(s)
                low_h, high_h = float(self.push_force_range_hard[0]), float(self.push_force_range_hard[1])
                self._push_force_range_cur = [low_h * s, high_h * s]
                interval_s = _lerp(self.push_interval_s_easy, self.push_interval_s_hard, s)
                self._push_interval = max(1, int(interval_s / self.dt))
                self._push_enable_cur = True

        self._delay_max_cur = int(round(_lerp(
            float(self.delay_easy_max),
            float(self._max_delay_steps),
            dr_lvl,
        )))

        if self._cmd_curriculum:
            frac = _lerp(self._cmd_start_frac, 1.0, lvl)
            for key, full_range in self._cmd_full_ranges.items():
                center = (full_range[0] + full_range[1]) / 2.0
                half_span = (full_range[1] - full_range[0]) / 2.0
                self._cmd_cur_ranges[key] = [
                    center - half_span * frac,
                    center + half_span * frac,
                ]
        else:
            self._cmd_cur_ranges = {k: list(v) for k, v in self._cmd_full_ranges.items()}

        self.extras.setdefault("curriculum", {})
        self.extras["curriculum"].update(self.curriculum.state_dict())
        self.extras["curriculum"].update(
            {
                "obs_noise_level_cur": float(self._obs_noise_level_cur),
                "action_noise_std_cur": float(self._action_noise_std_cur),
                "push_enable": bool(self._push_enable_cur),
                "push_force_range_cur": list(self._push_force_range_cur),
                "push_interval_steps": int(self._push_interval),
                "delay_max_cur": int(self._delay_max_cur),
                "cmd_ranges": {k: list(v) for k, v in self._cmd_cur_ranges.items()},
                "terrain_max_row": int(lvl * (self._num_terrain_rows - 1)) if self._use_terrain else 0,
                "dr_level": float(dr_lvl),
                "lidar_noise_level": float(self._lidar_noise_level),
            }
        )

    def _maybe_update_curriculum_on_reset(self, envs_idx):
        if not self.curriculum.enabled:
            return

        n = int(len(envs_idx))
        if n <= 0:
            return

        timeouts = 0.0
        if "time_outs" in self.extras:
            timeouts = float(self.extras["time_outs"][envs_idx].sum().item())

        ep_steps = self.episode_length_buf[envs_idx].to(dtype=gs.tc_float).clamp_min(1.0)
        ep_seconds = ep_steps * self.dt

        tracking_int = 0.0
        if "tracking_lin_vel" in self.episode_sums:
            tracking_int = tracking_int + self.episode_sums["tracking_lin_vel"][envs_idx]
        if "tracking_ang_vel" in self.episode_sums:
            tracking_int = tracking_int + self.episode_sums["tracking_ang_vel"][envs_idx]

        tracking_per_sec = (tracking_int / ep_seconds)
        tracking_sum = float(tracking_per_sec.sum().item())

        self._curr_ep_total += n
        self._curr_timeout_total += timeouts
        self._curr_tracking_sum += tracking_sum
        self._curr_tracking_n += n

        if self._curr_ep_total >= self.curr_update_every_episodes:
            timeout_rate = float(self._curr_timeout_total / max(1, self._curr_ep_total))
            fall_rate = float(1.0 - timeout_rate)
            tracking_avg = float(self._curr_tracking_sum / max(1, self._curr_tracking_n))

            changed = self.curriculum.update(timeout_rate, tracking_avg, fall_rate)
            if changed:
                self._apply_curriculum_level()

            self._curr_ep_total = 0
            self._curr_timeout_total = 0.0
            self._curr_tracking_sum = 0.0
            self._curr_tracking_n = 0

    # ==========================================================
    # Domain randomisation
    # ==========================================================

    def _randomize_friction(self, envs_idx, t_sample: float):
        if "friction_range" not in self.env_cfg:
            return
        self._global_dr_reset_counter += len(envs_idx)
        if self._global_dr_reset_counter < self._global_dr_update_interval:
            return
        self._global_dr_reset_counter = 0
        mu_range = _lerp_range(self.friction_easy, self.friction_hard, t_sample)
        mu = float(gs_rand_float(mu_range[0], mu_range[1], (1,), gs.device).item())
        self.ground.set_friction(mu)
        self.robot.set_friction(mu)
        self._current_friction[:] = mu

    def _randomize_kp_kd(self, envs_idx, t_sample: float):
        n = len(envs_idx)
        if self.pls_enable:
            if "kp_factor_range" in self.env_cfg:
                kpf_range = _lerp_range(self.kp_factor_easy, self.kp_factor_hard, t_sample)
                self._kp_factors[envs_idx] = gs_rand_float(
                    kpf_range[0], kpf_range[1], (n, self.num_pos_actions), gs.device)
            if "kd_factor_range" in self.env_cfg:
                kdf_range = _lerp_range(self.kd_factor_easy, self.kd_factor_hard, t_sample)
                self._kd_factors[envs_idx] = gs_rand_float(
                    kdf_range[0], kdf_range[1], (n, self.num_pos_actions), gs.device)
        else:
            if "kp_range" in self.env_cfg:
                kp_range = _lerp_range(self.kp_easy, self.kp_hard, t_sample)
                kp_val = gs_rand_float(kp_range[0], kp_range[1], (n, 1), gs.device)
                kd_range = _lerp_range(self.kd_easy, self.kd_hard, t_sample)
                kd_val = gs_rand_float(kd_range[0], kd_range[1], (n, 1), gs.device)
                base_kp = kp_val.expand(n, self.num_pos_actions)
                base_kd = kd_val.expand(n, self.num_pos_actions)

                if "kp_factor_range" in self.env_cfg:
                    kpf_range = _lerp_range(self.kp_factor_easy, self.kp_factor_hard, t_sample)
                    self._kp_factors[envs_idx] = gs_rand_float(
                        kpf_range[0], kpf_range[1], (n, self.num_pos_actions), gs.device)
                if "kd_factor_range" in self.env_cfg:
                    kdf_range = _lerp_range(self.kd_factor_easy, self.kd_factor_hard, t_sample)
                    self._kd_factors[envs_idx] = gs_rand_float(
                        kdf_range[0], kdf_range[1], (n, self.num_pos_actions), gs.device)

                self._effective_kp[envs_idx] = base_kp * self._kp_factors[envs_idx]
                self._effective_kd[envs_idx] = base_kd * self._kd_factors[envs_idx]

                if not self._use_manual_pd:
                    mean_kp = float(self._effective_kp[envs_idx].mean().item())
                    mean_kd = float(self._effective_kd[envs_idx].mean().item())
                    self.robot.set_dofs_kp([mean_kp] * self.num_pos_actions, self.motors_dof_idx)
                    self.robot.set_dofs_kv([mean_kd] * self.num_pos_actions, self.motors_dof_idx)

    def _randomize_mass(self, envs_idx, t_sample: float):
        if "mass_shift_range" in self.env_cfg:
            ms_range = _lerp_range(self.mass_easy, self.mass_hard, t_sample)
            shift = float(gs_rand_float(ms_range[0], ms_range[1], (1,), gs.device).item())
            self.robot.set_mass_shift([shift], [0])
            self._current_mass_shift[:] = shift

        if "com_shift_range" in self.env_cfg:
            cs_range = _lerp_range(self.com_easy, self.com_hard, t_sample)
            dx = float(gs_rand_float(cs_range[0], cs_range[1], (1,), gs.device).item())
            dy = float(gs_rand_float(cs_range[0], cs_range[1], (1,), gs.device).item())
            dz = float(gs_rand_float(cs_range[0], cs_range[1], (1,), gs.device).item())
            self.robot.set_COM_shift([[dx, dy, dz]], [0])
            self._current_com_shift[:] = torch.tensor([dx, dy, dz], device=gs.device)

    def _randomize_gravity_offset(self, envs_idx, t_sample: float):
        if "gravity_offset_range" not in self.env_cfg:
            return
        n = len(envs_idx)
        go_range = _lerp_range(self.gravity_offset_easy, self.gravity_offset_hard, t_sample)
        self._gravity_offset[envs_idx] = gs_rand_float(
            go_range[0], go_range[1], (n, 3), gs.device)

    def _randomize_leg_mass(self, envs_idx, t_sample: float):
        if not self._has_leg_mass_dr:
            return
        lm_range = _lerp_range(self.leg_mass_easy, self.leg_mass_hard, t_sample)
        shifts = []
        for i, link_idx in enumerate(self._hip_link_indices):
            shift = float(gs_rand_float(lm_range[0], lm_range[1], (1,), gs.device).item())
            try:
                self.robot.set_mass_shift([shift], [link_idx])
            except Exception:
                pass
            shifts.append(shift)
        self._current_leg_mass_shifts[:] = torch.tensor(shifts, device=gs.device)

    def _randomize_motor_strength(self, envs_idx, t_sample: float):
        if "motor_strength_range" not in self.env_cfg:
            return
        n = len(envs_idx)
        ms_range = _lerp_range(self.motor_strength_easy, self.motor_strength_hard, t_sample)
        self._motor_strength[envs_idx] = gs_rand_float(
            ms_range[0], ms_range[1], (n, self.num_pos_actions), gs.device)

    def _randomize_delay(self, envs_idx):
        n = len(envs_idx)
        max_d = max(self._min_delay_steps, min(self._delay_max_cur, self._max_delay_steps))
        self._delay_steps[envs_idx] = gs_rand_int(
            self._min_delay_steps, max_d, (n,), gs.device)

    # ==========================================================
    # Per-step DR
    # ==========================================================

    def _apply_push(self):
        if self.push_force_range_hard is None or not self._push_enable_cur:
            self._current_push_force[:] = 0.0
            return

        if self._push_counter % self._push_interval == 0:
            low, high = self._push_force_range_cur
            self._push_stored_force[:, 0] = gs_rand_float(low, high, (self.num_envs,), self.device)
            self._push_stored_force[:, 1] = gs_rand_float(low, high, (self.num_envs,), self.device)
            self._push_stored_force[:, 2] = 0.0

            dur_lo = self._push_duration_steps[0]
            dur_hi = self._push_duration_steps[1]
            self._push_remaining[:] = gs_rand_int(dur_lo, dur_hi, (self.num_envs,), self.device)

        active = (self._push_remaining > 0).unsqueeze(-1).float()
        force = self._push_stored_force * active

        self._current_push_force[:] = force
        self._push_remaining = (self._push_remaining - 1).clamp(min=0)

        try:
            self.scene.sim.rigid_solver.apply_links_external_force(
                force=force,
                links_idx=[self._base_link_idx],
                envs_idx=self._all_envs_idx,
            )
        except TypeError:
            self.scene.sim.rigid_solver.apply_links_external_force(
                force=force.detach().cpu().numpy(),
                links_idx=[self._base_link_idx],
                envs_idx=self._all_envs_idx.detach().cpu().numpy(),
            )

        self._push_counter += 1

    def _add_obs_noise(self):
        if self.obs_noise_vec is not None:
            self.obs_buf += torch.randn_like(self.obs_buf) * self.obs_noise_vec

    # ==========================================================
    # Action delay buffer
    # ==========================================================

    def _store_action(self, action):
        self._action_history[:, self._action_history_write_idx, :] = action
        self._action_history_write_idx = (self._action_history_write_idx + 1) % self._delay_buf_size

    def _get_delayed_action(self):
        read_idx = (self._action_history_write_idx - 1 - self._delay_steps) % self._delay_buf_size
        batch_idx = torch.arange(self.num_envs, device=gs.device)
        return self._action_history[batch_idx, read_idx, :]

    # ==========================================================
    # Commands
    # ==========================================================

    def _resample_commands(self, envs_idx):
        n = len(envs_idx)
        if n == 0:
            return

        if self._compound_commands:
            self.commands[envs_idx, 0] = gs_rand_float(
                self._cmd_cur_ranges["lin_vel_x"][0],
                self._cmd_cur_ranges["lin_vel_x"][1], (n,), gs.device)
            self.commands[envs_idx, 1] = gs_rand_float(
                self._cmd_cur_ranges["lin_vel_y"][0],
                self._cmd_cur_ranges["lin_vel_y"][1], (n,), gs.device)
            self.commands[envs_idx, 2] = gs_rand_float(
                self._cmd_cur_ranges["ang_vel"][0],
                self._cmd_cur_ranges["ang_vel"][1], (n,), gs.device)
        else:
            self.commands[envs_idx] = 0.0
            choice = torch.randint(0, 3, (n,), device=gs.device)
            for i, (key, dim) in enumerate(zip(["lin_vel_x", "lin_vel_y", "ang_vel"], [0, 1, 2])):
                mask = (choice == i)
                if mask.any():
                    self.commands[envs_idx[mask], dim] = gs_rand_float(
                        self._cmd_cur_ranges[key][0],
                        self._cmd_cur_ranges[key][1], (mask.sum(),), gs.device)

        if self._standing_env_indices is not None and len(self._standing_env_indices) > 0:
            standing_mask = torch.isin(envs_idx, self._standing_env_indices)
            if standing_mask.any():
                self.commands[envs_idx[standing_mask], :] = 0.0

    # ==========================================================
    # PLS
    # ==========================================================

    def _compute_pls_kp_kd(self, stiffness_actions):
        kp_per_leg = self.pls_kp_default + stiffness_actions * self.pls_kp_action_scale
        kp_per_leg = torch.clamp(kp_per_leg, self.pls_kp_range[0], self.pls_kp_range[1])
        kp_per_joint = kp_per_leg[:, :, None].expand(-1, 4, 3).reshape(-1, 12)
        kd_per_joint = 0.2 * torch.sqrt(kp_per_joint)
        effective_kp = kp_per_joint * self._kp_factors * self._motor_strength
        effective_kd = kd_per_joint * self._kd_factors
        return effective_kp, effective_kd

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        self._store_action(self.actions)
        delayed_actions = self._get_delayed_action()
        self._applied_actions[:] = delayed_actions

        pos_actions = delayed_actions[:, :self.num_pos_actions]
        target_dof_pos = pos_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        if self._action_noise_std_cur > 0.0:
            target_dof_pos = target_dof_pos + torch.randn_like(target_dof_pos) * self._action_noise_std_cur

        self._target_dof_pos[:] = target_dof_pos

        if self._use_manual_pd:
            if self.pls_enable:
                stiffness_actions = delayed_actions[:, self.num_pos_actions:]
                self._effective_kp, self._effective_kd = self._compute_pls_kp_kd(stiffness_actions)

            pos_error = target_dof_pos - self.dof_pos
            torque = self._effective_kp * pos_error - self._effective_kd * self.dof_vel
            torque = torch.clamp(torque, -self._torque_limits, self._torque_limits)
            self.robot.control_dofs_force(torque, self.motors_dof_idx)
        else:
            self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        self._apply_push()
        self.scene.step()

        self.episode_length_buf += 1

        # ---- State update ----
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        if self._has_foot_contact_rewards:
            self._update_foot_contacts()

        # ---- LIDAR scan update ---- ← NEW
        if self._lidar.enabled:
            self._update_actor_lidar_scan()
            self._update_critic_lidar_scan()

        # Command resampling
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # Termination → Rewards → Reset → Obs
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 2]) > self.env_cfg["termination_if_z_vel_greater_than"]
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 1]) > self.env_cfg["termination_if_y_vel_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ---- Build actor observation (NOW INCLUDES LIDAR SCAN) ----
        obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],          # 3
            self.projected_gravity + self._gravity_offset,            # 3
            self.commands * self.commands_scale,                       # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],               # 12
            self._applied_actions,                                    # num_actions (16)
        ]

        # Append actor LIDAR scan  ← NEW
        if self._lidar.enabled and self._lidar.num_actor_scan > 0:
            obs_parts.append(self._actor_scan_buf)                    # num_actor_scan (15)

        self.obs_buf = torch.cat(obs_parts, axis=-1)

        self._add_obs_noise()

        if self.num_privileged_obs is not None:
            self._build_privileged_obs()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = (
            self.privileged_obs_buf if self.num_privileged_obs else self.obs_buf
        )
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ==========================================================
    # Privileged observations
    # ==========================================================

    def _build_privileged_obs(self):
        priv_dim = self.privileged_obs_buf.shape[1]

        # Copy full actor obs (which now includes LIDAR scan)
        self.privileged_obs_buf[:, :self.num_obs] = self.obs_buf

        idx = self.num_obs

        # --- Privileged-only observations ---
        self.privileged_obs_buf[:, idx:idx+3] = self.base_lin_vel * self.obs_scales["lin_vel"]
        idx += 3
        self.privileged_obs_buf[:, idx] = self._current_friction
        idx += 1
        self.privileged_obs_buf[:, idx:idx+12] = self._kp_factors
        idx += 12
        self.privileged_obs_buf[:, idx:idx+12] = self._kd_factors
        idx += 12
        self.privileged_obs_buf[:, idx:idx+12] = self._motor_strength
        idx += 12
        self.privileged_obs_buf[:, idx] = self._current_mass_shift
        idx += 1
        self.privileged_obs_buf[:, idx:idx+3] = self._current_com_shift
        idx += 3
        self.privileged_obs_buf[:, idx:idx+4] = self._current_leg_mass_shifts
        idx += 4
        self.privileged_obs_buf[:, idx:idx+3] = self._gravity_offset
        idx += 3
        self.privileged_obs_buf[:, idx:idx+3] = self._current_push_force
        idx += 3
        if self._max_delay_steps > 0:
            self.privileged_obs_buf[:, idx] = self._delay_steps.float() / float(self._max_delay_steps)
        idx += 1

        # Terrain difficulty
        if idx < priv_dim:
            if self._use_terrain:
                self.privileged_obs_buf[:, idx] = (
                    self._env_terrain_row.float() / max(1, self._num_terrain_rows - 1)
                )
            idx += 1

        # ---- Critic LIDAR scan (clean, dense, privileged) ----  ← NEW
        if self._lidar.enabled and self._lidar.num_critic_scan > 0 and (idx + self._lidar.num_critic_scan) <= priv_dim:
            self.privileged_obs_buf[:, idx:idx + self._lidar.num_critic_scan] = self._critic_scan_buf
            idx += self._lidar.num_critic_scan

    # ==========================================================

    def get_observations(self):
        self.extras["observations"]["critic"] = (
            self.privileged_obs_buf if self.num_privileged_obs else self.obs_buf
        )
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        if self.num_privileged_obs is not None:
            return self.privileged_obs_buf
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self._maybe_update_curriculum_on_reset(envs_idx)

        curriculum_level = self.curriculum.level if self.curriculum.enabled else 1.0
        t_sample = self._get_dr_level(curriculum_level)

        if not self._lock_terrain_rows:
            self._assign_terrain_rows(envs_idx, curriculum_level)

        self._randomize_friction(envs_idx, t_sample)
        self._randomize_kp_kd(envs_idx, t_sample)
        self._randomize_mass(envs_idx, t_sample)
        self._randomize_leg_mass(envs_idx, t_sample)
        self._randomize_gravity_offset(envs_idx, t_sample)
        self._randomize_motor_strength(envs_idx, t_sample)
        self._randomize_delay(envs_idx)

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        spawn_pos = self._get_terrain_spawn_pos(envs_idx)
        self.base_pos[envs_idx] = spawn_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        if "init_pos_z_range" in self.env_cfg:
            low, high = self.env_cfg["init_pos_z_range"]
            z_offset = gs_rand_float(low, high, (len(envs_idx),), gs.device)
            self.base_pos[envs_idx, 2] = spawn_pos[:, 2] + z_offset - self.base_init_pos[2] + self.base_init_pos[2]

        if "init_euler_range" in self.env_cfg:
            low_deg, high_deg = self.env_cfg["init_euler_range"]
            low_rad = math.radians(low_deg)
            high_rad = math.radians(high_deg)
            n = len(envs_idx)
            roll = gs_rand_float(low_rad, high_rad, (n,), gs.device)
            pitch = gs_rand_float(low_rad, high_rad, (n,), gs.device)
            yaw = torch.zeros(n, device=gs.device, dtype=gs.tc_float)
            self.base_quat[envs_idx] = euler_to_quat_wxyz(roll, pitch, yaw)

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self._applied_actions[envs_idx] = 0.0
        self._action_history[envs_idx] = 0.0

        self._last_base_pos_x[envs_idx] = self.base_pos[envs_idx, 0]

        if self.push_force_range_hard is not None:
            self._push_stored_force[envs_idx] = 0.0
            self._push_remaining[envs_idx] = 0

        if self._has_foot_contact_rewards:
            self._feet_air_time[envs_idx] = 0.0
            self._foot_contact[envs_idx] = False
            self._last_foot_contact[envs_idx] = False

        # Reset LIDAR scan buffers for reset envs
        if self._lidar.enabled:
            self._actor_scan_buf[envs_idx] = 0.0
            self._actor_scan_prev[envs_idx] = 0.0
            self._critic_scan_buf[envs_idx] = 0.0

        self.extras["episode"] = {}
        ep_steps = self.episode_length_buf[envs_idx].to(dtype=gs.tc_float).clamp_min(1.0)
        ep_seconds = ep_steps * self.dt
        for key in self.episode_sums.keys():
            per_sec = (self.episode_sums[key][envs_idx] / ep_seconds).mean().item()
            self.extras["episode"]["rew_" + key] = per_sec
            self.episode_sums[key][envs_idx] = 0.0

        if self._use_terrain:
            mean_row = self._env_terrain_row[envs_idx].float().mean().item()
            self.extras["episode"]["terrain_mean_row"] = mean_row

        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ================================================================
    # Reward functions
    # ================================================================

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        z_vel = self.base_lin_vel[:, 2]
        deadzone = float(self.reward_cfg.get("lin_vel_z_deadzone", 0.0))
        if deadzone > 0.0:
            z_vel_excess = torch.clamp(torch.abs(z_vel) - deadzone, min=0.0)
            return torch.square(z_vel_excess)
        return torch.square(z_vel)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        if self._use_terrain:
            terrain_z = self._get_terrain_height(
                self.base_pos[:, 0], self.base_pos[:, 1])
            height_above_ground = self.base_pos[:, 2] - terrain_z
        else:
            height_above_ground = self.base_pos[:, 2]
        return torch.square(height_above_ground - self.reward_cfg["base_height_target"])

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_orientation_penalty(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_roll_only(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_stand_still(self):
        cmd_norm = torch.norm(self.commands, dim=1)
        still_mask = (cmd_norm < 0.1).float()
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * still_mask

    def _reward_stand_still_vel(self):
        cmd_norm = torch.norm(self.commands, dim=1)
        still_mask = (cmd_norm < 0.1).float()
        lin_vel_penalty = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        ang_vel_penalty = torch.square(self.base_ang_vel[:, 2])
        return (lin_vel_penalty + 0.5 * ang_vel_penalty) * still_mask

    def _reward_feet_stance(self):
        cmd_norm = torch.norm(self.commands, dim=1)
        still_mask = (cmd_norm < 0.1).float()
        feet_in_air = (~self._foot_contact).float()
        air_penalty = torch.sum(self._feet_air_time, dim=1) + torch.sum(feet_in_air, dim=1)
        return air_penalty * still_mask

    def _reward_feet_air_time(self):
        contact = self._foot_contact
        first_contact = (self._feet_air_time > 0.0) & contact
        self._feet_air_time += self.dt
        self._feet_air_time *= ~contact
        target = self.reward_cfg.get("feet_air_time_target", 0.1)
        rew = torch.sum((self._feet_air_time - target) * first_contact.float(), dim=1)
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return rew * moving

    def _reward_foot_slip(self):
        contact = self._foot_contact.float()
        if self._has_link_vel:
            link_vel = self.robot.get_links_vel()
            slip = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            for i, link_idx in enumerate(self._foot_link_indices):
                foot_vel_xy = link_vel[:, link_idx, :2]
                slip += contact[:, i] * torch.sum(torch.square(foot_vel_xy), dim=1)
            return slip
        else:
            any_contact = contact.sum(dim=1).clamp(max=1.0)
            base_xy_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
            return any_contact * base_xy_vel_sq

    def _reward_foot_clearance(self):
        if not self._has_link_pos:
            return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        link_pos = self.robot.get_links_pos()
        target_height = self.reward_cfg.get("feet_height_target", 0.075)
        swing = (~self._foot_contact).float()
        penalty = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        if self._has_link_vel:
            link_vel = self.robot.get_links_vel()

        for i, link_idx in enumerate(self._foot_link_indices):
            foot_x = link_pos[:, link_idx, 0]
            foot_y = link_pos[:, link_idx, 1]
            foot_z = link_pos[:, link_idx, 2]

            if self._use_terrain:
                terrain_z_at_foot = self._get_terrain_height(foot_x, foot_y)
                foot_z_rel = foot_z - terrain_z_at_foot
            else:
                foot_z_rel = foot_z

            height_error = torch.square(target_height - foot_z_rel)

            if self._has_link_vel:
                foot_vel_xy_norm = torch.norm(link_vel[:, link_idx, :2], dim=1)
                penalty += swing[:, i] * height_error * foot_vel_xy_norm
            else:
                penalty += swing[:, i] * height_error

        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return penalty * moving

    def _reward_forward_progress(self):
        dx = self.base_pos[:, 0] - self._last_base_pos_x
        self._last_base_pos_x[:] = self.base_pos[:, 0]
        return dx

    def _reward_joint_tracking(self):
        return torch.sum(torch.square(self._target_dof_pos - self.dof_pos), dim=1)

    def _reward_energy(self):
        torques = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return torch.sum(torch.abs(torques * self.dof_vel), dim=1)

    def _reward_torque_load(self):
        tau = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return torch.sum(torch.abs(tau), dim=1)