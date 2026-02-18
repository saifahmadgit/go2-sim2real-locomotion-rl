import math
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def euler_to_quat_wxyz(roll, pitch, yaw):
    """Convert batched euler angles (rad) to quaternion (w,x,y,z)."""
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
    """a,b are [low,high]."""
    return [_lerp(a[0], b[0], t), _lerp(a[1], b[1], t)]


class CurriculumManager:
    """
    Metric-gated curriculum:
      - Tracks EMA of timeout_rate, fall_rate, tracking_per_sec
      - Increases level when "ready" condition holds for N checks
      - Decreases level when "too hard" holds for M checks
    """

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

        # Mixture sampling (for per-episode sampling of some params)
        self.mix_prob_current = float(self.cfg.get("mix_prob_current", 0.80))
        self.mix_level_low = float(self.cfg.get("mix_level_low", 0.0))
        self.mix_level_high = float(self.cfg.get("mix_level_high", 0.6))

        # Internal state
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


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
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

        self.ground = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

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

        # Nominal PD (can be overridden by DR at reset)
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Rewards
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
        # Foot contact tracking (for feet_air_time reward)
        # Only initialized if a foot-contact reward is active
        # ----------------------------------------------------------
        self._foot_reward_names = ["feet_air_time"]
        self._has_foot_contact_rewards = any(
            name in self.reward_scales and self.reward_scales[name] != 0.0
            for name in self._foot_reward_names
        )
        if self._has_foot_contact_rewards:
            self._init_foot_contact_tracking()

        # ----------------------------------------------------------
        # DR base config (max / "hard" settings as in env_cfg)
        # Curriculum will scale these.
        # ----------------------------------------------------------
        self.obs_noise_components = self.env_cfg.get("obs_noise", None)
        # FIX: obs_noise_level is the HARD maximum multiplier.
        # At full curriculum (level=1.0), effective noise = component * obs_scale * obs_noise_level.
        # Set to 1.0 so at full curriculum you get official-grade noise.
        self.obs_noise_level_max = float(self.env_cfg.get("obs_noise_level", 0.0)) if self.obs_noise_components else 0.0

        self.action_noise_std_max = float(self.env_cfg.get("action_noise_std", 0.0)) if ("action_noise_std" in self.env_cfg) else 0.0

        # Push base settings
        self.push_interval_s_hard = float(self.env_cfg.get("push_interval_s", 5.0))
        self.push_force_range_hard = self.env_cfg.get("push_force_range", None)

        # Cache solver indices for push
        self._push_counter = 0
        if self.push_force_range_hard is not None:
            self._all_envs_idx = torch.arange(self.num_envs, device=gs.device, dtype=gs.tc_int)
            try:
                self._base_link_idx = int(self.robot.links[1].idx)
            except Exception:
                self._base_link_idx = int(self.robot.links[0].idx)

        # ----------------------------------------------------------
        # Command curriculum
        # ----------------------------------------------------------
        self._cmd_curriculum = bool(command_cfg.get("cmd_curriculum", False))
        self._cmd_start_frac = float(command_cfg.get("cmd_curriculum_start_frac", 0.1))
        self._rel_standing_envs = float(command_cfg.get("rel_standing_envs", 0.0))

        # Store the FULL command ranges from config
        self._cmd_full_ranges = {
            "lin_vel_x": list(command_cfg["lin_vel_x_range"]),
            "lin_vel_y": list(command_cfg["lin_vel_y_range"]),
            "ang_vel": list(command_cfg["ang_vel_range"]),
        }
        # Current (curriculum-adjusted) ranges — start at full if no curriculum
        self._cmd_cur_ranges = {k: list(v) for k, v in self._cmd_full_ranges.items()}

        # Standing env indices (bottom N% of envs always get zero commands)
        n_standing = int(self._rel_standing_envs * self.num_envs)
        self._standing_env_indices = torch.arange(n_standing, device=gs.device) if n_standing > 0 else None

        # ----------------------------------------------------------
        # Buffers
        # ----------------------------------------------------------
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

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

        self.friction_easy = self.curr_cfg.get("friction_easy", [0.85, 0.95])
        self.friction_hard = self.env_cfg.get("friction_range", self.friction_easy)

        kp_nom = float(self.env_cfg.get("kp", 60.0))
        kd_nom = float(self.env_cfg.get("kd", 2.0))
        self.kp_easy = self.curr_cfg.get("kp_easy", [0.9 * kp_nom, 1.1 * kp_nom])
        self.kd_easy = self.curr_cfg.get("kd_easy", [0.75 * kd_nom, 1.25 * kd_nom])

        self.kp_hard = self.env_cfg.get("kp_range", self.kp_easy)
        self.kd_hard = self.env_cfg.get("kd_range", self.kd_easy)

        # Mass DR curriculum ranges
        self.mass_easy = self.curr_cfg.get("mass_shift_easy", [-0.1, 0.1])
        self.mass_hard = self.env_cfg.get("mass_shift_range", self.mass_easy)
        self.com_easy = self.curr_cfg.get("com_shift_easy", [-0.005, 0.005])
        self.com_hard = self.env_cfg.get("com_shift_range", self.com_easy)

        # Push curriculum shaping
        self.push_start = float(self.curr_cfg.get("push_start", 0.30))
        self.push_interval_s_easy = float(self.curr_cfg.get("push_interval_easy_s", 6.0))

        # Runtime (current) values driven by curriculum level
        self._obs_noise_level_cur = 0.0
        self._action_noise_std_cur = 0.0
        self._push_enable_cur = False
        self._push_interval = int(self.push_interval_s_easy / self.dt) if self.push_force_range_hard is not None else 1
        self._push_force_range_cur = [0.0, 0.0]

        self.obs_noise_vec = None
        self._apply_curriculum_level(force=True)

    # ==========================================================
    # Foot contact tracking init (with explicit error handling)
    # ==========================================================

    def _init_foot_contact_tracking(self):
        """
        Initialize foot contact detection for feet_air_time reward.
        Raises RuntimeError with diagnostic info if Genesis doesn't support
        the required APIs or if foot link names don't match the URDF.
        """
        foot_names = self.env_cfg.get("foot_names", [])
        if not foot_names:
            raise ValueError(
                "env_cfg['foot_names'] is required for feet_air_time reward. "
                "Expected e.g.: ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']"
            )

        # ---- Build link name -> index mapping ----
        link_name_map = {}
        for i, link in enumerate(self.robot.links):
            name = getattr(link, "name", None)
            if name is not None:
                link_name_map[name] = i

        if not link_name_map:
            # Fallback: try .link_names if available
            if hasattr(self.robot, "link_names"):
                for i, name in enumerate(self.robot.link_names):
                    link_name_map[name] = i

        if not link_name_map:
            raise RuntimeError(
                f"Could not enumerate robot link names. "
                f"Robot has {len(self.robot.links)} links but none have a 'name' attribute "
                f"and robot.link_names is not available. "
                f"feet_air_time reward requires named links to identify feet. "
                f"Link objects: {[type(l).__name__ for l in self.robot.links[:5]]}..."
            )

        # ---- Match foot names ----
        self._foot_link_indices = []
        for fname in foot_names:
            if fname not in link_name_map:
                raise RuntimeError(
                    f"Foot link '{fname}' NOT FOUND in robot URDF.\n"
                    f"Available link names ({len(link_name_map)}): {sorted(link_name_map.keys())}\n"
                    f"Check env_cfg['foot_names'] matches your go2.urdf link names exactly."
                )
            self._foot_link_indices.append(link_name_map[fname])

        # ---- Verify contact force API works ----
        try:
            cf = self.robot.get_links_net_contact_force()
        except AttributeError as e:
            raise RuntimeError(
                f"Genesis robot has no 'get_links_net_contact_force()' method.\n"
                f"This API is required for feet_air_time reward.\n"
                f"Available methods: {[m for m in dir(self.robot) if 'contact' in m.lower() or 'force' in m.lower()]}\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error calling get_links_net_contact_force(): {e}\n"
                f"feet_air_time reward requires working contact force sensing."
            ) from e

        if cf is None:
            raise RuntimeError(
                "get_links_net_contact_force() returned None.\n"
                "Contact sensing may not be enabled in your Genesis build.\n"
                "feet_air_time reward requires contact forces."
            )

        cf_shape = cf.shape
        if len(cf_shape) < 2 or cf_shape[0] != self.num_envs:
            raise RuntimeError(
                f"Contact force tensor has unexpected shape {cf_shape}.\n"
                f"Expected first dimension = num_envs ({self.num_envs}).\n"
                f"Cannot reliably detect foot contacts."
            )

        # Verify foot link indices are in range
        n_links_in_cf = cf_shape[1] if len(cf_shape) >= 2 else 0
        for fname, lidx in zip(foot_names, self._foot_link_indices):
            if lidx >= n_links_in_cf:
                raise RuntimeError(
                    f"Foot link '{fname}' has index {lidx} but contact force tensor "
                    f"only has {n_links_in_cf} links (shape={cf_shape}).\n"
                    f"There may be a mismatch between link enumeration and contact API."
                )

        # ---- Allocate buffers ----
        n_feet = len(foot_names)
        self._n_feet = n_feet
        self._foot_contact = torch.zeros(self.num_envs, n_feet, device=self.device, dtype=torch.bool)
        self._last_foot_contact = torch.zeros_like(self._foot_contact)
        self._feet_air_time = torch.zeros(self.num_envs, n_feet, device=self.device, dtype=gs.tc_float)

        self._foot_contact_threshold = float(self.env_cfg.get("foot_contact_threshold", 1.0))

        print(f"[Go2Env] Foot contact tracking INITIALIZED successfully:")
        print(f"  foot_names       : {foot_names}")
        print(f"  link_indices     : {self._foot_link_indices}")
        print(f"  contact_cf_shape : {cf_shape}")
        print(f"  threshold        : {self._foot_contact_threshold} N")

    def _update_foot_contacts(self):
        """Update foot contact states from contact forces. Call every step before rewards."""
        cf = self.robot.get_links_net_contact_force()  # (num_envs, num_links, 3)
        self._last_foot_contact[:] = self._foot_contact[:]
        for i, link_idx in enumerate(self._foot_link_indices):
            # Use z-component magnitude for ground contact
            foot_force_z = cf[:, link_idx, 2]
            self._foot_contact[:, i] = torch.abs(foot_force_z) > self._foot_contact_threshold

    # ==========================================================
    # Curriculum application
    # ==========================================================

    def _rebuild_obs_noise_vec(self, level: float):
        """
        Build observation noise vector scaled by curriculum level.

        Noise formula: component_value * obs_scale * lerp(0, obs_noise_level_max, level)
        This matches official legged_gym: noise_scales.X * noise_level * obs_scales.X

        With obs_noise_level_max=1.0 and level=1.0:
          ang_vel: 0.2 * 0.25 * 1.0 = 0.05 in obs space -> 0.2 rad/s raw  (matches official)
          dof_vel: 1.5 * 0.05 * 1.0 = 0.075 in obs space -> 1.5 rad/s raw (matches official)
        """
        if self.obs_noise_components is None or self.obs_noise_level_max <= 0.0:
            self.obs_noise_vec = None
            return

        noise_level = _lerp(0.0, self.obs_noise_level_max, level)
        nc = self.obs_noise_components

        nv = torch.zeros(self.num_obs, device=gs.device, dtype=gs.tc_float)
        nv[0:3] = nc.get("ang_vel", 0.0) * self.obs_scales["ang_vel"] * noise_level
        nv[3:6] = nc.get("gravity", 0.0) * noise_level
        nv[6:9] = 0.0  # commands — no noise
        nv[9:21] = nc.get("dof_pos", 0.0) * self.obs_scales["dof_pos"] * noise_level
        nv[21:33] = nc.get("dof_vel", 0.0) * self.obs_scales["dof_vel"] * noise_level
        nv[33:45] = 0.0  # actions — no noise
        self.obs_noise_vec = nv

    def _apply_curriculum_level(self, force: bool = False):
        """Update step-time knobs (noise, pushes, commands) from current curriculum level."""
        if not self.curriculum.enabled and not force:
            return

        lvl = float(self.curriculum.level) if self.curriculum.enabled else 1.0

        # ---- Noise ramps linearly ----
        self._obs_noise_level_cur = _lerp(0.0, self.obs_noise_level_max, lvl)
        self._action_noise_std_cur = _lerp(0.0, self.action_noise_std_max, lvl)
        self._rebuild_obs_noise_vec(lvl)

        # ---- Push ramps after push_start ----
        if self.push_force_range_hard is None:
            self._push_enable_cur = False
            self._push_force_range_cur = [0.0, 0.0]
            self._push_interval = 10**9
        else:
            if lvl < self.push_start:
                self._push_enable_cur = False
                self._push_force_range_cur = [0.0, 0.0]
                self._push_interval = int(self.push_interval_s_easy / self.dt)
            else:
                s = (lvl - self.push_start) / max(1e-6, (1.0 - self.push_start))
                s = _clamp01(s)
                low_h, high_h = float(self.push_force_range_hard[0]), float(self.push_force_range_hard[1])
                self._push_force_range_cur = [low_h * s, high_h * s]
                interval_s = _lerp(self.push_interval_s_easy, self.push_interval_s_hard, s)
                self._push_interval = max(1, int(interval_s / self.dt))
                self._push_enable_cur = True

        # ---- Command curriculum ----
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

        # ---- Expose in infos ----
        self.extras.setdefault("curriculum", {})
        self.extras["curriculum"].update(self.curriculum.state_dict())
        self.extras["curriculum"].update(
            {
                "obs_noise_level_cur": float(self._obs_noise_level_cur),
                "action_noise_std_cur": float(self._action_noise_std_cur),
                "push_enable": bool(self._push_enable_cur),
                "push_force_range_cur": list(self._push_force_range_cur),
                "push_interval_steps": int(self._push_interval),
                "cmd_ranges": {k: list(v) for k, v in self._cmd_cur_ranges.items()},
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
    # Domain randomisation — per episode (called at reset)
    # ==========================================================

    def _randomize_friction(self, t_sample: float):
        if "friction_range" not in self.env_cfg:
            return
        mu_range = _lerp_range(self.friction_easy, self.friction_hard, t_sample)
        mu = float(gs_rand_float(mu_range[0], mu_range[1], (1,), gs.device).item())
        self.ground.set_friction(mu)
        self.robot.set_friction(mu)
        self.extras.setdefault("domain_randomization", {})
        self.extras["domain_randomization"]["friction"] = mu

    def _randomize_kp_kd(self, t_sample: float):
        if "kp_range" not in self.env_cfg:
            return
        kp_range = _lerp_range(self.kp_easy, self.kp_hard, t_sample)
        kd_range = _lerp_range(self.kd_easy, self.kd_hard, t_sample)
        kp = float(gs_rand_float(kp_range[0], kp_range[1], (1,), gs.device).item())
        kd = float(gs_rand_float(kd_range[0], kd_range[1], (1,), gs.device).item())
        self.robot.set_dofs_kp([kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([kd] * self.num_actions, self.motors_dof_idx)
        self.extras.setdefault("domain_randomization", {})
        self.extras["domain_randomization"]["kp"] = kp
        self.extras["domain_randomization"]["kd"] = kd

    def _randomize_mass(self, t_sample: float):
        """Mass/CoM DR with curriculum scaling."""
        self.extras.setdefault("domain_randomization", {})

        if "mass_shift_range" in self.env_cfg:
            ms_range = _lerp_range(self.mass_easy, self.mass_hard, t_sample)
            shift = float(gs_rand_float(ms_range[0], ms_range[1], (1,), gs.device).item())
            self.robot.set_mass_shift([shift], [0])
            self.extras["domain_randomization"]["mass_shift"] = shift

        if "com_shift_range" in self.env_cfg:
            cs_range = _lerp_range(self.com_easy, self.com_hard, t_sample)
            dx = float(gs_rand_float(cs_range[0], cs_range[1], (1,), gs.device).item())
            dy = float(gs_rand_float(cs_range[0], cs_range[1], (1,), gs.device).item())
            dz = float(gs_rand_float(cs_range[0], cs_range[1], (1,), gs.device).item())
            self.robot.set_COM_shift([[dx, dy, dz]], [0])
            self.extras["domain_randomization"]["com_shift"] = [dx, dy, dz]

    # ==========================================================
    # Domain randomisation — per step
    # ==========================================================

    def _apply_push(self):
        if self.push_force_range_hard is None:
            return
        if not self._push_enable_cur:
            return

        force = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        if self._push_counter % self._push_interval == 0:
            low, high = self._push_force_range_cur
            force[:, 0] = gs_rand_float(low, high, (self.num_envs,), self.device)
            force[:, 1] = gs_rand_float(low, high, (self.num_envs,), self.device)

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

    def _resample_commands(self, envs_idx):
        n = len(envs_idx)
        if n == 0:
            return

        # Zero all commands first
        self.commands[envs_idx] = 0.0

        # Randomly pick which single command each env gets: 0=Vx, 1=Vy, 2=yaw
        choice = torch.randint(0, 3, (n,), device=gs.device)

        vx_mask = (choice == 0)
        vy_mask = (choice == 1)
        yaw_mask = (choice == 2)

        if vx_mask.any():
            self.commands[envs_idx[vx_mask], 0] = gs_rand_float(
                self._cmd_cur_ranges["lin_vel_x"][0],
                self._cmd_cur_ranges["lin_vel_x"][1],
                (vx_mask.sum(),), gs.device,
            )

        if vy_mask.any():
            self.commands[envs_idx[vy_mask], 1] = gs_rand_float(
                self._cmd_cur_ranges["lin_vel_y"][0],
                self._cmd_cur_ranges["lin_vel_y"][1],
                (vy_mask.sum(),), gs.device,
            )

        if yaw_mask.any():
            self.commands[envs_idx[yaw_mask], 2] = gs_rand_float(
                self._cmd_cur_ranges["ang_vel"][0],
                self._cmd_cur_ranges["ang_vel"][1],
                (yaw_mask.sum(),), gs.device,
            )

        # Standing envs override
        if self._standing_env_indices is not None and len(self._standing_env_indices) > 0:
            standing_mask = torch.isin(envs_idx, self._standing_env_indices)
            if standing_mask.any():
                self.commands[envs_idx[standing_mask], :] = 0.0

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        if self._action_noise_std_cur > 0.0:
            target_dof_pos = target_dof_pos + torch.randn_like(target_dof_pos) * self._action_noise_std_cur

        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        self._apply_push()
        self.scene.step()

        self.episode_length_buf += 1

        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # ---- Update foot contacts (before rewards) ----
        if self._has_foot_contact_rewards:
            self._update_foot_contacts()

        # ---- Command resampling ----
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # ---- Termination ----
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 2]) > self.env_cfg["termination_if_z_vel_greater_than"]
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 1]) > self.env_cfg["termination_if_y_vel_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # ---- Rewards ----
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ---- Observations ----
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis=-1,
        )

        self._add_obs_noise()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self._maybe_update_curriculum_on_reset(envs_idx)

        t_sample = self.curriculum.sample_level()

        # DR applied per episode
        self._randomize_friction(t_sample)
        self._randomize_kp_kd(t_sample)
        self._randomize_mass(t_sample)  # FIX: now uses curriculum level

        # Reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        if "init_pos_z_range" in self.env_cfg:
            low, high = self.env_cfg["init_pos_z_range"]
            self.base_pos[envs_idx, 2] = gs_rand_float(low, high, (len(envs_idx),), gs.device)

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

        # Reset foot contact tracking
        if self._has_foot_contact_rewards:
            self._feet_air_time[envs_idx] = 0.0
            self._foot_contact[envs_idx] = False
            self._last_foot_contact[envs_idx] = False

        # Log episode summaries
        self.extras["episode"] = {}
        ep_steps = self.episode_length_buf[envs_idx].to(dtype=gs.tc_float).clamp_min(1.0)
        ep_seconds = ep_steps * self.dt
        for key in self.episode_sums.keys():
            per_sec = (self.episode_sums[key][envs_idx] / ep_seconds).mean().item()
            self.extras["episode"]["rew_" + key] = per_sec
            self.episode_sums[key][envs_idx] = 0.0

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
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_orientation_penalty(self):
        """Penalize projected gravity xy components (body tilt)."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_stand_still(self):
        still_mask = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * still_mask

    # ---- NEW: feet_air_time (official-style gait reward) ----
    def _reward_feet_air_time(self):
        """
        Reward proper swing/stance timing. Encourages each foot to spend
        ~target seconds in the air per stride. Only active when moving.

        Requires foot contact tracking (_init_foot_contact_tracking).
        """
        contact = self._foot_contact  # (num_envs, n_feet)

        # Detect first contact after air phase
        first_contact = (self._feet_air_time > 0.0) & contact

        # Increment air time; reset on contact
        self._feet_air_time += self.dt
        self._feet_air_time *= ~contact  # zero when foot is touching ground

        target = self.reward_cfg.get("feet_air_time_target", 0.5)
        rew = torch.sum((self._feet_air_time - target) * first_contact.float(), dim=1)

        # Only reward when robot should be moving
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return rew * moving

    # ---- NEW: energy (penalize power consumption) ----
    def _reward_energy(self):
        """
        Penalize mechanical power: sum(|torque * joint_velocity|).
        Uses Genesis control force API.
        """
        torques = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return torch.sum(torch.abs(torques * self.dof_vel), dim=1)

    def _reward_torque_load(self):
        tau = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return torch.sum(torch.abs(tau), dim=1)

    # ---- Existing rewards (unchanged) ----
    def _reward_jump_impulse(self):
        vz = self.base_lin_vel[:, 2]
        z = self.base_pos[:, 2]
        gate = (z < 0.50).float()
        return gate * torch.clamp(vz, min=0.0)

    def _reward_jump_apex(self):
        target = self.reward_cfg["jump_apex_height"]
        sigma = self.reward_cfg.get("jump_apex_sigma", 0.05)
        z = self.base_pos[:, 2]
        return torch.exp(-torch.square((z - target) / sigma))

    def _reward_xy_stability(self):
        v = self.robot.get_vel()
        return -(v[:, 0] ** 2 + v[:, 1] ** 2)

    def _reward_orientation(self):
        return -self.projected_gravity[:, 2]

    def _reward_no_shake(self):
        return -torch.sum(self.base_ang_vel ** 2, dim=1) / 1.0

    def _reward_crouch(self):
        z = self.base_pos[:, 2]
        return (z < 0.25).float()

    def _reward_crouch_2(self):
        z = self.base_pos[:, 2]
        in_band = (z <= 0.30) & (z >= 0.20)
        return in_band.float()

    def _reward_ground_penalty(self):
        z = self.base_pos[:, 2]
        z_safe = 0.15
        z_min = 0.05
        x = (z_safe - z) / (z_safe - z_min)
        x = torch.clamp(x, 0.0, 1.0)
        return -(x ** 2)

    def _reward_crouch_target(self):
        z = self.base_pos[:, 2]
        z_t = 0.15
        sigma = 0.03
        return torch.exp(-((z - z_t) / sigma) ** 2)

    def _reward_no_fall(self):
        vz = self.base_lin_vel[:, 2]
        thresh = 0.5
        downward = torch.clamp(-vz - thresh, min=0.0)
        return -(downward ** 2)

    def _reward_y_stability(self):
        v = self.robot.get_vel()
        return -(v[:, 1] ** 2)

    def _reward_crouch_progress(self):
        z = self.base_pos[:, 2]
        return torch.clamp(0.35 - z, min=0.0)

    def _reward_crouch_speed(self):
        vz = self.base_lin_vel[:, 2]
        return -(vz ** 2)