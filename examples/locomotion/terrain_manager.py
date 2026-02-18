"""
TerrainManager — builds a Genesis terrain with mixed difficulty.

Supports TWO modes:
  1. Sub-terrain grid (Isaac Gym style): pass subterrain_types in config
  2. Custom heightmap: pass height_field (numpy array) in config

No terrain curriculum: all tiles are available from step 0.
The robot learns to handle everything through brute-force exposure.

Key responsibilities:
  - Build the terrain morph (before scene.build)
  - Assign random tiles to envs at reset (grid mode) or random positions (heightmap mode)
  - Query ground height for terrain-aware rewards
  - Track spawn origin for boundary resets
"""

import genesis as gs
import numpy as np
import torch


class TerrainManager:
    def __init__(self, terrain_cfg: dict, num_envs: int, device=None):
        self.cfg = terrain_cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.device = device or gs.device
        self.num_envs = num_envs

        if not self.enabled:
            return

        # ---- Detect mode: custom heightmap vs subterrain grid ----
        self.use_custom_heightmap = (
            "height_field" in self.cfg and self.cfg["height_field"] is not None
        )

        self.horizontal_scale = float(self.cfg.get("horizontal_scale", 0.25))
        self.vertical_scale = float(self.cfg.get("vertical_scale", 0.005))
        self.spawn_height_offset = float(self.cfg.get("spawn_height_offset", 0.05))
        self.boundary_margin = float(self.cfg.get("boundary_margin", 1.0))

        if self.use_custom_heightmap:
            self._init_custom_heightmap()
        else:
            self._init_subterrain_grid()

        # Per-env tracking (common to both modes)
        self._env_tile_row = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._env_tile_col = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._env_spawn_pos = torch.zeros(
            num_envs, 3, device=self.device, dtype=gs.tc_float
        )

        # Height field (populated after build, or immediately for custom)
        self._height_field = None
        self._terrain_origin = None

    # ------------------------------------------------------------------
    # Init: custom heightmap mode
    # ------------------------------------------------------------------

    def _init_custom_heightmap(self):
        """Initialise for custom numpy heightmap mode."""
        self._height_field_np = np.array(self.cfg["height_field"], dtype=np.float32)
        hf_rows, hf_cols = self._height_field_np.shape

        # Compute terrain world size from heightmap dimensions
        self._terrain_size_x = hf_rows * self.horizontal_scale
        self._terrain_size_y = hf_cols * self.horizontal_scale

        # Max wander distance from spawn center
        half_size = min(self._terrain_size_x, self._terrain_size_y) / 2.0
        self.max_wander = half_size - self.boundary_margin

        # No subterrain grid in this mode
        self.subterrain_types = None
        self.n_rows = 1
        self.n_cols = 1
        self.subterrain_size = (self._terrain_size_x, self._terrain_size_y)
        self.randomize = False
        self._flat_row_indices = []
        self._tile_centers = None

        hmin = self._height_field_np.min()
        hmax = self._height_field_np.max()
        print("[TerrainManager] Custom heightmap mode")
        print(f"  shape      : {self._height_field_np.shape}")
        print(f"  h_scale    : {self.horizontal_scale} m/px")
        print(f"  v_scale    : {self.vertical_scale}")
        print(f"  raw range  : [{hmin:.4f}, {hmax:.4f}]")
        print(
            f"  world range: [{hmin * self.vertical_scale:.4f}, {hmax * self.vertical_scale:.4f}] m"
        )
        print(
            f"  covers     : {self._terrain_size_x:.1f} x {self._terrain_size_y:.1f} m"
        )
        print(f"  max_wander : {self.max_wander:.1f} m")

    # ------------------------------------------------------------------
    # Init: subterrain grid mode (original)
    # ------------------------------------------------------------------

    def _init_subterrain_grid(self):
        """Initialise for subterrain grid mode (Isaac Gym style)."""
        self.subterrain_types = self.cfg["subterrain_types"]
        self.n_rows = len(self.subterrain_types)
        self.n_cols = len(self.subterrain_types[0])

        self.subterrain_size = tuple(self.cfg.get("subterrain_size", (8.0, 8.0)))
        self.randomize = bool(self.cfg.get("randomize", True))

        # Max distance from tile center before episode is terminated
        self.max_wander = min(self.subterrain_size) / 2.0 - self.boundary_margin

        self._height_field_np = None
        self._flat_row_indices = []
        for r, row in enumerate(self.subterrain_types):
            if all(t == "flat_terrain" for t in row):
                self._flat_row_indices.append(r)

        print(
            f"[TerrainManager] Grid: {self.n_rows} rows x {self.n_cols} cols "
            f"({self.n_rows * self.n_cols} tiles)"
        )
        print(
            f"[TerrainManager] Tile size: {self.subterrain_size}, "
            f"max_wander: {self.max_wander:.1f}m"
        )
        for i, row in enumerate(self.subterrain_types):
            tag = " (FLAT)" if i in self._flat_row_indices else ""
            print(f"  Row {i}{tag}: {row}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_terrain_morph(self):
        """Returns gs.morphs.Terrain morph. Call BEFORE scene.build()."""
        if not self.enabled:
            return None

        if self.use_custom_heightmap:
            return self._build_custom_heightmap_morph()
        else:
            return self._build_subterrain_morph()

    def _build_custom_heightmap_morph(self):
        """Build terrain from custom numpy heightmap."""
        hf = self._height_field_np

        # Origin: center the terrain at world (0, 0)
        self._terrain_origin = np.array(
            [
                -self._terrain_size_x / 2.0,
                -self._terrain_size_y / 2.0,
                0.0,
            ]
        )

        morph = gs.morphs.Terrain(
            height_field=hf,
            pos=tuple(self._terrain_origin),
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            name=self.cfg.get("terrain_name", None),
        )

        # Pre-cache the height field (we already have it)
        self._height_field = hf

        print(
            f"[TerrainManager] Custom heightmap morph created, "
            f"origin=({self._terrain_origin[0]:.1f}, {self._terrain_origin[1]:.1f})"
        )

        return morph

    def _build_subterrain_morph(self):
        """Build terrain from subterrain grid (original path)."""
        total_x = self.n_rows * self.subterrain_size[0]
        total_y = self.n_cols * self.subterrain_size[1]
        self._terrain_origin = np.array([-total_x / 2.0, -total_y / 2.0, 0.0])

        # Precompute tile centers in world coords
        self._tile_centers = torch.zeros(
            (self.n_rows, self.n_cols, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cx = self._terrain_origin[0] + (r + 0.5) * self.subterrain_size[0]
                cy = self._terrain_origin[1] + (c + 0.5) * self.subterrain_size[1]
                self._tile_centers[r, c, 0] = cx
                self._tile_centers[r, c, 1] = cy

        morph_kwargs = dict(
            pos=tuple(self._terrain_origin),
            n_subterrains=(self.n_rows, self.n_cols),
            subterrain_size=self.subterrain_size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            subterrain_types=self.subterrain_types,
            randomize=self.randomize,
            name=self.cfg.get("terrain_name", None),
        )

        # Pass subterrain_parameters if provided
        sub_params = self.cfg.get("subterrain_parameters", None)
        if sub_params is not None:
            morph_kwargs["subterrain_parameters"] = sub_params
            print(f"[TerrainManager] subterrain_parameters: {sub_params}")

        morph = gs.morphs.Terrain(**morph_kwargs)
        return morph

    def post_build(self, terrain_entity):
        """Call AFTER scene.build(). Tries to cache height field."""
        if not self.enabled:
            return

        # Custom heightmap already has the height field cached
        if self.use_custom_heightmap and self._height_field is not None:
            print(
                f"[TerrainManager] Height field already cached (custom mode): "
                f"shape={self._height_field.shape}"
            )
            return

        # Try multiple attribute names that Genesis might use
        hf = None
        for attr in ("height_field", "heightfield", "_height_field", "hf"):
            hf = getattr(terrain_entity, attr, None)
            if hf is not None:
                break

        # Also try going through the morph
        if hf is None:
            morph = getattr(terrain_entity, "morph", None)
            if morph is not None:
                for attr in ("height_field", "heightfield", "_height_field", "hf"):
                    hf = getattr(morph, attr, None)
                    if hf is not None:
                        break

        if hf is not None:
            self._height_field = np.array(hf, dtype=np.float32)
            print(
                f"[TerrainManager] Height field cached: shape={self._height_field.shape}"
            )
        else:
            print("[TerrainManager] WARNING: Could not extract height field.")
            print(
                "  Height queries will return 0. base_height reward will be "
                "approximate on non-flat terrain."
            )
            print(
                "  If terrain-aware rewards are important, inspect the terrain "
                "entity attributes after build to find the height data."
            )

    # ------------------------------------------------------------------
    # Spawn & boundary
    # ------------------------------------------------------------------

    def sample_spawn(self, envs_idx, base_init_z: float = 0.42):
        """
        Assign each resetting env to a spawn position.
        Grid mode: random tile. Heightmap mode: random position on terrain.
        Returns spawn positions (N, 3).
        """
        if not self.enabled:
            return None

        n = len(envs_idx)
        if n == 0:
            return None

        if self.use_custom_heightmap:
            return self._sample_spawn_heightmap(envs_idx, n, base_init_z)
        else:
            return self._sample_spawn_grid(envs_idx, n, base_init_z)

    def _sample_spawn_heightmap(self, envs_idx, n, base_init_z):
        """Spawn at random positions on custom heightmap."""
        spawn_pos = torch.zeros(n, 3, device=self.device, dtype=gs.tc_float)

        # Random XY within safe bounds (away from edges)
        safe_half_x = self._terrain_size_x / 2.0 - self.boundary_margin - 0.5
        safe_half_y = self._terrain_size_y / 2.0 - self.boundary_margin - 0.5

        if safe_half_x > 0:
            spawn_pos[:, 0] = (torch.rand(n, device=self.device) * 2 - 1) * safe_half_x
        if safe_half_y > 0:
            spawn_pos[:, 1] = (torch.rand(n, device=self.device) * 2 - 1) * safe_half_y

        # Ground height at spawn point
        ground_z = self._query_height_batch(spawn_pos[:, 0], spawn_pos[:, 1])
        spawn_pos[:, 2] = ground_z + base_init_z + self.spawn_height_offset

        self._env_spawn_pos[envs_idx] = spawn_pos
        self._env_tile_row[envs_idx] = 0
        self._env_tile_col[envs_idx] = 0

        return spawn_pos

    def _sample_spawn_grid(self, envs_idx, n, base_init_z):
        """Spawn at random tile centers (original grid mode)."""
        # Random row and column — uniform across all tiles
        rows = torch.randint(0, self.n_rows, (n,), device=self.device)
        cols = torch.randint(0, self.n_cols, (n,), device=self.device)

        self._env_tile_row[envs_idx] = rows
        self._env_tile_col[envs_idx] = cols

        # Look up tile centers
        spawn_pos = torch.zeros(n, 3, device=self.device, dtype=gs.tc_float)
        for i in range(n):
            spawn_pos[i] = self._tile_centers[int(rows[i]), int(cols[i])].clone()

        # Random XY jitter within tile (stay away from edges)
        jitter_x = self.subterrain_size[0] / 2.0 - self.boundary_margin - 0.5
        jitter_y = self.subterrain_size[1] / 2.0 - self.boundary_margin - 0.5
        if jitter_x > 0:
            spawn_pos[:, 0] += (torch.rand(n, device=self.device) * 2 - 1) * jitter_x
        if jitter_y > 0:
            spawn_pos[:, 1] += (torch.rand(n, device=self.device) * 2 - 1) * jitter_y

        # Ground height at spawn point
        ground_z = self._query_height_batch(spawn_pos[:, 0], spawn_pos[:, 1])
        spawn_pos[:, 2] = ground_z + base_init_z + self.spawn_height_offset

        self._env_spawn_pos[envs_idx] = spawn_pos

        return spawn_pos

    def check_boundary(self, base_pos_xy):
        """
        Returns bool tensor (num_envs,): True if robot has wandered
        too far from its tile center -> should be terminated.
        """
        if not self.enabled:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        spawn_xy = self._env_spawn_pos[:, :2]
        dist = torch.norm(base_pos_xy - spawn_xy, dim=1)
        return dist > self.max_wander

    # ------------------------------------------------------------------
    # Height queries
    # ------------------------------------------------------------------

    def _query_height_batch(self, world_x, world_y):
        """
        Look up terrain height at world (x, y) coordinates.
        Bilinear interpolation on the height field.
        """
        n = len(world_x)
        if self._height_field is None:
            return torch.zeros(n, device=self.device, dtype=gs.tc_float)

        hf = self._height_field
        hf_rows, hf_cols = hf.shape

        # World -> height field pixel
        lx = (world_x.cpu().numpy() - self._terrain_origin[0]) / self.horizontal_scale
        ly = (world_y.cpu().numpy() - self._terrain_origin[1]) / self.horizontal_scale

        lx = np.clip(lx, 0, hf_rows - 2).astype(np.float64)
        ly = np.clip(ly, 0, hf_cols - 2).astype(np.float64)

        ix = lx.astype(np.int32)
        iy = ly.astype(np.int32)
        fx = lx - ix
        fy = ly - iy

        # Bilinear interpolation
        h00 = hf[ix, iy]
        h10 = hf[np.minimum(ix + 1, hf_rows - 1), iy]
        h01 = hf[ix, np.minimum(iy + 1, hf_cols - 1)]
        h11 = hf[np.minimum(ix + 1, hf_rows - 1), np.minimum(iy + 1, hf_cols - 1)]

        h = (
            h00 * (1 - fx) * (1 - fy)
            + h10 * fx * (1 - fy)
            + h01 * (1 - fx) * fy
            + h11 * fx * fy
        )

        heights = h * self.vertical_scale
        return torch.tensor(heights, device=self.device, dtype=gs.tc_float)

    def get_height_at_robot(self, base_pos_xy):
        """Convenience: (num_envs, 2) -> (num_envs,)"""
        return self._query_height_batch(base_pos_xy[:, 0], base_pos_xy[:, 1])

    # ------------------------------------------------------------------
    # Info for logging / privileged obs
    # ------------------------------------------------------------------

    def get_env_tile_rows(self):
        """Per-env terrain row index."""
        return self._env_tile_row

    def get_terrain_type_string(self, env_idx: int) -> str:
        if self.use_custom_heightmap:
            return "custom_heightmap"
        r = int(self._env_tile_row[env_idx].item())
        c = int(self._env_tile_col[env_idx].item())
        return self.subterrain_types[r][c]
