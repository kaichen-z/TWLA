"""Configuration of the world head: adapter, SANA, DC-AE and the world loss."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ..config_validation import REQUIRED, require_parameters
except ImportError:   # imported with T-WLA_foundation_model itself on sys.path
    from config_validation import REQUIRED, require_parameters


@dataclass
class AdapterConfig:
    """The adapter from image and memory tokens to SANA's caption input, trained from scratch.

    `out_dim` is the width of SANA's text encoder (Gemma2-2B), so the pretrained caption
    projection (2304 -> 1152) is reused.
    """

    image_tokens: int = 64        # E_t, one per SigLIP patch after pixel shuffle
    memory_tokens: int = 8        # m_f, the decoder's last `num_slots` positions
    in_dim: int = 960             # backbone width (VLMConfig.hidden_size)
    out_dim: int = 2304           # SANA `caption_channels`

    # Horizon tags carried by the adapter; above 1 only with horizon_inject="condition".
    num_horizons: int = 1

    norm_eps: float = 1e-6        # eps of the two input RMSNorms

    proj_bias: bool = True

    # RMS-normalise each group before concatenation: image and memory tokens arrive at
    # very different scales. False is for ablation.
    norm_first: bool = True

    def __post_init__(self) -> None:
        if self.in_dim <= 0 or self.out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        if self.image_tokens <= 0 or self.memory_tokens <= 0:
            raise ValueError("image_tokens and memory_tokens must be positive")
        if self.num_horizons <= 0:
            raise ValueError("num_horizons must be positive")

    @property
    def condition_tokens(self) -> int:
        """Length of the condition sequence SANA sees as its caption (64 + 8 = 72)."""
        return self.image_tokens + self.memory_tokens

    @property
    def num_parameters(self) -> int:
        """Closed-form parameter count: 2,217,984 with the defaults."""
        norms = 2 * self.in_dim
        types = 2 * self.in_dim
        proj = self.in_dim * self.out_dim + (self.out_dim if self.proj_bias else 0)
        horizons = (self.num_horizons - 1) * self.out_dim
        return norms + types + proj + horizons


@dataclass
class SanaConfig:
    """The 4-block SANA that predicts the future latent, initialised from four blocks of
    the 28-layer SANA-600M. Architecture fields follow that checkpoint's config."""

    # Pretrained weights. The DC-AE is loaded from the same repo at the same revision.
    source_repo: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    source_revision: str | None = "83d7a190bfd1fd070570a793d2dab5c7a3231b9d"
    source_block_indices: tuple[int, ...] = (0, 9, 18, 27)   # copied into blocks 0..3
    source_num_layers: int = 28             # for validating the indices

    # Architecture.
    num_layers: int = 4
    in_channels: int = 32
    out_channels: int = 32
    num_attention_heads: int = 36           # self-attn: 36 x 32 = 1152
    attention_head_dim: int = 32
    num_cross_attention_heads: int = 16     # cross-attn: 16 x 72 = 1152
    cross_attention_head_dim: int = 72
    cross_attention_dim: int = 1152
    caption_channels: int = 2304            # Gemma2-2B width; AdapterConfig.out_dim
    mlp_ratio: float = 2.5                  # FFN inner 1152 * 2.5 = 2880
    patch_size: int = 1                     # one latent cell = one token
    sample_size: int = 16                   # 16 x 16 latent grid (512 / 32 from DC-AE)
    norm_elementwise_affine: bool = False   # scale comes from the tau modulation instead
    norm_eps: float = 1e-6
    attention_bias: bool = False
    dropout: float = 0.0

    # tau in [0, 1] is multiplied by this before it reaches SANA's sinusoidal timestep
    # embedding, which is built for diffusers' [0, 1000] range.
    timestep_scale: float = 1000.0

    max_parameters: int = 100_000_000       # checked after the weights are loaded

    def __post_init__(self) -> None:
        self.source_block_indices = tuple(self.source_block_indices)   # JSON gives a list
        if len(self.source_block_indices) != self.num_layers:
            raise ValueError(
                f"source_block_indices has {len(self.source_block_indices)} entries but "
                f"num_layers is {self.num_layers}")
        if any(i < 0 or i >= self.source_num_layers for i in self.source_block_indices):
            raise ValueError(
                f"source_block_indices {self.source_block_indices} out of range for a "
                f"{self.source_num_layers}-layer source")
        if len(set(self.source_block_indices)) != len(self.source_block_indices):
            raise ValueError(f"duplicate source blocks: {self.source_block_indices}")
        if self.num_attention_heads * self.attention_head_dim != self.cross_attention_dim:
            raise ValueError("self-attention heads x head_dim must equal the model width")
        if (self.num_cross_attention_heads * self.cross_attention_head_dim
                != self.cross_attention_dim):
            raise ValueError("cross-attention heads x head_dim must equal the model width")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim      # 1152

    @property
    def num_patches(self) -> int:
        return (self.sample_size // self.patch_size) ** 2              # 256

    def diffusers_kwargs(self) -> dict:
        """The keyword arguments of `SanaTransformer2DModel.__init__`."""
        return dict(
            in_channels=self.in_channels, out_channels=self.out_channels,
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            num_layers=self.num_layers,
            num_cross_attention_heads=self.num_cross_attention_heads,
            cross_attention_head_dim=self.cross_attention_head_dim,
            cross_attention_dim=self.cross_attention_dim,
            caption_channels=self.caption_channels,
            mlp_ratio=self.mlp_ratio, dropout=self.dropout,
            attention_bias=self.attention_bias, sample_size=self.sample_size,
            patch_size=self.patch_size,
            norm_elementwise_affine=self.norm_elementwise_affine,
            norm_eps=self.norm_eps,
        )


@dataclass
class DCAEConfig:
    """The frozen DC-AE that encodes a target frame into the latent SANA predicts."""

    source_repo: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    subfolder: str = "vae"
    source_revision: str | None = "83d7a190bfd1fd070570a793d2dab5c7a3231b9d"   # same repo, same commit, as SanaConfig

    latent_channels: int = 32
    spatial_compression: int = 32          # 512 / 32 = 16
    image_size: int = 512                  # the model's image canvas

    # From DC-AE's config.json; brings the latents to roughly unit variance.
    scaling_factor: float = 0.41407

    # Whether `dc_ae.encode` multiplies by `scaling_factor`. Must be False if the
    # autoencoder's own `encode()` already scales; `dc_ae.check_scaling` tells which.
    apply_scaling: bool = True

    # Range the std of the latents must fall in for `check_scaling` to pass.
    expected_std_range: tuple[float, float] = (0.5, 2.0)

    def __post_init__(self) -> None:
        self.expected_std_range = tuple(self.expected_std_range)   # JSON gives a list

    @property
    def latent_size(self) -> int:
        return self.image_size // self.spatial_compression      # 16

    @property
    def latent_numel(self) -> int:
        """Elements per latent frame: 32 * 16 * 16 = 8192."""
        return self.latent_channels * self.latent_size ** 2


@dataclass
class WorldLossConfig:
    """`L_world` and the future frames it is computed against.

    The target frames follow from the action chunk: a 16-step chunk at 80 Hz with 8
    executed steps reaches t + 0.1 s and t + 0.2 s, source frames t + 3 and t + 6 at 30 fps.
    """

    # The fields below have no defaults: they are set in the run config.

    # Source-frame offsets of the target frames. Must equal what the action chunk reaches
    # (`validate_against_action`).
    horizon_frames: tuple[int, ...] = REQUIRED

    # How many future anchors the supervised step is guaranteed to have: the window
    # sampler's upper bound is reduced by this much.
    reserve_anchors: int = REQUIRED

    # Per-horizon weights. None = uniform.
    horizon_weights: tuple[float, ...] | None = REQUIRED

    # lambda on `L_total = L_action + lambda * L_world`, piecewise constant in the training
    # step: ((start_step, lambda), ...), ascending, starting at step 0. The first
    # `lam_warmup_steps` steps ramp linearly from 0.
    lam_schedule: tuple[tuple[int, float], ...] = REQUIRED
    lam_warmup_steps: int = REQUIRED

    # Range tau is sampled from. z_tau = (1-tau) z0 + tau eps, u = eps - z0, tau = 1 is the
    # noise end -- the same convention as `TWLA/flow.py`.
    tau_low: float = REQUIRED
    tau_high: float = REQUIRED

    # Per-domain multiplier on lambda, lam_d = lambda(step) * mult_d, applied from step
    # `domain_lam_start`. "off": 1 for every domain. "fixed": taken from
    # `domain_lam_fixed`, unlisted domains 1. "adaptive": follows an EMA of
    # cos(g_world_d, g_action_d) on the memory parameters, measured on training batches.
    domain_lam_mode: str = REQUIRED
    domain_lam_fixed: dict | None = REQUIRED
    domain_lam_start: int = REQUIRED
    domain_lam_ema: float = REQUIRED
    # Adaptive rule. "sign": mult_d = 1 if ema_d >= threshold else domain_lam_min.
    # "linear": mult_d = clip((ema_d - threshold) / scale, domain_lam_min, 1).
    domain_lam_rule: str = REQUIRED
    domain_lam_threshold: float = REQUIRED
    domain_lam_scale: float = REQUIRED
    domain_lam_min: float = REQUIRED

    def __post_init__(self) -> None:
        require_parameters(self, "world.loss.")
        self.lam_schedule = tuple((int(s), float(v)) for s, v in self.lam_schedule)
        starts = [s for s, _ in self.lam_schedule]
        if not starts or starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError(f"lam_schedule must start at step 0 and be strictly ascending, "
                             f"got {self.lam_schedule}")
        if any(v < 0 for _, v in self.lam_schedule) or self.lam_warmup_steps < 0:
            raise ValueError("lam_schedule values and lam_warmup_steps must be non-negative")
        if self.domain_lam_mode not in ("off", "fixed", "adaptive"):
            raise ValueError(f"domain_lam_mode must be off|fixed|adaptive, got {self.domain_lam_mode!r}")
        if self.domain_lam_rule not in ("sign", "linear"):
            raise ValueError(f"domain_lam_rule must be sign|linear, got {self.domain_lam_rule!r}")
        if not 0.0 <= self.domain_lam_ema < 1.0:
            raise ValueError(f"domain_lam_ema must be in [0, 1), got {self.domain_lam_ema}")
        if self.domain_lam_mode == "fixed" and not self.domain_lam_fixed:
            raise ValueError("domain_lam_mode=fixed needs a non-empty domain_lam_fixed mapping")
        if not 0.0 <= self.domain_lam_min <= 1.0:
            raise ValueError(f"domain_lam_min must be in [0, 1], got {self.domain_lam_min}")
        # JSON gives lists; the checks below compare against tuples.
        self.horizon_frames = tuple(self.horizon_frames)
        if self.horizon_weights is not None:
            self.horizon_weights = tuple(self.horizon_weights)

        if not self.horizon_frames:
            raise ValueError("horizon_frames may not be empty")
        if any(h <= 0 for h in self.horizon_frames):
            raise ValueError(f"horizon_frames must all be positive: {self.horizon_frames}")
        if len(set(self.horizon_frames)) != len(self.horizon_frames):
            raise ValueError(f"duplicate horizon_frames: {self.horizon_frames}")
        if list(self.horizon_frames) != sorted(self.horizon_frames):
            raise ValueError(
                f"horizon_frames must be ascending: {self.horizon_frames}. The order is "
                "the horizon embedding's index, so reordering silently relabels them.")
        if self.horizon_weights is not None:
            if len(self.horizon_weights) != len(self.horizon_frames):
                raise ValueError(
                    f"horizon_weights has {len(self.horizon_weights)} entries for "
                    f"{len(self.horizon_frames)} horizons")
            if any(w < 0 for w in self.horizon_weights):
                raise ValueError("horizon_weights must be non-negative")
            if sum(self.horizon_weights) <= 0:
                raise ValueError("horizon_weights must not be all zero")
        if not 0.0 <= self.tau_low < self.tau_high <= 1.0:
            raise ValueError(f"need 0 <= tau_low < tau_high <= 1, got "
                             f"({self.tau_low}, {self.tau_high})")

    @property
    def num_horizons(self) -> int:
        return len(self.horizon_frames)

    def weights(self) -> tuple[float, ...]:
        """Horizon weights normalised to sum to 1."""
        w = self.horizon_weights or (1.0,) * self.num_horizons
        total = float(sum(w))
        return tuple(float(x) / total for x in w)

    def anchor_offsets(self, stride: int) -> tuple[int, ...]:
        """`horizon_frames` in anchor steps. Anchors sit `stride` source frames apart, so
        (3, 6) at stride 3 becomes (1, 2)."""
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        out = []
        for h in self.horizon_frames:
            if h % stride:
                raise ValueError(
                    f"horizon {h} source frames is not a whole number of {stride}-frame "
                    "control steps, so no anchor lands on it")
            out.append(h // stride)
        return tuple(out)

    def validate_against_action(self, action_cfg) -> None:
        """Raise unless `horizon_frames` are the frames the action chunk reaches."""
        a = action_cfg
        span_s = a.chunk_size / a.action_hz                       # chunk duration [s]
        step_s = a.executed_steps / a.action_hz                   # one control period [s]
        n = int(round(span_s / step_s))                           # control periods per chunk
        if abs(n * step_s - span_s) > 1e-9:
            raise ValueError(
                f"chunk span {span_s}s is not a whole number of {step_s}s control "
                f"periods; the target frames would not line up with the chunk")

        expected = []
        for k in range(1, n + 1):
            f = (k * step_s) * a.source_fps
            if abs(f - round(f)) > 1e-9:
                raise ValueError(
                    f"horizon {k * step_s}s is {f} source frames at {a.source_fps} fps, "
                    "not an integer -- the target frame does not exist and would have to "
                    "be interpolated, which the image side has no way to do")
            expected.append(int(round(f)))

        if tuple(expected) != tuple(self.horizon_frames):
            raise ValueError(
                f"horizon_frames {self.horizon_frames} does not match what the action "
                f"config implies: {tuple(expected)} (chunk {a.chunk_size} @ {a.action_hz} "
                f"Hz, executed {a.executed_steps}, source {a.source_fps} fps)")


@dataclass
class WorldConfig:
    """The world head, as one field of `TWLAConfig`. `world = None` builds no head.

    The DC-AE is not a submodule: the training loop calls `dc_ae.encode_targets`, which
    keeps the autoencoder out of the checkpoints.
    """

    adapter: AdapterConfig = None            # type: ignore[assignment]
    sana: SanaConfig = None                  # type: ignore[assignment]
    dcae: DCAEConfig = None                  # type: ignore[assignment]
    loss: WorldLossConfig = None             # type: ignore[assignment]

    # How the horizon index reaches SANA. "modulation": a learned per-horizon vector
    # added to the timestep embedding, as DiT does for class labels. "condition": a
    # per-horizon bias on the 72 condition tokens (ablation).
    horizon_inject: str = "modulation"

    # The fields below have no defaults: they are set in the run config.

    # Stage 1 trains the head alone against `L_world` with everything upstream frozen;
    # stage 2 adds `L_action` and lets gradient into memory.
    freeze_v4: bool = REQUIRED

    # Stage 2 only: keep the world head fixed and let only `m_f` move to satisfy it.
    freeze_head: bool = REQUIRED

    lr_adapter: float = REQUIRED             # learning rate of the adapter
    lr_sana: float = REQUIRED                # learning rate of the SANA blocks

    def __post_init__(self) -> None:
        require_parameters(self, "world.")
        if self.loss is None:
            self.loss = WorldLossConfig()
        if self.adapter is None:
            # The adapter carries the horizon tags only under "condition"; under
            # "modulation" they live in SANA's timestep path.
            self.adapter = AdapterConfig(
                num_horizons=(self.loss.num_horizons
                              if self.horizon_inject == "condition" else 1))
        if self.sana is None:
            self.sana = SanaConfig()
        if self.dcae is None:
            self.dcae = DCAEConfig()

        self.validate()

    def validate(self) -> None:
        """Check every place where two of the sub-configs have to agree."""
        if self.horizon_inject not in ("modulation", "condition"):
            raise ValueError(
                f"horizon_inject must be 'modulation' or 'condition', got "
                f"{self.horizon_inject!r}")
        if self.freeze_v4 and self.freeze_head:
            raise ValueError("freeze_v4 and freeze_head together leave nothing to train")

        if not 0 <= self.loss.reserve_anchors:
            raise ValueError(f"reserve_anchors must be >= 0, got "
                             f"{self.loss.reserve_anchors}")
        if self.loss.reserve_anchors > len(self.loss.horizon_frames):
            raise ValueError(
                f"reserve_anchors={self.loss.reserve_anchors} exceeds the number of "
                f"horizons ({len(self.loss.horizon_frames)}). Reserving past the furthest "
                "target buys no validity and only shortens the supervised range.")

        if self.horizon_inject == "condition":
            if self.adapter.num_horizons != self.loss.num_horizons:
                raise ValueError(
                    f"horizon_inject='condition' needs adapter.num_horizons == "
                    f"{self.loss.num_horizons} (the horizon count), got "
                    f"{self.adapter.num_horizons}. They index the same tag table, so a "
                    "mismatch silently gives two target frames one tag.")
        elif self.adapter.num_horizons != 1:
            raise ValueError(
                f"horizon_inject='modulation' puts the tag in SANA's timestep path, so "
                f"the adapter must carry none (num_horizons=1), got "
                f"{self.adapter.num_horizons}. Two tag tables would be two answers to the "
                "same question, and only one of them is read.")

        if self.sana.in_channels != self.dcae.latent_channels:
            raise ValueError(
                f"sana.in_channels {self.sana.in_channels} != dcae.latent_channels "
                f"{self.dcae.latent_channels}. DC-AE's weights fix its width and the "
                "transformer must match it.")
        if self.sana.out_channels != self.dcae.latent_channels:
            raise ValueError(
                f"sana.out_channels {self.sana.out_channels} != dcae.latent_channels "
                f"{self.dcae.latent_channels}")

        if self.sana.sample_size != self.dcae.latent_size:
            raise ValueError(
                f"sana.sample_size {self.sana.sample_size} != the {self.dcae.latent_size} "
                f"that {self.dcae.image_size}px at {self.dcae.spatial_compression}x "
                "compression produces")

        if self.adapter.out_dim != self.sana.caption_channels:
            raise ValueError(
                f"adapter.out_dim {self.adapter.out_dim} != sana.caption_channels "
                f"{self.sana.caption_channels}. The adapter's whole job is to land on the "
                "width the inherited caption projection expects.")

        if self.sana.source_repo != self.dcae.source_repo:
            raise ValueError(
                f"transformer and autoencoder come from different repos "
                f"({self.sana.source_repo} vs {self.dcae.source_repo}). They were trained "
                "together; pairing halves of two releases is not a supported combination.")

    def validate_against_v4(self, v4_cfg) -> None:
        """Cross-check against the model config the head is attached to."""
        if self.adapter.in_dim != v4_cfg.vlm.hidden_size:
            raise ValueError(
                f"adapter.in_dim {self.adapter.in_dim} != vlm.hidden_size "
                f"{v4_cfg.vlm.hidden_size}; E_t and m_f are both that wide")
        if self.adapter.memory_tokens != v4_cfg.memory.num_slots:
            raise ValueError(
                f"adapter.memory_tokens {self.adapter.memory_tokens} != memory.num_slots "
                f"{v4_cfg.memory.num_slots}; m_f carries exactly num_slots tokens")
        expect_img = v4_cfg.vlm.tokens_per_frame * v4_cfg.vlm.num_cameras
        if self.adapter.image_tokens != expect_img:
            raise ValueError(
                f"adapter.image_tokens {self.adapter.image_tokens} != "
                f"{v4_cfg.vlm.tokens_per_frame} x {v4_cfg.vlm.num_cameras} = {expect_img}")
        if self.dcae.image_size != v4_cfg.vlm.image_size:
            raise ValueError(
                f"dcae.image_size {self.dcae.image_size} != vlm.image_size "
                f"{v4_cfg.vlm.image_size}. The target frame must be on the same canvas as "
                "the conditioning frame, or the head is asked to predict a shifted picture.")
        self.loss.validate_against_action(v4_cfg.action)
