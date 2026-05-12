from __future__ import annotations

import torch
from torch import BoolTensor, FloatTensor, LongTensor

from src.corruption.categorical import _NUM_SPECIAL, _sample_random_tokens


class DiffusionSchedule:
    """Precompute a linear DDPM-style schedule indexed by timesteps 1..T."""

    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("need 0 < beta_start <= beta_end < 1")

        self.num_steps = int(num_steps)
        betas = torch.linspace(float(beta_start), float(beta_end), self.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Index 0 is the clean state. Training samples t in [1, num_steps].
        self.betas = torch.cat([torch.zeros(1), betas])
        self.alphas = torch.cat([torch.ones(1), alphas])
        self.alpha_bars = torch.cat([torch.ones(1), alpha_bars])

    def alpha_bar(self, timesteps: LongTensor, dtype: torch.dtype | None = None) -> FloatTensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be rank-1 [B]")
        if timesteps.numel() and (
            int(timesteps.min().item()) < 0 or int(timesteps.max().item()) > self.num_steps
        ):
            raise ValueError(f"timesteps must be in [0, {self.num_steps}]")
        values = self.alpha_bars.to(device=timesteps.device)[timesteps]
        return values.to(dtype=dtype) if dtype is not None else values


def _expand_batch_values(values: FloatTensor, ndim: int) -> FloatTensor:
    while values.ndim < ndim:
        values = values.unsqueeze(-1)
    return values


def _make_valid_mask(attention_mask: BoolTensor, ndim: int) -> BoolTensor:
    mask = attention_mask
    while mask.ndim < ndim:
        mask = mask.unsqueeze(-1)
    return mask


class DiffusionCorruptionPipeline:
    """Create multi-step noised batches for fixed-length diffusion denoising.

    The returned structure intentionally mirrors CorruptionPipeline:
    (corrupted_batch, targets, masks). The difference is that loss masks cover
    all valid positions, not only positions changed by the sampled noising step.
    """

    def __init__(
        self,
        config: dict,
        vocab_sizes: dict | None = None,
    ) -> None:
        self._cfg = config
        self._vocab_sizes = vocab_sizes or {}

        d_cfg = config.get("diffusion", {})
        self.schedule = DiffusionSchedule(
            num_steps=int(d_cfg.get("num_steps", 100)),
            beta_start=float(d_cfg.get("beta_start", 1e-4)),
            beta_end=float(d_cfg.get("beta_end", 2e-2)),
        )
        timestep_sampling = d_cfg.get("timestep_sampling", "uniform")
        if timestep_sampling != "uniform":
            raise ValueError("Only timestep_sampling='uniform' is supported")
        self.timestep_sampling = timestep_sampling

        self.discrete_mask_fraction = float(d_cfg.get("discrete_mask_fraction", 0.80))
        if not 0.0 <= self.discrete_mask_fraction <= 1.0:
            raise ValueError("discrete_mask_fraction must be in [0, 1]")

        d3pm_cfg = config.get("d3pm", {})
        self.d3pm_enabled = bool(d3pm_cfg.get("enabled", False))
        self.d3pm_transition = d3pm_cfg.get("transition", "absorbing_uniform")
        if self.d3pm_enabled and self.d3pm_transition != "absorbing_uniform":
            raise ValueError("Only d3pm.transition='absorbing_uniform' is supported")
        self.d3pm_apply_to = set(d3pm_cfg.get("apply_to", ["event_type"]))
        self.d3pm_event_type_enabled = self.d3pm_enabled and "event_type" in self.d3pm_apply_to

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> LongTensor:
        return torch.randint(1, self.schedule.num_steps + 1, (batch_size,), device=device)

    def _corrupt_event_type(
        self,
        event_type: LongTensor,
        attention_mask: BoolTensor,
        alpha_bar: FloatTensor,
    ) -> LongTensor:
        vocab_size = int(self._vocab_sizes.get("event_type", 0))
        corrupted = event_type.clone()
        noise_prob = _expand_batch_values(1.0 - alpha_bar, event_type.ndim)

        selected = (torch.rand_like(event_type, dtype=torch.float32) < noise_prob) & attention_mask
        if not bool(selected.any()):
            return torch.where(attention_mask, corrupted, torch.zeros_like(corrupted))

        as_mask = (torch.rand_like(event_type, dtype=torch.float32) < self.discrete_mask_fraction)
        is_mask = selected & as_mask
        is_random = selected & ~as_mask

        if bool(is_random.any()) and vocab_size > _NUM_SPECIAL + 1:
            random_tokens = _sample_random_tokens(event_type, vocab_size)
            corrupted = torch.where(is_random, random_tokens, corrupted)
        else:
            is_mask = is_mask | is_random

        corrupted = torch.where(is_mask, torch.full_like(corrupted, 2), corrupted)
        return torch.where(attention_mask, corrupted, torch.zeros_like(corrupted))

    def _corrupt_event_type_d3pm_pair(
        self,
        event_type: LongTensor,
        attention_mask: BoolTensor,
        timesteps: LongTensor,
    ) -> tuple[LongTensor, LongTensor]:
        """Sample adjacent discrete states x_t and x_{t-1} for the D3PM auxiliary."""
        prev_timesteps = torch.clamp(timesteps - 1, min=0)
        alpha_bar_prev = self.schedule.alpha_bar(prev_timesteps, dtype=torch.float32)
        x_prev = self._corrupt_event_type(event_type, attention_mask, alpha_bar_prev)

        alpha_step = self.schedule.alphas.to(device=timesteps.device)[timesteps].to(torch.float32)
        x_t = self._corrupt_event_type(x_prev, attention_mask, alpha_step)
        return x_t, torch.where(attention_mask, x_prev, torch.zeros_like(x_prev))

    def _corrupt_categorical_features(
        self,
        cat_features: LongTensor,
        attention_mask: BoolTensor,
        alpha_bar: FloatTensor,
    ) -> LongTensor:
        B, L, n_cat = cat_features.shape
        corrupted = cat_features.clone()
        noise_prob = _expand_batch_values(1.0 - alpha_bar, cat_features.ndim)
        valid = attention_mask.unsqueeze(-1).expand(B, L, n_cat)

        selected = (torch.rand_like(cat_features, dtype=torch.float32) < noise_prob) & valid
        if not bool(selected.any()):
            return torch.where(valid, corrupted, torch.zeros_like(corrupted))

        as_mask = (torch.rand_like(cat_features, dtype=torch.float32) < self.discrete_mask_fraction)
        is_mask = selected & as_mask
        is_random = selected & ~as_mask

        vocab_sizes = self._vocab_sizes.get("cat_features") or []
        for j in range(n_cat):
            col_random = is_random[:, :, j]
            if not bool(col_random.any()):
                continue
            vocab_size = int(vocab_sizes[j]) if j < len(vocab_sizes) else 0
            if vocab_size > _NUM_SPECIAL + 1:
                random_tokens = _sample_random_tokens(cat_features[:, :, j], vocab_size)
                corrupted[:, :, j] = torch.where(
                    col_random, random_tokens, corrupted[:, :, j]
                )
            else:
                is_mask[:, :, j] = is_mask[:, :, j] | col_random

        corrupted = torch.where(is_mask, torch.full_like(corrupted, 3), corrupted)
        return torch.where(valid, corrupted, torch.zeros_like(corrupted))

    def _corrupt_continuous(
        self,
        values: FloatTensor,
        attention_mask: BoolTensor,
        alpha_bar: FloatTensor,
    ) -> tuple[FloatTensor, FloatTensor]:
        valid = _make_valid_mask(attention_mask, values.ndim)
        alpha = _expand_batch_values(alpha_bar.to(dtype=values.dtype), values.ndim)
        noise_scale = torch.sqrt(torch.clamp(1.0 - alpha, min=0.0))
        clean_scale = torch.sqrt(torch.clamp(alpha, min=0.0))
        eps = torch.randn_like(values)
        corrupted = clean_scale * values + noise_scale * eps
        corrupted = torch.where(valid, corrupted, values)
        eps = torch.where(valid, eps, torch.zeros_like(eps))
        return corrupted, eps

    def __call__(self, clean_batch: dict) -> tuple[dict, dict, dict]:
        attention_mask: BoolTensor = clean_batch["attention_mask"]
        batch = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in clean_batch.items()
        }

        B = attention_mask.shape[0]
        timesteps = self._sample_timesteps(B, attention_mask.device)
        alpha_bar = self.schedule.alpha_bar(timesteps, dtype=torch.float32)
        batch["diffusion_t"] = timesteps

        targets: dict = {}
        masks: dict = {"event_level": torch.zeros_like(attention_mask)}

        if "event_type" in clean_batch:
            targets["event_type"] = clean_batch["event_type"].clone()
            if self.d3pm_event_type_enabled:
                event_type_t, event_type_prev = self._corrupt_event_type_d3pm_pair(
                    clean_batch["event_type"], attention_mask, timesteps
                )
                batch["event_type"] = event_type_t
                targets["d3pm_event_type_prev"] = event_type_prev
                masks["d3pm_event_type_prev"] = attention_mask.clone()
            else:
                batch["event_type"] = self._corrupt_event_type(
                    clean_batch["event_type"], attention_mask, alpha_bar
                )
            masks["event_type"] = attention_mask.clone()

        if "cat_features" in clean_batch:
            targets["cat_features"] = clean_batch["cat_features"].clone()
            batch["cat_features"] = self._corrupt_categorical_features(
                clean_batch["cat_features"], attention_mask, alpha_bar
            )
            masks["cat_features"] = attention_mask.unsqueeze(-1).expand_as(
                clean_batch["cat_features"]
            ).clone()

        if "time_delta" in clean_batch:
            targets["time_delta"] = clean_batch["time_delta"].clone()
            corrupted_time, eps_time = self._corrupt_continuous(
                clean_batch["time_delta"], attention_mask, alpha_bar
            )
            batch["time_delta"] = corrupted_time
            targets["time_delta_eps"] = eps_time
            masks["time_delta"] = attention_mask.clone()
            masks["time_delta_eps"] = attention_mask.clone()

        if "num_features" in clean_batch:
            targets["num_features"] = clean_batch["num_features"].clone()
            corrupted_num, eps_num = self._corrupt_continuous(
                clean_batch["num_features"], attention_mask, alpha_bar
            )
            batch["num_features"] = corrupted_num
            targets["num_features_eps"] = eps_num
            masks["num_features"] = attention_mask.clone()
            masks["num_features_eps"] = attention_mask.clone()

        return batch, targets, masks


def _sample_prefix_lengths(
    attention_mask: BoolTensor,
    config: dict,
    *,
    suffix_len: int,
) -> LongTensor:
    g_cfg = config.get("generation", {})
    min_ratio = float(g_cfg.get("prefix_min_ratio", 0.50))
    max_ratio = float(g_cfg.get("prefix_max_ratio", 0.80))
    if not 0.0 <= min_ratio <= max_ratio <= 1.0:
        raise ValueError("generation prefix ratios must satisfy 0 <= min <= max <= 1")

    lengths = attention_mask.sum(dim=1).long()
    prefix_lengths = torch.zeros_like(lengths)
    for i, n_raw in enumerate(lengths.tolist()):
        n = int(n_raw)
        if n <= 1:
            prefix_lengths[i] = 0
            continue
        cut_low = max(1, int(torch.floor(torch.tensor(float(n) * min_ratio)).item()))
        cut_high = min(n - 1, int(torch.floor(torch.tensor(float(n) * max_ratio)).item()))
        if cut_high < cut_low:
            cut_low = max(1, n - min(max(1, suffix_len), n - 1))
            cut_high = n - 1
        prefix_lengths[i] = torch.randint(
            cut_low,
            cut_high + 1,
            (1,),
            device=attention_mask.device,
        )
    return prefix_lengths


def _prefix_suffix_masks(
    attention_mask: BoolTensor,
    prefix_lengths: LongTensor,
    suffix_len: int,
) -> tuple[BoolTensor, BoolTensor]:
    B, L = attention_mask.shape
    positions = torch.arange(L, device=attention_mask.device).unsqueeze(0).expand(B, L)
    prefix_end = prefix_lengths.unsqueeze(1)
    suffix_end = (prefix_lengths + int(suffix_len)).unsqueeze(1)
    prefix_mask = (positions < prefix_end) & attention_mask
    suffix_mask = (positions >= prefix_end) & (positions < suffix_end) & attention_mask
    return prefix_mask, suffix_mask


class ConditionalSuffixDiffusionPipeline(DiffusionCorruptionPipeline):
    """Noise only a future suffix while leaving the conditioning prefix clean."""

    def __init__(
        self,
        config: dict,
        vocab_sizes: dict | None = None,
    ) -> None:
        super().__init__(config=config, vocab_sizes=vocab_sizes)
        g_cfg = config.get("generation", {})
        self.suffix_len = int(g_cfg.get("suffix_len", 16))
        if self.suffix_len < 1:
            raise ValueError("generation.suffix_len must be >= 1")

    def __call__(self, clean_batch: dict) -> tuple[dict, dict, dict]:
        attention_mask: BoolTensor = clean_batch["attention_mask"]
        batch = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in clean_batch.items()
        }

        B = attention_mask.shape[0]
        timesteps = self._sample_timesteps(B, attention_mask.device)
        alpha_bar = self.schedule.alpha_bar(timesteps, dtype=torch.float32)
        batch["diffusion_t"] = timesteps

        if "forecast_cut" in clean_batch:
            prefix_lengths = clean_batch["forecast_cut"].to(
                device=attention_mask.device,
                dtype=torch.long,
            )
            prefix_lengths = torch.clamp(
                prefix_lengths,
                min=1,
                max=max(1, attention_mask.shape[1] - 1),
            )
            prefix_lengths = torch.minimum(prefix_lengths, attention_mask.sum(dim=1).long())
        else:
            prefix_lengths = _sample_prefix_lengths(
                attention_mask,
                self._cfg,
                suffix_len=self.suffix_len,
            )
        prefix_mask, suffix_mask = _prefix_suffix_masks(
            attention_mask,
            prefix_lengths,
            self.suffix_len,
        )
        effective_attention = prefix_mask | suffix_mask
        batch["attention_mask"] = effective_attention

        targets: dict = {}
        masks: dict = {
            "event_level": torch.zeros_like(attention_mask),
            "generation_prefix": prefix_mask,
            "generation_suffix": suffix_mask,
        }

        if "event_type" in clean_batch:
            targets["event_type"] = clean_batch["event_type"].clone()
            if self.d3pm_event_type_enabled:
                corrupted_suffix, event_type_prev = self._corrupt_event_type_d3pm_pair(
                    clean_batch["event_type"], suffix_mask, timesteps
                )
                targets["d3pm_event_type_prev"] = event_type_prev
                masks["d3pm_event_type_prev"] = suffix_mask.clone()
            else:
                corrupted_suffix = self._corrupt_event_type(
                    clean_batch["event_type"], suffix_mask, alpha_bar
                )
            event_type = torch.where(suffix_mask, corrupted_suffix, clean_batch["event_type"])
            batch["event_type"] = torch.where(
                effective_attention,
                event_type,
                torch.zeros_like(event_type),
            )
            masks["event_type"] = suffix_mask.clone()

        if "cat_features" in clean_batch:
            targets["cat_features"] = clean_batch["cat_features"].clone()
            corrupted_suffix = self._corrupt_categorical_features(
                clean_batch["cat_features"], suffix_mask, alpha_bar
            )
            suffix_mask_exp = suffix_mask.unsqueeze(-1).expand_as(clean_batch["cat_features"])
            effective_exp = effective_attention.unsqueeze(-1).expand_as(clean_batch["cat_features"])
            cat_features = torch.where(
                suffix_mask_exp,
                corrupted_suffix,
                clean_batch["cat_features"],
            )
            batch["cat_features"] = torch.where(
                effective_exp,
                cat_features,
                torch.zeros_like(cat_features),
            )
            masks["cat_features"] = suffix_mask_exp.clone()

        if "time_delta" in clean_batch:
            targets["time_delta"] = clean_batch["time_delta"].clone()
            corrupted_time, eps_time = self._corrupt_continuous(
                clean_batch["time_delta"], suffix_mask, alpha_bar
            )
            batch["time_delta"] = torch.where(
                effective_attention,
                torch.where(suffix_mask, corrupted_time, clean_batch["time_delta"]),
                torch.zeros_like(clean_batch["time_delta"]),
            )
            targets["time_delta_eps"] = eps_time
            masks["time_delta"] = suffix_mask.clone()
            masks["time_delta_eps"] = suffix_mask.clone()

        if "num_features" in clean_batch:
            targets["num_features"] = clean_batch["num_features"].clone()
            corrupted_num, eps_num = self._corrupt_continuous(
                clean_batch["num_features"], suffix_mask, alpha_bar
            )
            suffix_mask_exp = suffix_mask.unsqueeze(-1).expand_as(clean_batch["num_features"])
            effective_exp = effective_attention.unsqueeze(-1).expand_as(clean_batch["num_features"])
            num_features = torch.where(
                suffix_mask_exp,
                corrupted_num,
                clean_batch["num_features"],
            )
            batch["num_features"] = torch.where(
                effective_exp,
                num_features,
                torch.zeros_like(num_features),
            )
            targets["num_features_eps"] = eps_num
            masks["num_features"] = suffix_mask.clone()
            masks["num_features_eps"] = suffix_mask.clone()

        return batch, targets, masks
