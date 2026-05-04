import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # Pixel-RALU sampling hyper params
        self.use_ralu = getattr(args, "use_ralu", False)
        self.ralu_f0 = getattr(args, "ralu_f0", 2)
        self.ralu_N = getattr(args, "ralu_N", [10, 4, 8])
        self.ralu_e = getattr(args, "ralu_e", [0.35, 0.55, 1.0])
        self.ralu_up_ratio = getattr(args, "ralu_up_ratio", 0.3)
        self.ralu_hf_noise = getattr(args, "ralu_hf_noise", 0.25)

        self.model_name = args.model
        self.attn_dropout = args.attn_dropout
        self.proj_dropout = args.proj_dropout

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        if self.use_ralu:
            return self.generate_ralu(labels)

        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(
            interval_mask,
            torch.full_like(t, float(self.cfg_scale)),
            torch.ones_like(t),
        )

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    def reset_ralu_cache(self):
        self.__dict__.pop("_ralu_low_net", None)

    @torch.no_grad()
    def _make_ralu_low_net(self):
        cached = self.__dict__.get("_ralu_low_net", None)
        if cached is not None:
            return cached

        assert self.img_size % self.ralu_f0 == 0, "img_size must be divisible by ralu_f0"
        low_size = self.img_size // self.ralu_f0

        low_net = JiT_models[self.model_name](
            input_size=low_size,
            in_channels=3,
            num_classes=self.num_classes,
            attn_drop=self.attn_dropout,
            proj_drop=self.proj_dropout,
        )

        device = next(self.net.parameters()).device
        low_net.to(device)
        low_net.eval()

        src = self.net.state_dict()
        dst = low_net.state_dict()
        for key, value in src.items():
            if key in dst and dst[key].shape == value.shape:
                dst[key].copy_(value)
        low_net.load_state_dict(dst, strict=True)

        # Bypass nn.Module registration so strict load_state_dict stays unchanged.
        self.__dict__["_ralu_low_net"] = low_net
        return low_net

    def _expand_t(self, t, z):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z.device, dtype=z.dtype)
        t = t.to(device=z.device, dtype=z.dtype)

        if t.ndim == 0:
            return t.view(1, 1, 1, 1).expand(z.size(0), 1, 1, 1)
        if t.ndim == 1:
            return t.view(-1, 1, 1, 1)
        return t

    @torch.no_grad()
    def _cfg_v_and_x(self, net, z, t, labels):
        t = self._expand_t(t, z)
        t_flat = t.flatten()

        x_cond = net(z, t_flat, labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        null_labels = torch.full_like(labels, self.num_classes)
        x_uncond = net(z, t_flat, null_labels)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_on = torch.full_like(t, float(self.cfg_scale))
        cfg_off = torch.ones_like(t)
        cfg_scale_interval = torch.where(interval_mask, cfg_on, cfg_off)

        v = v_uncond + cfg_scale_interval * (v_cond - v_uncond)
        x0 = z + (1.0 - t).clamp_min(self.t_eps) * v
        return v, x0

    @torch.no_grad()
    def _ode_step_with_fn(self, forward_fn, z, t, t_next, labels):
        v_t, x0_t = forward_fn(z, t, labels)
        dt = (t_next - t).to(device=z.device, dtype=z.dtype).view(1, 1, 1, 1)
        z_euler = z + dt * v_t

        is_last = bool((t_next >= 1.0 - 1e-6).detach().cpu().item())
        if self.method == "heun" and not is_last:
            v_next, _ = forward_fn(z_euler, t_next, labels)
            z_next = z + dt * 0.5 * (v_t + v_next)
        elif self.method == "euler" or is_last:
            z_next = z_euler
        else:
            raise NotImplementedError(f"Unsupported sampler: {self.method}")

        return z_next, x0_t

    @torch.no_grad()
    def _lift_low_state_to_full(self, z_low, x0_low, t, full_hw):
        t = self._expand_t(t, z_low)
        eps_low = (z_low - t * x0_low) / (1.0 - t).clamp_min(self.t_eps)

        x0_full = F.interpolate(x0_low, size=full_hw, mode="bilinear", align_corners=False)
        eps_full = F.interpolate(eps_low, size=full_hw, mode="nearest")

        noise = torch.randn_like(eps_full) * self.noise_scale
        noise_low = F.avg_pool2d(noise, kernel_size=self.ralu_f0, stride=self.ralu_f0)
        noise_low = F.interpolate(noise_low, size=full_hw, mode="nearest")
        eps_full = eps_full + self.ralu_hf_noise * (noise - noise_low)

        return t * x0_full + (1.0 - t) * eps_full

    @torch.no_grad()
    def _edge_layout_from_low_x0(self, x0_low):
        bsz, _, h_low, w_low = x0_low.shape
        p = self.net.patch_size
        f0 = self.ralu_f0
        assert h_low % p == 0 and w_low % p == 0, "low-res size must be divisible by patch size"

        x = (x0_low.clamp(-1, 1) + 1) * 0.5
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = (gx.square() + gy.square()).sqrt()

        score = F.avg_pool2d(mag, kernel_size=p, stride=p).mean(dim=0).flatten()
        num_low = score.numel()
        if self.ralu_up_ratio <= 0:
            k = 0
            edge_low_idx = torch.empty(0, device=x.device, dtype=torch.long)
        else:
            k = min(num_low, max(1, int(num_low * self.ralu_up_ratio)))
            edge_low_idx = torch.topk(score, k=k, largest=True).indices

        edge_low_mask = torch.zeros(num_low, device=x.device, dtype=torch.bool)
        if k > 0:
            edge_low_mask.scatter_(0, edge_low_idx, True)
        keep_low_idx = torch.nonzero(~edge_low_mask, as_tuple=False).flatten()

        g_low = h_low // p
        g_full = g_low * f0
        center = (f0 - 1) * 0.5

        low_rows = keep_low_idx // g_low
        low_cols = keep_low_idx % g_low
        low_coords = torch.stack(
            [low_rows.float() * f0 + center, low_cols.float() * f0 + center],
            dim=1,
        )

        if k > 0:
            edge_rows = edge_low_idx // g_low
            edge_cols = edge_low_idx % g_low
            offsets = torch.arange(f0, device=x.device, dtype=torch.long)
            off_r, off_c = torch.meshgrid(offsets, offsets, indexing="ij")
            full_rows = edge_rows[:, None] * f0 + off_r.flatten()[None, :]
            full_cols = edge_cols[:, None] * f0 + off_c.flatten()[None, :]
            edge_full_idx = (full_rows * g_full + full_cols).flatten()
            edge_coords = torch.stack([full_rows.flatten().float(), full_cols.flatten().float()], dim=1)
        else:
            edge_full_idx = torch.empty(0, device=x.device, dtype=torch.long)
            edge_coords = torch.empty(0, 2, device=x.device, dtype=torch.float32)

        coords = torch.cat([low_coords, edge_coords], dim=0)
        return {
            "coords": coords,
            "keep_low_idx": keep_low_idx,
            "edge_full_idx": edge_full_idx,
            "num_low_tokens": keep_low_idx.numel(),
            "g_low": g_low,
            "g_full": g_full,
        }

    @torch.no_grad()
    def _build_mixed_tokens(self, z_full, layout, low_net):
        h, w = z_full.shape[-2:]
        f0 = self.ralu_f0
        z_low = F.interpolate(z_full, size=(h // f0, w // f0), mode="bilinear", align_corners=False)

        low_tokens = low_net.x_embedder(z_low)
        full_tokens = self.net.x_embedder(z_full)

        low_part = low_tokens.index_select(1, layout["keep_low_idx"])
        edge_part = full_tokens.index_select(1, layout["edge_full_idx"])
        return torch.cat([low_part, edge_part], dim=1)

    @torch.no_grad()
    def _scatter_mixed_patches_to_full(self, patch_pred, layout, full_hw):
        bsz = patch_pred.size(0)
        p = self.net.patch_size
        f0 = self.ralu_f0
        channels = self.net.out_channels
        h, w = full_hw

        out = torch.zeros(bsz, channels, h, w, device=patch_pred.device, dtype=patch_pred.dtype)
        n_low = layout["num_low_tokens"]
        g_low = layout["g_low"]
        g_full = layout["g_full"]

        if n_low > 0:
            low_pred = patch_pred[:, :n_low]
            low_pred = low_pred.reshape(bsz, n_low, p, p, channels).permute(0, 1, 4, 2, 3)
            low_pred = low_pred.reshape(bsz * n_low, channels, p, p)
            low_pred = F.interpolate(low_pred, size=(p * f0, p * f0), mode="bilinear", align_corners=False)
            low_pred = low_pred.reshape(bsz, n_low, channels, p * f0, p * f0)

            for token_id, low_idx in enumerate(layout["keep_low_idx"].tolist()):
                row = low_idx // g_low
                col = low_idx % g_low
                y0 = row * p * f0
                x0 = col * p * f0
                out[:, :, y0:y0 + p * f0, x0:x0 + p * f0] = low_pred[:, token_id]

        edge_pred = patch_pred[:, n_low:]
        if edge_pred.numel() > 0:
            edge_pred = edge_pred.reshape(bsz, -1, p, p, channels).permute(0, 1, 4, 2, 3)
            for token_id, full_idx in enumerate(layout["edge_full_idx"].tolist()):
                row = full_idx // g_full
                col = full_idx % g_full
                y0 = row * p
                x0 = col * p
                out[:, :, y0:y0 + p, x0:x0 + p] = edge_pred[:, token_id]

        return out

    @torch.no_grad()
    def _mixed_sparse_forward(self, z_full, t, labels, layout, low_net):
        t = self._expand_t(t, z_full)
        t_flat = t.flatten()
        tokens = self._build_mixed_tokens(z_full, layout, low_net)
        coords = layout["coords"].to(device=z_full.device)

        x_cond_patches = self.net.forward_mixed_tokens(tokens, coords, t_flat, labels)
        x_cond = self._scatter_mixed_patches_to_full(x_cond_patches, layout, z_full.shape[-2:])
        v_cond = (x_cond - z_full) / (1.0 - t).clamp_min(self.t_eps)

        null_labels = torch.full_like(labels, self.num_classes)
        x_uncond_patches = self.net.forward_mixed_tokens(tokens, coords, t_flat, null_labels)
        x_uncond = self._scatter_mixed_patches_to_full(x_uncond_patches, layout, z_full.shape[-2:])
        v_uncond = (x_uncond - z_full) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(
            interval_mask,
            torch.full_like(t, float(self.cfg_scale)),
            torch.ones_like(t),
        )

        v = v_uncond + cfg_scale_interval * (v_cond - v_uncond)
        x0 = z_full + (1.0 - t).clamp_min(self.t_eps) * v
        return v, x0

    @torch.no_grad()
    def generate_ralu(self, labels):
        return self.generate_ralu_diagnostic(labels)["sample"]

    @torch.no_grad()
    def generate_ralu_diagnostic(self, labels):
        """
        Conservative two-stage Pixel-RALU diagnostic path.

        Stage 1: low-res JiT denoising from 0 -> ralu_e[0].
        Stage 2: directly lift to full resolution.
        Stage 3: full-res JiT refinement from ralu_e[0] -> 1.

        This intentionally does not use mixed-token forward. The returned
        intermediates make it possible to check whether low-res x0 is usable.
        """
        assert len(self.ralu_N) == 3, "ralu_N must contain 3 stage lengths"
        assert len(self.ralu_e) == 3, "ralu_e must contain 3 stage end times"
        assert abs(self.ralu_e[-1] - 1.0) < 1e-6, "ralu_e must end at 1.0"

        device = labels.device
        bsz = labels.size(0)
        f0 = self.ralu_f0
        assert self.img_size % f0 == 0, "img_size must be divisible by ralu_f0"

        full_hw = (self.img_size, self.img_size)
        low_hw = (self.img_size // f0, self.img_size // f0)
        low_net = self._make_ralu_low_net()

        z = self.noise_scale * torch.randn(bsz, 3, low_hw[0], low_hw[1], device=device)

        ts = torch.linspace(0.0, self.ralu_e[0], self.ralu_N[0] + 1, device=device)
        for i in range(self.ralu_N[0]):
            z, _ = self._ode_step_with_fn(
                lambda zz, tt, yy: self._cfg_v_and_x(low_net, zz, tt, yy),
                z,
                ts[i],
                ts[i + 1],
                labels,
            )

        _, x0_low = self._cfg_v_and_x(low_net, z, ts[-1], labels)
        x0_low_up = F.interpolate(x0_low, size=full_hw, mode="bilinear", align_corners=False)
        z = self._lift_low_state_to_full(z, x0_low, ts[-1], full_hw)
        z_lift = z

        full_steps = self.ralu_N[2]
        ts = torch.linspace(self.ralu_e[0], self.ralu_e[2], full_steps + 1, device=device)
        for i in range(full_steps):
            z, _ = self._ode_step_with_fn(
                lambda zz, tt, yy: self._cfg_v_and_x(self.net, zz, tt, yy),
                z,
                ts[i],
                ts[i + 1],
                labels,
            )

        return {
            "sample": z,
            "x0_low": x0_low,
            "x0_low_up": x0_low_up,
            "z_lift": z_lift,
            "t_lift": torch.tensor(self.ralu_e[0], device=device),
        }

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
