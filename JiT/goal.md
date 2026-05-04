核心目的： 将 RALU（一种用于扩散 Transformer 的免训练、推理加速算法）移植并应用到 JiT（一个基于纯像素空间的极简图像生成 Transformer）上，大幅提升它的图像生成速度。 
JiT：https://github.com/LTH14/JiT.git 
RALU：https://github.com/ignoww/RALU.git


## 结论

不要直接把 `RALU/pipeline_flux_RALU.py` 复制进 JiT。RALU 原版面向 FLUX/SD3 的 **latent-token pipeline**；JiT 是 **pixel-space denoising Transformer**，README 明确说它采用 pixel-level high-resolution image diffusion 的极简自包含设计。([GitHub][1])

要移植，核心是把 RALU 的“latent upsampling”改成 **pixel/patch upsampling**：

```text
RALU 原版:
低分辨率 latent denoise
→ 边缘区域 latent 提前上采样
→ 全 latent 上采样 refinement

JiT 版本:
低分辨率 pixel image denoise
→ 边缘 patch 区域提前进入高分辨率 denoise
→ 全分辨率 pixel refinement
```

JiT 当前最大障碍是：`model_jit.py` 固定输入分辨率。`BottleneckPatchEmbed.forward()` 会 assert 输入图像尺寸等于模型初始化尺寸；同时 `pos_embed`、RoPE 都按 `input_size // patch_size` 固定生成。([GitHub][2]) ([GitHub][2]) 所以融合分两条路线：

| 路线                        |       是否训练 | 改动量 | 加速幅度 | 质量风险 | 建议用途           |
| ------------------------- | ---------: | --: | ---: | ---: | -------------- |
| **A. Pixel-RALU 推理版 MVP** |        不训练 |   中 |  中到高 |    中 | 先跑通、做 ablation |
| **B. 真 mixed-token RALU** | 可不训练，但建议微调 |   大 |    高 |   中低 | 最接近 RALU 论文目标  |
| **C. RALU-aware 训练/蒸馏**   |      训练或微调 |   大 |    高 |   最低 | 最终方案           |

下面按“不训练”和“训练”分别给具体改法。

---

# 一、不训练方向：Pixel-RALU MVP

## 1. 修改位置

需要改 3 个文件：

```text
main_jit.py       # 增加 RALU 参数
denoiser.py       # 增加 generate_ralu()
engine_jit.py     # EMA 切换前后重置 RALU cache
```

JiT 的采样入口在 `Denoiser.generate()`：它从高斯噪声开始，构造 `torch.linspace(0, 1, steps+1)`，然后用 Euler/Heun ODE 采样。([GitHub][3]) CFG 在 `_forward_sample()` 里通过 conditional/unconditional 两次 `self.net()` 实现。([GitHub][3]) 所以 RALU 应该插在 `denoiser.py` 的 `generate()` 路径，而不是训练 loop。

RALU 原版有三阶段，默认 4× 配置是 `N=[5,6,7]`、`e=[0.3,0.45,1.0]`、`up_ratio=0.3`，7× 配置是 `N=[2,3,5]`、`e=[0.2,0.3,1.0]`、`up_ratio=0.1`。([GitHub][4]) 原版 Stage 1 先降分辨率，Stage 2 用边缘检测选 top-k patch 提前上采样，Stage 3 再补全所有 latent。([GitHub][4]) ([GitHub][4])

---

## 2. `main_jit.py`：加参数

在 `get_args_parser()` 的 sampling 参数区加：

```python
# RALU / Pixel-RALU sampling
parser.add_argument('--use_ralu', action='store_true',
                    help='Use Pixel-RALU accelerated sampling for JiT.')
parser.add_argument('--ralu_f0', type=int, default=2,
                    help='Spatial downsample factor for RALU stage 1.')
parser.add_argument('--ralu_N', type=int, nargs='+', default=[10, 4, 8],
                    help='Number of sampling steps for 3 RALU stages.')
parser.add_argument('--ralu_e', type=float, nargs='+', default=[0.35, 0.55, 1.0],
                    help='End t for 3 RALU stages. Must end with 1.0.')
parser.add_argument('--ralu_up_ratio', type=float, default=0.3,
                    help='Fraction of low-res edge patches to enter full-res early.')
parser.add_argument('--ralu_hf_noise', type=float, default=0.25,
                    help='High-frequency noise injected when lifting low-res state.')
```

建议先别直接用 RALU 的 7× 设置。JiT 是 pixel-space，VAE latent 的稳定性假设不完全成立。先用保守配置：

```bash
--use_ralu --ralu_N 10 4 8 --ralu_e 0.35 0.55 1.0 --ralu_up_ratio 0.3
```

跑通后再压到：

```bash
--use_ralu --ralu_N 6 3 6 --ralu_e 0.25 0.45 1.0 --ralu_up_ratio 0.2
```

---

## 3. `denoiser.py`：增加 Pixel-RALU 推理

### 3.1 增加 import

```python
import copy
import torch.nn.functional as F
```

### 3.2 在 `Denoiser.__init__()` 里保存参数

在原来的 generation hyper params 后面加：

```python
# Pixel-RALU hyper params
self.use_ralu = getattr(args, "use_ralu", False)
self.ralu_f0 = getattr(args, "ralu_f0", 2)
self.ralu_N = getattr(args, "ralu_N", [10, 4, 8])
self.ralu_e = getattr(args, "ralu_e", [0.35, 0.55, 1.0])
self.ralu_up_ratio = getattr(args, "ralu_up_ratio", 0.3)
self.ralu_hf_noise = getattr(args, "ralu_hf_noise", 0.25)

self.model_name = args.model
self.attn_dropout = args.attn_dropout
self.proj_dropout = args.proj_dropout
```

### 3.3 在 `generate()` 开头分流

把原来的：

```python
@torch.no_grad()
def generate(self, labels):
    device = labels.device
    ...
```

改成：

```python
@torch.no_grad()
def generate(self, labels):
    if self.use_ralu:
        return self.generate_ralu(labels)

    device = labels.device
    ...
```

### 3.4 在 `Denoiser` 类里新增这些方法

直接放在 `_forward_sample()` 前后都可以。

```python
def reset_ralu_cache(self):
    self.__dict__.pop("_ralu_low_net", None)


@torch.no_grad()
def _make_ralu_low_net(self):
    """
    Build a low-resolution JiT with the same weights.

    Important:
    Do NOT assign it as self.net_low, otherwise it becomes a registered
    nn.Module and will break strict load_state_dict in engine_jit.evaluate().
    """
    cached = self.__dict__.get("_ralu_low_net", None)
    if cached is not None:
        return cached

    assert self.img_size % self.ralu_f0 == 0
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

    # Copy only shape-compatible weights.
    # pos_embed has different shape and should be kept from low_net initialization.
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)

    low_net.load_state_dict(dst, strict=True)

    # Avoid registering as submodule.
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
    """
    Same as _forward_sample(), but supports a supplied net and also returns
    CFG-guided x0 prediction.
    """
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
    """
    forward_fn(z, t, labels) -> (v, x0)
    Uses Heun except when t_next == 1.0, matching JiT's original
    final Euler behavior.
    """
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
def _edge_mask_from_low_x0(self, x0_low):
    """
    Build a full-resolution binary mask from low-res x0.
    Selected low-res patches are expanded to full-res pixel regions.

    x0_low: [B, 3, H/f0, W/f0]
    return: [B, 1, H, W]
    """
    bsz, _, h_low, w_low = x0_low.shape
    p = self.net.patch_size
    f0 = self.ralu_f0

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

    # One score per low-res JiT patch.
    assert h_low % p == 0 and w_low % p == 0
    score = F.avg_pool2d(mag, kernel_size=p, stride=p)  # [B,1,Hp,Wp]

    flat = score.flatten(1)
    k = max(1, int(flat.size(1) * self.ralu_up_ratio))
    top_idx = torch.topk(flat, k=k, dim=1, largest=True).indices

    mask_grid = torch.zeros_like(flat, dtype=torch.bool)
    mask_grid.scatter_(1, top_idx, True)
    mask_grid = mask_grid.view_as(score).float()

    # A low-res patch corresponds to f0*p pixels in full-res image.
    mask = mask_grid.repeat_interleave(p * f0, dim=2).repeat_interleave(p * f0, dim=3)
    return mask


@torch.no_grad()
def _lift_low_state_to_full(self, z_low, x0_low, t, full_hw):
    """
    JiT-compatible resolution transition.

    JiT forward process:
        z_t = t * x + (1 - t) * eps

    So when lifting low-res state, lift x0 and eps separately.
    This is the pixel-space replacement for RALU's latent noise matching.
    """
    t = self._expand_t(t, z_low)

    eps_low = (z_low - t * x0_low) / (1.0 - t).clamp_min(self.t_eps)

    x0_full = F.interpolate(
        x0_low,
        size=full_hw,
        mode="bilinear",
        align_corners=False,
    )
    eps_full = F.interpolate(
        eps_low,
        size=full_hw,
        mode="nearest",
    )

    # Add high-frequency-only noise so full-res details are not over-smoothed.
    noise = torch.randn_like(eps_full) * self.noise_scale
    f0 = self.ralu_f0
    noise_low = F.avg_pool2d(noise, kernel_size=f0, stride=f0)
    noise_low = F.interpolate(noise_low, size=full_hw, mode="nearest")
    high_freq_noise = noise - noise_low

    eps_full = eps_full + self.ralu_hf_noise * high_freq_noise

    return t * x0_full + (1.0 - t) * eps_full


@torch.no_grad()
def _mixed_proxy_forward(self, z_full, t, labels, mask_full, low_net):
    """
    MVP mixed-resolution proxy.

    It still calls the full JiT on the whole image for edge regions, then
    combines it with an upsampled low-res prediction outside the edge mask.

    This is not the final true sparse RALU. It is a robust no-training bridge.
    """
    h, w = z_full.shape[-2:]
    f0 = self.ralu_f0

    v_full, _ = self._cfg_v_and_x(self.net, z_full, t, labels)

    z_low = F.interpolate(
        z_full,
        size=(h // f0, w // f0),
        mode="bilinear",
        align_corners=False,
    )
    v_low, _ = self._cfg_v_and_x(low_net, z_low, t, labels)
    v_low = F.interpolate(
        v_low,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    v = mask_full * v_full + (1.0 - mask_full) * v_low
    t = self._expand_t(t, z_full)
    x0 = z_full + (1.0 - t).clamp_min(self.t_eps) * v
    return v, x0


@torch.no_grad()
def generate_ralu(self, labels):
    """
    Pixel-RALU MVP:
      Stage 1: low-res JiT denoising
      Stage 2: edge-mask mixed proxy
      Stage 3: full-res refinement
    """
    assert len(self.ralu_N) == 3
    assert len(self.ralu_e) == 3
    assert abs(self.ralu_e[-1] - 1.0) < 1e-6

    device = labels.device
    bsz = labels.size(0)

    f0 = self.ralu_f0
    full_hw = (self.img_size, self.img_size)
    low_hw = (self.img_size // f0, self.img_size // f0)

    low_net = self._make_ralu_low_net()

    starts = [0.0, self.ralu_e[0], self.ralu_e[1]]
    ends = self.ralu_e

    # Stage 1: start directly in low resolution.
    z = self.noise_scale * torch.randn(
        bsz, 3, low_hw[0], low_hw[1],
        device=device,
    )

    # -------- Stage 1: low-res denoising --------
    ts = torch.linspace(starts[0], ends[0], self.ralu_N[0] + 1, device=device)
    for i in range(self.ralu_N[0]):
        z, _ = self._ode_step_with_fn(
            lambda zz, tt, yy: self._cfg_v_and_x(low_net, zz, tt, yy),
            z,
            ts[i],
            ts[i + 1],
            labels,
        )

    _, x0_low = self._cfg_v_and_x(low_net, z, ts[-1], labels)

    # Edge-sensitive regions selected from predicted clean image.
    mask_full = self._edge_mask_from_low_x0(x0_low).to(device=device, dtype=z.dtype)

    # Lift state to full resolution with JiT-compatible x0/eps decomposition.
    z = self._lift_low_state_to_full(z, x0_low, ts[-1], full_hw)

    # -------- Stage 2: mixed proxy --------
    ts = torch.linspace(starts[1], ends[1], self.ralu_N[1] + 1, device=device)
    for i in range(self.ralu_N[1]):
        z, _ = self._ode_step_with_fn(
            lambda zz, tt, yy: self._mixed_proxy_forward(zz, tt, yy, mask_full, low_net),
            z,
            ts[i],
            ts[i + 1],
            labels,
        )

    # -------- Stage 3: full-res refinement --------
    ts = torch.linspace(starts[2], ends[2], self.ralu_N[2] + 1, device=device)
    for i in range(self.ralu_N[2]):
        z, _ = self._ode_step_with_fn(
            lambda zz, tt, yy: self._cfg_v_and_x(self.net, zz, tt, yy),
            z,
            ts[i],
            ts[i + 1],
            labels,
        )

    return z
```

---

## 4. `engine_jit.py`：EMA 切换时清 cache

`evaluate()` 里切 EMA 后、生成前加：

```python
if hasattr(model_without_ddp, "reset_ralu_cache"):
    model_without_ddp.reset_ralu_cache()
```

在恢复非 EMA 参数后也加一次：

```python
if hasattr(model_without_ddp, "reset_ralu_cache"):
    model_without_ddp.reset_ralu_cache()
```

原因：低分辨率 `low_net` 是从当前 `self.net` 权重复制出来的。JiT evaluation 会先把模型切到 EMA 参数再采样，然后再切回来；不重置 cache 会导致 low-res net 和当前 EMA 权重不同步。

---

## 5. 不训练 MVP 的运行命令

以 JiT-B/16 256 为例：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_jit.py \
  --model JiT-B/16 \
  --img_size 256 \
  --noise_scale 1.0 \
  --gen_bsz 256 \
  --num_images 50000 \
  --cfg 3.0 \
  --interval_min 0.1 \
  --interval_max 1.0 \
  --output_dir ${CKPT_DIR} \
  --resume ${CKPT_DIR} \
  --data_path ${IMAGENET_PATH} \
  --evaluate_gen \
  --use_ralu \
  --ralu_N 10 4 8 \
  --ralu_e 0.35 0.55 1.0 \
  --ralu_up_ratio 0.3 \
  --ralu_hf_noise 0.25
```

### 调参优先级

| 参数              | 影响                | 建议           |
| --------------- | ----------------- | ------------ |
| `ralu_N[0]`     | 语义布局稳定性           | 先不要太小，8–12   |
| `ralu_N[2]`     | 细节恢复              | 质量差时优先加这个    |
| `ralu_e[0]`     | 低分辨率结束时刻          | 太早会语义差，太晚会模糊 |
| `ralu_up_ratio` | 提前 full-res 的区域比例 | 0.2–0.4      |
| `ralu_hf_noise` | 上采样后的高频随机性        | 0.1–0.35     |

---

# 二、不训练方向：真正 mixed-token RALU

上面的 MVP 能跑通，但 Stage 2 仍然调用一次 full JiT，因此不是完全的空间稀疏加速。要获得接近 RALU 的大幅加速，需要改 `model_jit.py`，让 JiT 支持 **mixed-resolution token sequence**。

RALU 原版 Stage 2 是：低分辨率 token 里选 edge top-k，edge token 提前上采样成多个高分辨率 token，其他区域继续保持低分辨率。它还在分辨率切换后加噪声匹配。([GitHub][4]) ([GitHub][4]) JiT 当前 `forward()` 是固定流程：`x_embedder(x)`、加固定 `pos_embed`、blocks、去掉 in-context token、final layer、unpatchify。([GitHub][2]) 因此要把 `forward()` 拆成 token 级接口。

## 1. `model_jit.py` 需要拆出 4 个函数

建议把当前 `JiT.forward()` 拆成：

```python
def patchify(self, x):
    # [B,3,H,W] -> [B,N,C]
    return self.x_embedder(x)

def add_pos(self, tokens, coords, grid_size):
    # coords: [N,2], token coordinates in full-res patch coordinate system
    # return tokens + dynamic sincos pos
    ...

def forward_tokens(self, tokens, t, y, coords):
    # transformer blocks on arbitrary token sequence
    ...

def decode_tokens(self, tokens, layout):
    # scatter mixed tokens back to image / low image
    ...
```

关键是不能再用固定 `self.pos_embed` 和固定 `VisionRotaryEmbeddingFast`。原版 RoPE 在初始化时根据 `pt_seq_len` 生成固定 `freqs_cos/freqs_sin`，forward 时直接和整个序列相乘。([GitHub][5]) mixed token 的坐标不是完整规则网格，所以要改成 **按 coords 索引/生成 RoPE**。

## 2. 新增 indexable RoPE

在 `util/model_util.py` 新增：

```python
class IndexedVisionRoPE(nn.Module):
    def __init__(self, dim, theta=10000, num_cls_token=0):
        super().__init__()
        self.dim = dim
        self.num_cls_token = num_cls_token
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)

    def _build_cos_sin(self, coords, dtype, device):
        """
        coords: [N, 2], full-res patch coordinates, e.g. [row, col].
        """
        coords = coords.to(device=device, dtype=torch.float32)
        freqs = self.freqs.to(device=device)

        h = torch.einsum("n,d->nd", coords[:, 0], freqs)
        w = torch.einsum("n,d->nd", coords[:, 1], freqs)

        h = torch.repeat_interleave(h, 2, dim=-1)
        w = torch.repeat_interleave(w, 2, dim=-1)
        rope = torch.cat([h, w], dim=-1)

        cos = rope.cos().to(dtype=dtype)
        sin = rope.sin().to(dtype=dtype)
        return cos, sin

    def forward(self, q_or_k, coords, has_cls_tokens=False):
        """
        q_or_k: [B, heads, N, head_dim]
        coords: [N_img, 2]
        """
        if has_cls_tokens and self.num_cls_token > 0:
            cls_n = self.num_cls_token
            img = q_or_k[:, :, cls_n:, :]
            cls = q_or_k[:, :, :cls_n, :]

            cos, sin = self._build_cos_sin(img.new_tensor(coords), img.dtype, img.device)
            img = img * cos[None, None, :, :] + rotate_half(img) * sin[None, None, :, :]
            return torch.cat([cls, img], dim=2)

        cos, sin = self._build_cos_sin(coords, q_or_k.dtype, q_or_k.device)
        return q_or_k * cos[None, None, :, :] + rotate_half(q_or_k) * sin[None, None, :, :]
```

然后改 `Attention.forward()`：

```python
def forward(self, x, rope, coords=None, has_cls_tokens=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = self.q_norm(q)
    k = self.k_norm(k)

    if coords is None:
        q = rope(q)
        k = rope(k)
    else:
        q = rope(q, coords, has_cls_tokens=has_cls_tokens)
        k = rope(k, coords, has_cls_tokens=has_cls_tokens)

    x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

同时让 `JiTBlock.forward()` 接收 `coords`：

```python
def forward(self, x, c, feat_rope=None, coords=None, has_cls_tokens=False):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=-1)

    x = x + gate_msa.unsqueeze(1) * self.attn(
        modulate(self.norm1(x), shift_msa, scale_msa),
        rope=feat_rope,
        coords=coords,
        has_cls_tokens=has_cls_tokens,
    )
    x = x + gate_mlp.unsqueeze(1) * self.mlp(
        modulate(self.norm2(x), shift_mlp, scale_mlp)
    )
    return x
```

## 3. mixed-token layout

设 full-res patch grid 为 `G x G`，低分辨率 factor 为 `f0=2`。Stage 2 的 token 数：

```text
低分辨率总 token: M / 4
选中比例: r
非边缘低分辨率 token: (1-r) * M/4
边缘高分辨率 token: r * M/4 * 4 = r*M

Stage2 token count = M * (0.25 + 0.75r)
```

例如 `r=0.3` 时，Stage 2 token count 是 `0.475M`。attention 理论复杂度约为 `0.475² ≈ 22.6%` full attention，MLP/linear 约为 `47.5%` full token 量。

实现 layout：

```python
def build_mixed_indices(G_low, up_ratio, edge_score):
    """
    G_low = G_full // 2
    edge_score: [B, G_low*G_low]
    """
    k = int(G_low * G_low * up_ratio)
    edge_idx = edge_score.topk(k, dim=1).indices

    all_idx = torch.arange(G_low * G_low, device=edge_score.device)
    # For simplicity, batch size 1 version. For B>1, make per-sample lists.
    edge = torch.sort(edge_idx[0])[0]
    keep_low = all_idx[~torch.isin(all_idx, edge)]

    return keep_low, edge
```

低分辨率 token 坐标用 coarse cell center：

```python
# low token at low-grid (i,j) corresponds to full-grid center (2i+0.5, 2j+0.5)
coords_low = torch.stack([2 * i + 0.5, 2 * j + 0.5], dim=-1)
```

edge token 展开成 4 个 full patch：

```python
coords_edge = [
    [2*i + 0, 2*j + 0],
    [2*i + 0, 2*j + 1],
    [2*i + 1, 2*j + 0],
    [2*i + 1, 2*j + 1],
]
```

mixed sequence：

```text
tokens = concat(
  low_tokens[:, keep_low],
  high_tokens[:, expanded_edge_indices]
)

coords = concat(
  coords_low_keep,
  coords_edge_high
)
```

## 4. 为什么这个才是真正 RALU

MVP 的 `_mixed_proxy_forward()` 仍调用 full JiT，因此 Stage 2 没完全省掉空间计算。真正 mixed-token 版只把 `tokens` 输入 Transformer，token 数变少，attention/MLP 都少算。这才对应 RALU 的空间加速思想：只让 edge-sensitive regions 提前进入高分辨率。

---

# 三、训练方向：RALU-aware fine-tuning / distillation

不训练能做出 baseline，但 JiT 没见过低分辨率/混合分辨率 token 分布，质量上限有限。最终建议做 **teacher-student distillation**：

```text
Teacher: 原始 full-res JiT EMA，不改
Student: 支持 mixed-token 的 JiT，初始化为 Teacher 权重
训练目标:
  1. 继续学真实 denoising target
  2. 学 Teacher 的 full-res v_pred / x_pred
  3. 随机模拟 RALU 三阶段状态
```

## 1. 训练时改 `Denoiser.forward()`

原始 JiT 训练逻辑是：

```python
z = t * x + (1 - t) * e
v = (x - z) / (1 - t)
x_pred = self.net(z, t, labels)
v_pred = (x_pred - z) / (1 - t)
loss = mse(v, v_pred)
```

这在源码里已经是 x-prediction 转 v-loss 的形式。([GitHub][3]) 训练版 RALU 不要改目标，只是让 student 在 low/mixed 输入下仍输出 full-res 预测。

新增参数：

```python
parser.add_argument('--train_ralu', action='store_true')
parser.add_argument('--ralu_train_prob', type=float, default=0.5)
parser.add_argument('--ralu_distill_weight', type=float, default=1.0)
parser.add_argument('--ralu_gt_weight', type=float, default=1.0)
parser.add_argument('--ralu_mask_from', type=str, default='clean',
                    choices=['clean', 'teacher', 'student'])
```

在 `Denoiser.__init__()` 加：

```python
self.train_ralu = getattr(args, "train_ralu", False)
self.ralu_train_prob = getattr(args, "ralu_train_prob", 0.5)
self.ralu_distill_weight = getattr(args, "ralu_distill_weight", 1.0)
self.ralu_gt_weight = getattr(args, "ralu_gt_weight", 1.0)
```

`forward()` 改成：

```python
def forward(self, x, labels):
    if self.training and self.train_ralu:
        if torch.rand((), device=x.device) < self.ralu_train_prob:
            return self.forward_ralu_train(x, labels)

    # original JiT loss
    labels_dropped = self.drop_labels(labels) if self.training else labels

    t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
    e = torch.randn_like(x) * self.noise_scale

    z = t * x + (1 - t) * e
    v = (x - z) / (1 - t).clamp_min(self.t_eps)

    x_pred = self.net(z, t.flatten(), labels_dropped)
    v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

    loss = (v - v_pred) ** 2
    loss = loss.mean(dim=(1, 2, 3)).mean()
    return loss
```

## 2. `forward_ralu_train()` 逻辑

如果还没做 true mixed-token，可以先训练 MVP 版；但最推荐训练的是 mixed-token Student。

伪代码：

```python
def forward_ralu_train(self, x, labels):
    labels_dropped = self.drop_labels(labels)

    B, C, H, W = x.shape
    f0 = self.ralu_f0

    # sample t from original JiT distribution
    t = self.sample_t(B, device=x.device).view(-1, 1, 1, 1)
    e = torch.randn_like(x) * self.noise_scale
    z = t * x + (1 - t) * e
    v_gt = (x - z) / (1 - t).clamp_min(self.t_eps)

    # Teacher full-res prediction
    with torch.no_grad():
        x_teacher = self.teacher_net(z, t.flatten(), labels)
        v_teacher = (x_teacher - z) / (1 - t).clamp_min(self.t_eps)

    # Build low-res state
    x_low = F.interpolate(x, scale_factor=1 / f0, mode='bilinear', align_corners=False)
    z_low = F.interpolate(z, scale_factor=1 / f0, mode='bilinear', align_corners=False)

    # Edge mask from clean x or teacher x0
    if self.ralu_mask_from == "clean":
        edge_source = x_low
    else:
        edge_source = F.interpolate(x_teacher, scale_factor=1 / f0, mode='bilinear', align_corners=False)

    mask_full = self._edge_mask_from_low_x0(edge_source)

    # Student mixed-token forward.
    # This requires model_jit.py forward_mixed_tokens().
    x_student = self.net.forward_mixed_tokens(
        z_low=z_low,
        z_full=z,
        t=t.flatten(),
        y=labels_dropped,
        mask_full=mask_full,
        f0=f0,
        up_ratio=self.ralu_up_ratio,
    )

    v_student = (x_student - z) / (1 - t).clamp_min(self.t_eps)

    loss_gt = (v_student - v_gt).square().mean()
    loss_distill = (v_student - v_teacher).square().mean()

    return self.ralu_gt_weight * loss_gt + self.ralu_distill_weight * loss_distill
```

## 3. Teacher 怎么构造

在 `Denoiser.__init__()` 里：

```python
self.teacher_net = None
```

checkpoint load 完后，在 `main_jit.py` 或 `engine_jit.py` 里加：

```python
if args.train_ralu:
    model_without_ddp.teacher_net = copy.deepcopy(model_without_ddp.net)
    model_without_ddp.teacher_net.eval()
    for p in model_without_ddp.teacher_net.parameters():
        p.requires_grad_(False)
```

更稳的做法是用 EMA 权重初始化 teacher，而 student 继续训练。

## 4. 训练 curriculum

建议三阶段训练：

### Stage A：低分辨率鲁棒性

```text
mode = full-res original loss: 50%
mode = low-res-only then lift: 50%
```

目的：让模型适应低分辨率 denoising 轨迹。

### Stage B：mixed-token distillation

```text
mode = full-res original loss: 30%
mode = mixed-token RALU loss: 70%
up_ratio randomly sampled from [0.1, 0.5]
f0 = 2
```

目的：让模型学会非规则 token layout。

### Stage C：真实推理 schedule 微调

固定成最终推理设置，例如：

```text
ralu_N = [6, 3, 6]
ralu_e = [0.25, 0.45, 1.0]
up_ratio = 0.2 或 0.3
```

用 teacher distillation + FID/IS online eval 选 checkpoint。

---

# 四、关键实现差异：JiT 不能照搬 RALU 的 NT-DM

RALU 原版的 noise-timestep rescheduling 是针对 FLUX/SD3 scheduler 和 latent 分布设计的，`pipeline_flux_RALU.py` 里会用 `shift`、`Z`、`alpha/beta`，并在 Stage 2/3 切换时做 `latents = beta * latents + alpha * noise`。([GitHub][4]) ([GitHub][4])

JiT 的前向噪声形式更直接：

```python
z = t * x + (1 - t) * e
```

所以在 JiT 里更合理的切换公式是：

```python
eps_low = (z_low - t * x0_low) / (1 - t)
z_full = t * upsample(x0_low) + (1 - t) * upsample_or_mix(eps_low)
```

这就是上面 `_lift_low_state_to_full()` 的核心。这个比直接套 RALU 的 `alpha/beta/Z` 更贴合 JiT。

---

# 五、最终建议路线

## 先做这个

1. 实现 **Pixel-RALU MVP**。
2. 用 `--ralu_N 10 4 8 --ralu_e 0.35 0.55 1.0` 跑 5k 或 10k images。
3. 对比：

   * 原始 `num_sampling_steps=50`
   * 简单减少步数，例如 `num_sampling_steps=18`
   * Pixel-RALU 22 steps
4. 看 FID/IS、单图 latency、吞吐。

## 然后做这个

实现 true mixed-token：

```text
model_jit.py:
  - dynamic/indexed pos_embed
  - indexed RoPE
  - forward_tokens()
  - forward_mixed_tokens()
  - mixed scatter/unpatchify

denoiser.py:
  - generate_ralu_sparse()
```

## 最后做训练版

用 teacher-student distillation 微调 mixed-token Student。这个版本最有可能同时做到：

```text
高速度 + 低 artifact + 接近原 JiT 质量
```

核心判断：**不训练版本可以验证方向；真正要“大幅提升速度且质量稳定”，需要 mixed-token forward，最好再做 RALU-aware distillation。**

[1]: https://github.com/LTH14/JiT/blob/main/README.md "JiT/README.md at main · LTH14/JiT · GitHub"
[2]: https://github.com/LTH14/JiT/blob/main/model_jit.py "JiT/model_jit.py at main · LTH14/JiT · GitHub"
[3]: https://github.com/LTH14/JiT/blob/main/denoiser.py "JiT/denoiser.py at main · LTH14/JiT · GitHub"
[4]: https://github.com/ignoww/RALU/blob/master/pipeline_flux_RALU.py "RALU/pipeline_flux_RALU.py at master · ignoww/RALU · GitHub"
[5]: https://raw.githubusercontent.com/LTH14/JiT/main/util/model_util.py "raw.githubusercontent.com"
