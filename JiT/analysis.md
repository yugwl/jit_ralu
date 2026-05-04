# RALU → JiT 移植计划深度分析

基于对 JiT 和 RALU 实际源码的逐行比对，以下是你的计划中存在的 **13 个关键问题**，从"会直接报错崩溃"到"设计层面可能导致质量严重退化"排列。

---

## 🔴 会直接报错的 Bug（必须修复才能运行）

### 1. `BottleneckPatchEmbed` 的 hard assert 会阻止低分辨率 net 接受任何非预设尺寸的输入

你的计划里 `_make_ralu_low_net()` 用 `input_size=low_size` 构造了一个低分辨率 JiT，再复制权重。但 [BottleneckPatchEmbed](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/model_jit.py#L32-L35) 的 `forward()` 有：

```python
assert H == self.img_size[0] and W == self.img_size[1]
```

这个 assert 本身不是问题（因为低分辨率 net 的 `img_size` 就是 `low_size`），**但真正的问题在下一条**。

### 2. `BottleneckPatchEmbed.proj1` 权重复制会**静默成功但语义完全错误**

[proj1](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/model_jit.py#L29) 是 `nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size)`。

- 原始 net（256px, patch_size=16）：`proj1.weight` shape = `[128, 3, 16, 16]`
- 低分辨率 net（128px, patch_size=16）：`proj1.weight` shape = `[128, 3, 16, 16]` ← **shape 完全一样！**

你的代码 `if k in dst and dst[k].shape == v.shape: dst[k].copy_(v)` 会把权重复制过来，且 shape 匹配。**这不会报错，但不是 Bug 的问题。** 实际上这是对的—patch embedding 的卷积核大小与图像大小无关。

> [!WARNING]
> **真正严重的问题是 `pos_embed`**：
> - 原始 net：`pos_embed` shape = `[1, 256, 768]`（16×16 patches）
> - 低分辨率 net（128px）：`pos_embed` shape = `[1, 64, 768]`（8×8 patches）
> 
> 你的代码用 `if dst[k].shape == v.shape` 过滤，所以 `pos_embed` **不会被复制**，会保留低分辨率 net 初始化时自动生成的 sin-cos embedding。这是合理的。

但是 — **RoPE 的 `freqs_cos` / `freqs_sin` 不是 `nn.Parameter`，不在 `state_dict()` 里**！看 [VisionRotaryEmbeddingFast](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/util/model_util.py#L128-L132)：

```python
self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()  # ← 直接赋值！
self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()  # ← 不是 register_buffer！
```

所以 RoPE 是在 `JiT.__init__()` 里按 `input_size // patch_size` 创建的，不通过 state_dict 传播。低分辨率 net 构造时会自动按 `low_size // patch_size` 生成正确的 RoPE。**这部分恰好是对的。**

### 3. `in_context_posemb` 和 `in_context_len` 的交互问题

JiT 的 [forward()](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/model_jit.py#L346-L354) 里，从第 `in_context_start` 层开始注入 in-context token 并切换到 `feat_rope_incontext`：

```python
for i, block in enumerate(self.blocks):
    if self.in_context_len > 0 and i == self.in_context_start:
        in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
        in_context_tokens += self.in_context_posemb
        x = torch.cat([in_context_tokens, x], dim=1)
    x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
```

你的计划中 `_cfg_v_and_x()` 直接调用 `net(z, t_flat, labels)`，这会正常走 `JiT.forward()`，in-context 逻辑会正确执行。**这里没有问题。**

但你的 `_mixed_proxy_forward()` 同时调用 `self.net(z_full)` 和 `low_net(z_low)`，两者各自走各自的 in-context 逻辑，**in-context token 的 RoPE 维度必须匹配**：

- `self.net.feat_rope_incontext`：基于 `pt_seq_len = 256//16 = 16`，`num_cls_token=32`
- `low_net.feat_rope_incontext`：基于 `pt_seq_len = 128//16 = 8`，`num_cls_token=32`

RoPE 的 `freqs_cos` shape：
- full net：`[32 + 16*16, D] = [288, D]`
- low net：`[32 + 8*8, D] = [96, D]`

**这两个都是自洽的**，因为低分辨率 net 的 token 数自然就是 64 而非 256。所以 forward 本身不会出错。✅

### 4. `_forward_sample` vs `_cfg_v_and_x`：`t` 的 shape 不匹配

> [!CAUTION]
> **这是直接会导致 runtime 崩溃的 bug。**

原始 JiT 的 `_forward_sample()` 里，`t` 的 shape 是 `[B, 1, 1, 1]`（从 `generate()` 里的 `timesteps` 切片得到，见 [line 72](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/denoiser.py#L72)）。`t.flatten()` 传给 `self.net()` 会得到一个长度为 `B` 的 1D tensor。

你的 `_cfg_v_and_x()` 里：

```python
t = self._expand_t(t, z)  # 结果是 [B, 1, 1, 1]
t_flat = t.flatten()       # 长度为 B！✅
```

但在 `generate_ralu()` 的 Stage 1 循环中：

```python
z, _ = self._ode_step_with_fn(
    lambda zz, tt, yy: self._cfg_v_and_x(low_net, zz, tt, yy),
    z, ts[i], ts[i + 1], labels,
)
```

这里 `ts[i]` 是一个 **标量 tensor**（`torch.linspace` 的第 i 个元素）。传入 `_ode_step_with_fn` 后调用 `forward_fn(z, t, labels)`，`t` 是标量。

在 `_cfg_v_and_x` 里 `self._expand_t(t, z)` 会把它扩展成 `[B, 1, 1, 1]`。然后 `t_flat = t.flatten()` 变成长度 `B` 的 1D tensor，传给 `net(z, t_flat, labels)` —— ✅ 这里是对的。

**但问题出在 `_ode_step_with_fn` 里**：

```python
is_last = bool((t_next >= 1.0 - 1e-6).detach().cpu().item())
```

`t_next` 在 Stage 1 里是标量 tensor，`.item()` 可以工作。但如果 `t` 被 expand 过后变成 `[B,1,1,1]`，调用 `.item()` 会在 B>1 时报错。检查你的代码：`t_next` 是直接从 `ts[i+1]` 传入的原始标量，**没有经过 `_expand_t`**，所以这里是安全的。✅

### 5. `cfg_scale_interval` 类型不匹配 — `torch.where` 条件的 broadcast 问题

原始 JiT [_forward_sample](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/denoiser.py#L100-L103)：

```python
interval_mask = (t < high) & ((low == 0) | (t > low))
cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
```

这里 `self.cfg_scale` 是 Python float，`1.0` 也是 float。`torch.where(bool_tensor, float, float)` 返回的是标量 broadcast。

你的 `_cfg_v_and_x` 里：

```python
cfg_on = torch.full_like(t, float(self.cfg_scale))   # shape [B,1,1,1]
cfg_off = torch.ones_like(t)                          # shape [B,1,1,1]
cfg_scale_interval = torch.where(interval_mask, cfg_on, cfg_off)
```

这能工作，但 **比原版多了不必要的 tensor 创建**。这不是 bug，只是效率问题。原版直接用 `torch.where(mask, self.cfg_scale, 1.0)` 即可。

### 6. `torch.compile` 装饰器与动态输入的冲突

> [!IMPORTANT]
> **高风险问题。**

[JiTBlock.forward](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/model_jit.py#L197-L202) 和 [FinalLayer.forward](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/model_jit.py#L175-L180) 都有 `@torch.compile` 装饰器。

当你在同一进程中创建两个不同 `input_size` 的 JiT 模型时，`torch.compile` 会为每个模型的每种输入 shape 编译一次 graph。这意味着：
- 第一次调用 `low_net(z_low)` 时会触发 JIT 编译（几十秒到几分钟）
- 第一次调用 `self.net(z_full)` 时也会触发编译

**在推理时这会让第一次 RALU 生成非常慢**。后续调用会使用缓存的 compiled graph。

更危险的是：如果 `torch.compile` 的 cache 超过 `torch._dynamo.config.cache_size_limit = 128`（[main_jit.py line 162](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/main_jit.py#L162)），可能会出现 re-compilation storm。

**建议**：在 `_make_ralu_low_net()` 之后做一次 warmup forward 来预编译。

---

## 🟡 不会崩溃，但会导致质量严重退化的设计问题

### 7. Pixel-space 边缘检测质量远不如 latent-space

> [!WARNING]
> **这是整个 MVP 方案最大的质量风险。**

RALU 原版在 Stage 1→2 过渡时：
1. 通过 **VAE 解码** 低分辨率 latent 到 pixel 空间
2. 在 pixel 空间用 **Canny 边缘检测**（`cv2.Canny(gray, 100, 200)`）
3. 按 patch 求和找边缘区域

见 [RALU pipeline line 519-536](file:///C:/Users/Weining%20Zhang/.gemini/antigravity/brain/52f9f736-df89-4d8d-9267-7f5aecd383ba/.system_generated/steps/24/content.md#L519-L536)。

你的方案用 **Sobel 梯度幅值** 替代 Canny。Sobel 在嘈杂的 denoising 中间态（`t ≈ 0.35`，信噪比依然很低）上的表现比 Canny 差很多。Canny 有双阈值 + 非极大值抑制，能更好地区分真实边缘和噪声纹理。

**但此处有另一个更重要的差别**：RALU 是先 VAE decode 再做边缘检测。JiT 没有 VAE，你的 `_edge_mask_from_low_x0` 直接在 **denoising 中间态的 pixel 预测** `x0_low` 上做 Sobel。这个 `x0_low` 在 `t=0.35` 时仍然非常粗糙模糊。

**建议改进**：
- 用更保守的 `ralu_up_ratio`（0.4-0.5），让更多区域进入 full-res
- 或者替代 Sobel 用 `F.max_pool2d - F.avg_pool2d` 做简单的局部对比度检测，对噪声更鲁棒

### 8. `_lift_low_state_to_full` 的高频噪声注入公式有风险

你的 lift 公式：

```python
eps_low = (z_low - t * x0_low) / (1 - t)
x0_full = bilinear_upsample(x0_low)
eps_full = nearest_upsample(eps_low)
eps_full = eps_full + ralu_hf_noise * high_freq_only_noise
z_full = t * x0_full + (1 - t) * eps_full
```

RALU 原版的做法完全不同。它用的是 NT-DM（Noise-Timestep Distribution Matching），通过 `Z`、`alpha`、`beta` 参数来混合上采样后的 latent 和新噪声：

```python
# RALU line 574
latents = beta * latents + alpha * n_prime
```

其中 `n_prime` 是通过 block-diagonal covariance matrix 生成的 **相关噪声**（[RALU line 132-150](file:///C:/Users/Weining%20Zhang/.gemini/antigravity/brain/52f9f736-df89-4d8d-9267-7f5aecd383ba/.system_generated/steps/24/content.md#L119-L152)），不是简单的高频分离。

你的高频噪声注入虽然直觉上合理，但：
1. `ralu_hf_noise=0.25` 这个系数是拍脑袋的，需要仔细 ablation
2. 高频噪声只在频域上分离，没有考虑 patch 间的空间相关性
3. **最关键**：你用 `self.noise_scale`（默认 1.0）乘 randn 来生成 `noise`，但这个 noise 和 flow matching 的 `eps` 不在同一尺度——`eps` 已经乘过 `noise_scale` 了，而你的高频分量是从一个新的 `noise_scale * randn` 里提取的。**这使得高频噪声的幅度与原始噪声轨迹不一致。**

### 9. Stage 2 的 `_mixed_proxy_forward` 完全没有加速效果

> [!IMPORTANT]
> **这是 MVP 方案最大的架构问题。**

```python
def _mixed_proxy_forward(self, z_full, t, labels, mask_full, low_net):
    v_full, _ = self._cfg_v_and_x(self.net, z_full, t, labels)  # ← 已经调用了完整 full-res JiT！
    v_low, _ = self._cfg_v_and_x(low_net, z_low, t, labels)
    v = mask_full * v_full + (1 - mask_full) * v_low
```

Stage 2 仍然调用一次 **完整的** full-res forward pass (`self.net`)，再加一次 low-res forward pass。**这意味着 Stage 2 的代价比原始生成还高**（1 full + 1 low vs 1 full）。

你在计划里也提到了这个问题（"MVP 的 `_mixed_proxy_forward()` 仍调用 full JiT"），但没有意识到这会导致 **Stage 2 实际上是负加速**。

**实际的节省只来自**：
- Stage 1 省掉的步数（低分辨率 token 数少 4 倍 → attention 快 16 倍）
- Stage 3 步数可能比原始总步数少

**但 Stage 2 的代价抵消了大部分收益**。以你的默认配置 `ralu_N=[10,4,8]` 为例：
- 原始 50 步：50 × 2 NFE（Heun）= 100 full forward
- RALU MVP：Stage 1 = 10 × 2 low NFE + Stage 2 = 4 × 2 × (1 full + 1 low) NFE + Stage 3 = 8 × 2 full NFE

实算：20 low + 8 full + 8 low + 16 full = **24 full + 28 low** NFE  
vs 原始 **100 full** NFE

等等，Heun 每步 2 NFE，但最后一步 Euler 只需 1 NFE。重新算：
- 原始 50 步 Heun：49 × 2 + 1 = 99 full NFE
- RALU MVP Heun：
  - Stage 1：9 Heun (18 low) + 1 Euler (1 low) = 19 low NFE + 1 extra x0 prediction = 21 low
  - Stage 2：3 Heun (6 mixed) + 1 Euler (1 mixed) = 7 mixed (= 7 full + 7 low)
  - Stage 3：7 Heun (14 full) + 1 Euler (1 full) = 15 full

总计：**22 full + 28 low** NFE。假设 low ≈ 0.25 full cost → 等效 **29 full** NFE。

这确实比 99 有改善（约 3.4x），但注意 **总步数也只有 22 步了**。如果你直接用原始 JiT 跑 22 步 Heun，cost 是 **43 full** NFE。所以 RALU MVP 的净加速只有 43/29 ≈ **1.48x**，远不是宣传的 4-7x。

### 10. `generate_ralu()` 的 Heun 处理——最后一步应该是 Stage 3 的最后一步

你的 `_ode_step_with_fn` 检查 `t_next >= 1.0 - 1e-6` 来决定是否用 Euler。但每个 Stage 的最后一步的 `t_next` 是那个 Stage 的 `ends[i]`，**只有 Stage 3 的最后一步** `t_next = 1.0`。Stage 1 最后一步 `t_next = 0.35`、Stage 2 最后一步 `t_next = 0.55`，都会使用 Heun。

原始 JiT 的 `generate()` 里特殊处理了最后一步用 Euler（[line 87](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/denoiser.py#L87)），但 Stage 间的过渡步没有特殊处理。**你的实现在这里是正确的**——只有全局最后一步需要 Euler，中间 Stage 的结尾用 Heun 是合理的。✅

---

## 🟠 设计层面的重要考量

### 11. EMA cache 清理的位置和时机

你建议在 `evaluate()` 的 EMA 切换前后清理 cache。但看 [engine_jit.py evaluate()](file:///c:/Users/Weining%20Zhang/Desktop/JiT/JiT/engine_jit.py#L86-L93)：

```python
model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
    ema_state_dict[name] = model_without_ddp.ema_params1[i]
model_without_ddp.load_state_dict(ema_state_dict)
```

这里用的是 `load_state_dict(ema_state_dict)`，而你的 `_ralu_low_net` 是通过 `self.__dict__` 存储的，不会被 `load_state_dict` 影响。

> [!IMPORTANT]
> 更关键的问题是：`load_state_dict(ema_state_dict)` 会改变 `self.net` 的所有参数，但 `_ralu_low_net` 还是旧权重！你的 `reset_ralu_cache` 正确地清除了 cache，但问题是 **时机**。
> 
> 必须在 `load_state_dict()` **之后**、`generate()` **之前** 清除，这样下次调用 `_make_ralu_low_net()` 时会从 EMA 权重复制。你的建议位置是对的。

但还有一个微妙问题：`load_state_dict(ema_state_dict)` **不会改变非 Parameter 的属性**，比如 `self.net.feat_rope` 的 `freqs_cos`/`freqs_sin`。这些在 `__init__` 时已经固定了，不随 EMA 变化。这没问题，因为 RoPE 不需要 EMA。✅

### 12. 训练方向的 `teacher_net` 作为 `Denoiser` 子模块的 EMA 干扰

你的训练方案建议：

```python
model_without_ddp.teacher_net = copy.deepcopy(model_without_ddp.net)
```

但 `Denoiser` 继承 `nn.Module`，如果你给它赋一个 `nn.Module` 属性，**它会被注册为子模块**。这意味着：
- `model_without_ddp.parameters()` 会包含 teacher 的参数
- `update_ema()` 会把 teacher 参数也纳入 EMA 更新
- `optimizer` 的 `param_groups` 已经建好了，teacher 不在里面 → EMA 的 `source_params` 和 `ema_params1/2` 长度不匹配 → **EMA 更新会崩溃**

> [!CAUTION]
> **必须** 用和 `_ralu_low_net` 相同的技巧：`self.__dict__["teacher_net"] = ...` 来避免注册为子模块。或者在 teacher 的所有参数上设置 `requires_grad_(False)` 并确保 EMA 只追踪 `requires_grad=True` 的参数（但原始代码用的是 `list(self.parameters())`，包含所有参数）。

### 13. mixed-token RALU 的 `IndexedVisionRoPE` 维度对齐

你提出的 `IndexedVisionRoPE` 方案：

```python
h = torch.einsum("n,d->nd", coords[:, 0], freqs)  # [N, dim//2]
w = torch.einsum("n,d->nd", coords[:, 1], freqs)  # [N, dim//2]
h = torch.repeat_interleave(h, 2, dim=-1)          # [N, dim]
w = torch.repeat_interleave(w, 2, dim=-1)          # [N, dim]
rope = torch.cat([h, w], dim=-1)                    # [N, dim*2]
```

但原始 `VisionRotaryEmbeddingFast` 的 RoPE 维度是：

```python
half_head_dim = hidden_size // num_heads // 2  # 例如 768/12/2 = 32
freqs = 1/(theta ** (arange(0, 32, 2)[:16] / 32))  # 16 个频率
freqs = repeat(freqs, '... n -> ... (n r)', r=2)     # 32 维
freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)  # 64 维
```

所以 RoPE 的总维度是 `2 * half_head_dim = hidden_size // num_heads = head_dim`。

你的 `IndexedVisionRoPE`：
```python
freqs = 1 / (theta ** (arange(0, dim, 2) / dim))  # dim//2 个频率
h = einsum + repeat_interleave → dim 维
w = einsum + repeat_interleave → dim 维
rope = cat([h, w]) → 2*dim 维
```

如果 `dim = half_head_dim = 32`：
- 你的 `freqs` 有 16 个频率 ✅
- `h` after repeat_interleave = 32 维 ✅
- `w` = 32 维 ✅
- `rope` = 64 维 = `head_dim` ✅

维度对齐是正确的。但你的 `repeat_interleave(h, 2, dim=-1)` 产生的模式是 `[h0, h0, h1, h1, ...]`，而原版的 `repeat(freqs, '... n -> ... (n r)', r=2)` 产生的也是 `[f0, f0, f1, f1, ...]`。✅ **模式一致。**

不过 `rotate_half` 的实现是 `rearrange(x, '... (d r) -> ... d r', r=2)` 然后 swap。这需要 RoPE 的 cos/sin 和 q/k 的维度模式相匹配。你的方案用了相同的 `rotate_half`，所以是兼容的。✅

---

## 📋 总结优先级

| 优先级 | 问题 | 严重程度 | 修复难度 |
|--------|------|----------|----------|
| P0 | #9 Stage 2 负加速 | 架构缺陷 | 高（需要 mixed-token） |
| P0 | #12 teacher_net 破坏 EMA | 运行时崩溃 | 低（用 `__dict__`） |
| P1 | #7 Pixel-space 边缘检测质量差 | 质量退化 | 中（可调参缓解） |
| P1 | #8 高频噪声注入公式不严谨 | 质量退化 | 中（需推导） |
| P1 | #6 torch.compile 编译延迟 | 性能问题 | 低（加 warmup） |
| P2 | #11 EMA cache 清理时机 | 细节正确性 | 低 |

## 🎯 核心结论

1. **MVP 方案能跑通**，但 Stage 2 的 mixed proxy 是伪加速。真实加速比大约 **1.5x-2x**，不是 4-7x。想要大幅加速必须实现 mixed-token forward。

2. **最大风险**是 pixel-space 的边缘检测质量。RALU 原版在 latent space 做 Tweedie denoising 后 VAE 解码再做 Canny，效果远好于在粗糙的 pixel x0 上做 Sobel。建议增加 `ralu_e[0]`（多走几步低分辨率），让 x0 预测更清晰再做边缘检测。

3. **训练方向**必须注意 `teacher_net` 不能作为 `nn.Module` 子模块注册，否则会破坏 EMA 机制。

4. **总体方案的层次结构是合理的**：MVP → mixed-token → distillation 的渐进路线是对的。
