# generate_one_diag.py 使用说明

`generate_one_diag.py` 用于快速诊断 JiT / RALU 采样效果。它不训练模型，不需要真实 ImageNet，也不计算 FID/IS；只加载指定 checkpoint，生成单张图片，并在 RALU 模式下额外保存中间阶段图用于观察质量变化。

## 基本前提

在服务器上进入项目目录并激活环境：

```bash
cd /home/cvip/deyu/jit_ralu/JiT
conda activate jit
```

确认 checkpoint 存在：

```bash
ls -lh /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth
```

脚本默认使用：

```text
ckpt    = /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth
out_dir = /home/cvip/deyu/jit_ralu/JiT/result_diag_full_0.2
model   = JiT-H/32
label   = 2
seed    = 42
```

注意：脚本里当前写了：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
```

所以默认会使用第 7 号 GPU。若想用别的卡，需要修改脚本这一行，或删除这一行后在命令前使用 `CUDA_VISIBLE_DEVICES=0`。

## 生成一张原版 JiT 图片

不使用 RALU，只跑原始采样路径：

```bash
python generate_one_diag.py \
  --no_ralu \
  --ckpt /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_base_h32 \
  --model JiT-H/32 \
  --label 2 \
  --seed 42
```

输出示例：

```text
/home/cvip/deyu/jit_ralu/JiT/result_base_h32/label2_seed42_base.png
```

## 生成 RALU 两阶段诊断图

`low_full_diag` 是保守诊断模式：低分辨率采样后 lift 到高分辨率，再全分辨率 refine。

```bash
python generate_one_diag.py \
  --ckpt /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_diag_low_full \
  --model JiT-H/32 \
  --label 2 \
  --seed 42 \
  --ralu_mode low_full_diag \
  --sampling_method euler \
  --low_steps 16 \
  --low_end 0.35 \
  --full_steps 24 \
  --lift_mode fresh_noise \
  --ralu_hf_noise 0.0
```

这个模式通常会保存最终图和若干中间图，文件名前缀类似：

```text
label2_seed42_low_full_diag.png
label2_seed42_*.png
```

## 生成 mixed-token RALU 诊断图

`full_mixed_full` 对应三阶段路径：Stage 1 低分辨率、Stage 2 mixed-token、Stage 3 全分辨率 refinement。

```bash
python generate_one_diag.py \
  --ckpt /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_diag_mixed \
  --model JiT-H/32 \
  --label 2 \
  --seed 42 \
  --ralu_mode full_mixed_full \
  --sampling_method euler \
  --stage1_steps 20 \
  --stage1_end 0.30 \
  --mixed_steps 12 \
  --mixed_end 0.55 \
  --stage3_steps 32 \
  --ralu_up_ratio 0.3 \
  --lift_mode fresh_noise \
  --low_pos_mode scaled \
  --ralu_hf_noise 0.0
```

如果 mixed-token 图质量明显差于 base，可以先提高保守性：

```bash
python generate_one_diag.py \
  --ckpt /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_diag_mixed_safe \
  --model JiT-H/32 \
  --label 2 \
  --seed 42 \
  --ralu_mode full_mixed_full \
  --sampling_method euler \
  --stage1_steps 24 \
  --stage1_end 0.35 \
  --mixed_steps 12 \
  --mixed_end 0.60 \
  --stage3_steps 40 \
  --ralu_up_ratio 0.5 \
  --lift_mode fresh_noise \
  --low_pos_mode scaled \
  --ralu_hf_noise 0.0
```

## 固定随机种子复现

脚本用 `--seed` 固定随机种子。同一个 checkpoint、同一组参数、同一个 seed，输出应保持一致。

```bash
python generate_one_diag.py \
  --no_ralu \
  --label 281 \
  --seed 123 \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_seed123
```

换一批图只需要换 seed：

```bash
python generate_one_diag.py --no_ralu --label 281 --seed 124
```

## 更换类别

`--label` 是 ImageNet 类别编号，范围通常是 `0` 到 `999`。

常用测试命令：

```bash
python generate_one_diag.py --no_ralu --label 0   --seed 42 --out_dir /home/cvip/deyu/jit_ralu/JiT/result_label0
python generate_one_diag.py --no_ralu --label 281 --seed 42 --out_dir /home/cvip/deyu/jit_ralu/JiT/result_label281
python generate_one_diag.py --no_ralu --label 285 --seed 42 --out_dir /home/cvip/deyu/jit_ralu/JiT/result_label285
python generate_one_diag.py --no_ralu --label 951 --seed 42 --out_dir /home/cvip/deyu/jit_ralu/JiT/result_label951
```

## 计时 benchmark

开启 `--benchmark` 后，脚本会先 warmup，再多次计时，并打印总耗时和各阶段耗时。

原版 JiT 计时：

```bash
python generate_one_diag.py \
  --no_ralu \
  --benchmark \
  --warmup_runs 1 \
  --timed_runs 3 \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_benchmark_base
```

mixed-token RALU 计时：

```bash
python generate_one_diag.py \
  --ralu_mode full_mixed_full \
  --benchmark \
  --warmup_runs 1 \
  --timed_runs 3 \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_benchmark_mixed
```

输出里重点看：

```text
benchmark_avg_total
benchmark_avg_stage1
benchmark_avg_stage2
benchmark_avg_stage3
```

不同模式下中间阶段名称可能略有不同，以终端实际打印为准。

## 常用参数解释

```text
--ckpt                checkpoint-last.pth 路径
--out_dir             输出图片目录
--label               ImageNet 类别 id
--seed                随机种子
--model               模型名，例如 JiT-H/32、JiT-B/32
--img_size            输入分辨率；默认会从 checkpoint 的 pos_embed 自动推断
--noise_scale         噪声尺度；默认 512 用 2.0，256 用 1.0
--cfg                 CFG 强度；默认 512 用 2.5，256 用 2.4
--sampling_method     euler 或 heun
--num_sampling_steps  原版 JiT 采样步数
--no_ralu             关闭 RALU，使用原版 JiT
--no_ema              不使用 EMA 权重
--benchmark           开启计时模式
```

RALU 相关：

```text
--ralu_mode           low_full_diag 或 full_mixed_full
--ralu_f0             低分辨率下采样倍数，通常为 2
--ralu_up_ratio       mixed-token 阶段进入 full-res 的边缘区域比例
--low_steps           low_full_diag 的低分辨率步数
--low_end             low_full_diag 的低分辨率结束时间
--full_steps          low_full_diag 的全分辨率 refine 步数
--stage1_steps        full_mixed_full 的 Stage 1 步数
--stage1_end          full_mixed_full 的 Stage 1 结束时间
--mixed_steps         full_mixed_full 的 mixed-token 步数
--mixed_end           full_mixed_full 的 mixed-token 结束时间
--stage3_steps        full_mixed_full 的全分辨率 refine 步数
--lift_mode           fresh_noise、upsample_eps、mixed_noise
--low_pos_mode        scaled 或 native
--ralu_hf_noise       lift 时加入的高频噪声强度
```

## JiT-H/32 推荐起步命令

先看原版质量：

```bash
python generate_one_diag.py \
  --no_ralu \
  --ckpt /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_base_h32 \
  --model JiT-H/32 \
  --label 2 \
  --seed 42
```

再看 mixed-token RALU：

```bash
python generate_one_diag.py \
  --ckpt /home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth \
  --out_dir /home/cvip/deyu/jit_ralu/JiT/result_mixed_h32 \
  --model JiT-H/32 \
  --label 2 \
  --seed 42 \
  --ralu_mode full_mixed_full \
  --stage1_steps 20 \
  --stage1_end 0.30 \
  --mixed_steps 12 \
  --mixed_end 0.55 \
  --stage3_steps 32 \
  --ralu_up_ratio 0.3
```

这两张图使用同一个 label 和 seed，适合直接比较 base 与 RALU 的质量差异。
