import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 使用第 4 张卡
import time
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from denoiser import Denoiser


def _patch_size_from_model_name(model_name):
    return int(model_name.split("/")[-1])


def _model_size_from_model_name(model_name):
    return model_name.split("/")[0].split("-")[-1]


def _recommended_cfg(model_name, img_size):
    model_size = _model_size_from_model_name(model_name)
    if model_size == "B":
        return 3.0
    if model_size == "L":
        return 2.5 if img_size == 512 else 2.4
    if model_size == "H":
        return 2.3 if img_size == 512 else 2.2
    raise ValueError(f"Unsupported JiT model size in model name: {model_name}")


def apply_checkpoint_defaults(cli):
    ckpt = torch.load(cli.ckpt, map_location="cpu")
    pos_embed = ckpt["model"]["net.pos_embed"]
    grid_size = int(pos_embed.shape[1] ** 0.5)
    assert grid_size * grid_size == pos_embed.shape[1]

    if cli.img_size is None:
        cli.img_size = grid_size * _patch_size_from_model_name(cli.model)
    if cli.noise_scale is None:
        cli.noise_scale = 2.0 if cli.img_size == 512 else 1.0
    if cli.cfg is None:
        cli.cfg = _recommended_cfg(cli.model, cli.img_size)

    print("checkpoint pos_embed:", tuple(pos_embed.shape))
    print("inferred grid_size:", grid_size)
    print("runtime img_size:", cli.img_size)
    print("runtime noise_scale:", cli.noise_scale)
    print("runtime cfg:", cli.cfg)
    return cli


def build_args(cli):
    if cli.ralu_mode == "full_mixed_full":
        ralu_N = [cli.stage1_steps, cli.mixed_steps, cli.stage3_steps]
        ralu_e = [cli.stage1_end, cli.mixed_end, 1.0]
    else:
        ralu_N = [cli.low_steps, 0, cli.full_steps]
        ralu_e = [cli.low_end, cli.low_end, 1.0]

    return SimpleNamespace(
        # checkpoint / model
        model=cli.model,
        img_size=cli.img_size,
        class_num=1000,

        # architecture
        attn_dropout=0.0,
        proj_dropout=0.0,

        # training-related attrs required by Denoiser
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=cli.noise_scale,
        ema_decay1=0.9999,
        ema_decay2=0.9996,

        # sampling
        sampling_method=cli.sampling_method,
        num_sampling_steps=cli.num_sampling_steps,
        cfg=cli.cfg,
        interval_min=cli.interval_min,
        interval_max=cli.interval_max,

        # conservative two-stage Pixel-RALU diagnostic
        use_ralu=not cli.no_ralu,
        ralu_mode=cli.ralu_mode,
        ralu_f0=cli.ralu_f0,
        ralu_N=ralu_N,
        ralu_e=ralu_e,
        ralu_up_ratio=cli.ralu_up_ratio,
        ralu_hf_noise=cli.ralu_hf_noise,
        ralu_lift_mode=cli.lift_mode,
        ralu_low_pos_mode=cli.low_pos_mode,
    )


def load_checkpoint(model, ckpt_path, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    if use_ema and "model_ema1" in ckpt:
        state = model.state_dict()
        ema_state = ckpt["model_ema1"]
        for name, _ in model.named_parameters():
            if name in ema_state:
                state[name] = ema_state[name]
        model.load_state_dict(state, strict=True)

    return model


def save_sample(tensor, path):
    image = (tensor + 1.0) / 2.0
    image = image.clamp(0, 1)
    save_image(image, path)


def save_tensor_outputs(tensor, prefix, name):
    if tensor.size(0) == 1:
        save_sample(tensor, f"{prefix}_{name}.png")
        return

    save_sample(tensor, f"{prefix}_{name}_grid.png")
    for sample_idx in range(tensor.size(0)):
        save_sample(
            tensor[sample_idx:sample_idx + 1],
            f"{prefix}_{name}_sample{sample_idx:02d}.png",
        )


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_outputs(model, args, labels, device):
    sync_device(device)
    start = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        if args.use_ralu:
            if args.ralu_mode == "full_mixed_full":
                outputs = model.generate_ralu_full_mixed_diagnostic(labels)
            else:
                outputs = model.generate_ralu_diagnostic(labels)
        else:
            outputs = {"sample": model.generate(labels)}
    sync_device(device)
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def print_timings(title, seconds, outputs=None):
    print(f"{title}: {seconds:.4f}s")
    timings = outputs.get("timings") if outputs is not None else None
    if timings:
        for name, value in timings.items():
            print(f"  {name}: {value:.4f}s")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/cvip/deyu/jit_ralu/JiT/result_diag_full_0.2",
    )
    parser.add_argument("--label", type=int, default=281)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", default="JiT-H/32")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--sampling_method", default="euler", choices=["heun", "euler"])
    parser.add_argument("--num_sampling_steps", type=int, default=64)
    parser.add_argument("--ralu_mode", default="low_full_diag", choices=["low_full_diag", "full_mixed_full"])
    parser.add_argument("--ralu_f0", type=int, default=2)
    parser.add_argument("--ralu_up_ratio", type=float, default=0.3)
    parser.add_argument("--low_steps", type=int, default=16)
    parser.add_argument("--low_end", type=float, default=0.35)
    parser.add_argument("--full_steps", type=int, default=24)
    parser.add_argument("--stage1_steps", type=int, default=20)
    parser.add_argument("--stage1_end", type=float, default=0.30)
    parser.add_argument("--mixed_steps", type=int, default=20)
    parser.add_argument("--mixed_end", type=float, default=0.55)
    parser.add_argument("--stage3_steps", type=int, default=10)
    parser.add_argument("--ralu_hf_noise", type=float, default=0.0)
    parser.add_argument("--lift_mode", default="fresh_noise", choices=["fresh_noise", "upsample_eps", "mixed_noise"])
    parser.add_argument("--low_pos_mode", default="scaled", choices=["scaled", "native"])
    parser.add_argument("--no_ralu", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--timed_runs", type=int, default=3)
    cli = parser.parse_args()
    cli = apply_checkpoint_defaults(cli)

    os.makedirs(cli.out_dir, exist_ok=True)

    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    args = build_args(cli)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("use_ralu:", args.use_ralu)
    print("ralu_mode:", args.ralu_mode)
    print("ralu_N:", args.ralu_N)
    print("ralu_e:", args.ralu_e)
    print("ralu_up_ratio:", args.ralu_up_ratio)
    print("ralu_hf_noise:", args.ralu_hf_noise)
    print("ralu_lift_mode:", args.ralu_lift_mode)
    print("ralu_low_pos_mode:", args.ralu_low_pos_mode)

    model = Denoiser(args)
    model = load_checkpoint(model, cli.ckpt, use_ema=not cli.no_ema)
    model.to(device)
    model.eval()

    labels = torch.full((cli.num_samples,), cli.label, device=device, dtype=torch.long)

    if cli.benchmark:
        for i in range(cli.warmup_runs):
            outputs, elapsed = generate_outputs(model, args, labels, device)
            print_timings(f"warmup {i + 1}/{cli.warmup_runs}", elapsed, outputs)

        elapsed_runs = []
        stage_sums = {}
        outputs = None
        for i in range(cli.timed_runs):
            outputs, elapsed = generate_outputs(model, args, labels, device)
            elapsed_runs.append(elapsed)
            print_timings(f"timed {i + 1}/{cli.timed_runs}", elapsed, outputs)
            timings = outputs.get("timings", {})
            for name, value in timings.items():
                stage_sums[name] = stage_sums.get(name, 0.0) + float(value)

        avg_elapsed = sum(elapsed_runs) / len(elapsed_runs)
        print(f"benchmark_avg_total: {avg_elapsed:.4f}s")
        if stage_sums:
            for name, value in stage_sums.items():
                print(f"benchmark_avg_{name}: {value / len(elapsed_runs):.4f}s")
    else:
        outputs, elapsed = generate_outputs(model, args, labels, device)
        print_timings("single_run", elapsed, outputs)

    prefix = os.path.join(cli.out_dir, f"label{cli.label}_seed{cli.seed}")
    suffix = args.ralu_mode if args.use_ralu else "base"
    save_tensor_outputs(outputs["sample"], prefix, suffix)

    if args.use_ralu:
        for name, tensor in outputs.items():
            if name == "sample" or not torch.is_tensor(tensor) or tensor.ndim != 4:
                continue
            save_tensor_outputs(tensor, prefix, name)

    print("saved prefix:", prefix)


if __name__ == "__main__":
    main()
