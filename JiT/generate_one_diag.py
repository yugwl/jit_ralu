import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用第 4 张卡
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from denoiser import Denoiser


def build_args(cli):
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
        ralu_f0=cli.ralu_f0,
        ralu_N=[cli.low_steps, 0, cli.full_steps],
        ralu_e=[cli.low_end, cli.low_end, 1.0],
        ralu_up_ratio=0.0,
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/cvip/deyu/jit_ralu/checkpoints/jit-l-16/checkpoint-last.pth",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/cvip/deyu/jit_ralu/JiT/result_diag",
    )
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--model", default="JiT-L/16")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--sampling_method", default="heun", choices=["heun", "euler"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--ralu_f0", type=int, default=2)
    parser.add_argument("--low_steps", type=int, default=16)
    parser.add_argument("--low_end", type=float, default=0.35)
    parser.add_argument("--full_steps", type=int, default=24)
    parser.add_argument("--ralu_hf_noise", type=float, default=0.0)
    parser.add_argument("--lift_mode", default="fresh_noise", choices=["fresh_noise", "upsample_eps", "mixed_noise"])
    parser.add_argument("--low_pos_mode", default="scaled", choices=["scaled", "native"])
    parser.add_argument("--no_ralu", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)

    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    args = build_args(cli)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("use_ralu:", args.use_ralu)
    print("ralu_N:", args.ralu_N)
    print("ralu_e:", args.ralu_e)
    print("ralu_hf_noise:", args.ralu_hf_noise)
    print("ralu_lift_mode:", args.ralu_lift_mode)
    print("ralu_low_pos_mode:", args.ralu_low_pos_mode)

    model = Denoiser(args)
    model = load_checkpoint(model, cli.ckpt, use_ema=not cli.no_ema)
    model.to(device)
    model.eval()

    labels = torch.tensor([cli.label], device=device, dtype=torch.long)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        if args.use_ralu:
            outputs = model.generate_ralu_diagnostic(labels)
        else:
            outputs = {"sample": model.generate(labels)}

    prefix = os.path.join(cli.out_dir, f"label{cli.label}_seed{cli.seed}")
    save_sample(outputs["sample"], f"{prefix}_{'ralu_a' if args.use_ralu else 'base'}.png")

    if args.use_ralu:
        save_sample(outputs["x0_low"], f"{prefix}_x0_low.png")
        save_sample(outputs["x0_low_up"], f"{prefix}_x0_low_up.png")
        save_sample(outputs["z_lift"], f"{prefix}_z_lift.png")
        save_sample(outputs["x0_lift_full"], f"{prefix}_x0_lift_full.png")

    print("saved prefix:", prefix)


if __name__ == "__main__":
    main()
