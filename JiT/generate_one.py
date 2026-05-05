import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用第 4 张卡
import argparse
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from denoiser import Denoiser


def _patch_size_from_model_name(model_name):
    return int(model_name.split("/")[-1])


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
        cli.cfg = 2.5 if cli.img_size == 512 else 2.4

    print("checkpoint pos_embed:", tuple(pos_embed.shape))
    print("inferred grid_size:", grid_size)
    print("runtime img_size:", cli.img_size)
    return cli


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

        # RALU switch
        use_ralu=not cli.no_ralu,
        ralu_f0=cli.ralu_f0,
        ralu_N=[cli.low_steps, 0, cli.full_steps],
        ralu_e=[cli.low_end, cli.low_end, 1.0],
        ralu_up_ratio=0.0,
        ralu_hf_noise=cli.ralu_hf_noise,
        ralu_lift_mode=cli.lift_mode,
        ralu_low_pos_mode=cli.low_pos_mode,
    )


def load_checkpoint(model, ckpt_path, ema="ema1"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    if ema in ("ema1", "ema2"):
        ema_key = "model_ema1" if ema == "ema1" else "model_ema2"
        if ema_key not in ckpt:
            raise KeyError(f"{ema_key} not found in checkpoint")
        state = model.state_dict()
        ema_state = ckpt[ema_key]
        for name, _ in model.named_parameters():
            if name in ema_state:
                state[name] = ema_state[name]
        model.load_state_dict(state, strict=True)
    elif ema != "raw":
        raise ValueError(f"Unsupported ema mode: {ema}")

    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/cvip/deyu/jit_ralu/checkpoints/jit-l-16/checkpoint-last.pth",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/cvip/deyu/jit_ralu/JiT/result",
    )
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model", default="JiT-L/16")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--sampling_method", default="heun", choices=["heun", "euler"])
    parser.add_argument("--num_sampling_steps", type=int, default=80)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ema", default="ema1", choices=["ema1", "ema2", "raw"])
    parser.add_argument("--ralu_f0", type=int, default=2)
    parser.add_argument("--low_steps", type=int, default=16)
    parser.add_argument("--low_end", type=float, default=0.35)
    parser.add_argument("--full_steps", type=int, default=24)
    parser.add_argument("--ralu_hf_noise", type=float, default=0.0)
    parser.add_argument("--lift_mode", default="fresh_noise", choices=["fresh_noise", "upsample_eps", "mixed_noise"])
    parser.add_argument("--low_pos_mode", default="scaled", choices=["scaled", "native"])
    parser.add_argument("--no_ralu", action="store_true")
    args_cli = parser.parse_args()
    args_cli = apply_checkpoint_defaults(args_cli)

    os.makedirs(args_cli.out_dir, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    args = build_args(args_cli)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("use_ralu:", args.use_ralu)
    print("model:", args.model)
    print("img_size:", args.img_size)
    print("noise_scale:", args.noise_scale)
    print("cfg:", args.cfg_scale if hasattr(args, "cfg_scale") else args.cfg)
    print("num_sampling_steps:", args.num_sampling_steps)
    print("num_samples:", args_cli.num_samples)
    print("ema:", args_cli.ema)
    print("ralu_N:", args.ralu_N)
    print("ralu_e:", args.ralu_e)
    print("ralu_lift_mode:", args.ralu_lift_mode)
    print("ralu_low_pos_mode:", args.ralu_low_pos_mode)

    model = Denoiser(args)
    model = load_checkpoint(model, args_cli.ckpt, ema=args_cli.ema)
    model.to(device)
    model.eval()

    labels = torch.full((args_cli.num_samples,), args_cli.label, device=device, dtype=torch.long)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        images = model.generate(labels)

    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1)

    tag = "ralu" if args.use_ralu else "base"
    if args_cli.num_samples == 1:
        out_path = os.path.join(
            args_cli.out_dir,
            f"sample_label{args_cli.label}_{tag}_seed{args_cli.seed}_{args_cli.ema}_{args.num_sampling_steps}steps.png",
        )
        save_image(images, out_path)
        print("saved:", out_path)
    else:
        grid_path = os.path.join(
            args_cli.out_dir,
            f"sample_label{args_cli.label}_{tag}_seed{args_cli.seed}_{args_cli.ema}_{args.num_sampling_steps}steps_grid.png",
        )
        save_image(images, grid_path, nrow=min(4, args_cli.num_samples))
        print("saved:", grid_path)

        for i, image in enumerate(images):
            out_path = os.path.join(
                args_cli.out_dir,
                f"sample_label{args_cli.label}_{tag}_seed{args_cli.seed}_{args_cli.ema}_{args.num_sampling_steps}steps_{i:02d}.png",
            )
            save_image(image, out_path)


if __name__ == "__main__":
    main()
