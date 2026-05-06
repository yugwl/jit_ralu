import argparse
import os
import random

import numpy as np
import torch
from torchvision.utils import save_image

from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser("JiT image sampler", add_help=True)

    parser.add_argument("--model", default="JiT-H/32", type=str)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--noise_scale", default=2.0, type=float)
    parser.add_argument("--class_num", default=1000, type=int)

    parser.add_argument("--ckpt", required=True, type=str,
                        help="Path to checkpoint-last.pth")
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--labels", default="0,281,285,951", type=str,
                        help="Comma-separated ImageNet class ids")
    parser.add_argument("--num_per_class", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--sampling_method", default="heun", type=str)
    parser.add_argument("--num_sampling_steps", default=50, type=int)
    parser.add_argument("--cfg", default=2.3, type=float)
    parser.add_argument("--interval_min", default=0.1, type=float)
    parser.add_argument("--interval_max", default=1.0, type=float)

    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--ema_decay1", default=0.9999, type=float)
    parser.add_argument("--ema_decay2", default=0.9996, type=float)

    parser.add_argument("--use_ralu", action="store_true")
    parser.add_argument("--ralu_f0", default=2, type=int)
    parser.add_argument("--ralu_N", type=int, nargs="+", default=[10, 4, 8])
    parser.add_argument("--ralu_e", type=float, nargs="+", default=[0.35, 0.55, 1.0])
    parser.add_argument("--ralu_up_ratio", default=0.3, type=float)
    parser.add_argument("--ralu_hf_noise", default=0.25, type=float)

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    if "model_ema1" in checkpoint:
        state_dict = model.state_dict()
        ema_state_dict = checkpoint["model_ema1"]
        for name, _ in model.named_parameters():
            if name in ema_state_dict:
                state_dict[name] = ema_state_dict[name]
        model.load_state_dict(state_dict, strict=True)

    return model.to(device)


@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = [int(label) for label in args.labels.split(",") if label.strip()]
    batch_labels = []
    for label in labels:
        batch_labels.extend([label] * args.num_per_class)
    batch_labels = torch.tensor(batch_labels, device=device, dtype=torch.long)

    model = Denoiser(args)
    model = load_checkpoint(model, args.ckpt, device)
    model.eval()

    print("device:", device)
    print("model:", args.model)
    print("use_ralu:", args.use_ralu)
    print("labels:", labels)
    print("num_per_class:", args.num_per_class)
    print("seed:", args.seed)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        samples = model.generate(batch_labels)

    samples = ((samples + 1.0) / 2.0).clamp(0, 1)

    for idx, label in enumerate(batch_labels.tolist()):
        class_count = idx % args.num_per_class
        filename = f"label{label:04d}_seed{args.seed}_sample{class_count:02d}.png"
        save_image(samples[idx], os.path.join(args.out_dir, filename))

    grid_path = os.path.join(args.out_dir, f"grid_seed{args.seed}.png")
    save_image(samples, grid_path, nrow=args.num_per_class)
    print("saved images to:", args.out_dir)
    print("saved grid:", grid_path)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
