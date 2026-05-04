import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用第 4 张卡
import argparse
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from denoiser import Denoiser


def build_args():
    return SimpleNamespace(
        # checkpoint / model
        model="JiT-B/32",
        img_size=512,
        class_num=1000,

        # architecture
        attn_dropout=0.0,
        proj_dropout=0.0,

        # training-related attrs required by Denoiser
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=2.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,

        # sampling
        sampling_method="heun",
        num_sampling_steps=50,
        cfg=5.0,
        interval_min=0.1,
        interval_max=1.0,

        # RALU switch
        use_ralu=True,
        ralu_f0=2,
        # ralu_N=[10, 4, 8],
        ralu_N=[12, 4, 10],
        ralu_e=[0.35, 0.55, 1.0],
        ralu_up_ratio=0.3,
        ralu_hf_noise=0.25,
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/cvip/deyu/jit_ralu/checkpoints/jit-b-32/checkpoint-last.pth",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/cvip/deyu/jit_ralu/JiT/result",
    )
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--no_ralu", action="store_true")
    args_cli = parser.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)

    args = build_args()
    if args_cli.no_ralu:
        args.use_ralu = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("use_ralu:", args.use_ralu)

    model = Denoiser(args)
    model = load_checkpoint(model, args_cli.ckpt, use_ema=True)
    model.to(device)
    model.eval()

    labels = torch.tensor([args_cli.label], device=device, dtype=torch.long)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        image = model.generate(labels)

    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)

    out_path = os.path.join(
        args_cli.out_dir,
        f"sample_label{args_cli.label}_{'ralu' if args.use_ralu else 'base'}.png",
    )
    save_image(image, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    main()
