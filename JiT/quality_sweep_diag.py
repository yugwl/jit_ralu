import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import torch

from denoiser import Denoiser
from generate_one_diag import (
    apply_checkpoint_defaults,
    apply_runtime_config,
    build_args,
    generate_outputs,
    load_checkpoint,
    save_sample,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate one 50-step baseline image and a quality sweep of clearer JiT/RALU candidates."
    )
    parser.add_argument(
        "--ckpt",
        default="/home/cvip/deyu/jit_ralu/checkpoints/jit-h-32/checkpoint-last.pth",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/cvip/deyu/jit_ralu/JiT/quality_sweep_label281",
    )
    parser.add_argument("--label", type=int, default=281)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model", default="JiT-H/32")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--baseline_method", default="euler", choices=["euler", "heun"])
    parser.add_argument("--baseline_steps", type=int, default=50)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the baseline and three high-signal candidates.",
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Also save RALU diagnostic intermediate tensors such as z_after_mixed.",
    )
    return parser


def make_cli(base_cli, **overrides):
    cli = SimpleNamespace(**vars(base_cli))
    for key, value in overrides.items():
        setattr(cli, key, value)
    return cli


def cli_to_record(name, cli, elapsed, outputs):
    timings = outputs.get("timings", {}) if isinstance(outputs, dict) else {}
    return {
        "name": name,
        "elapsed_sec": elapsed,
        "label": cli.label,
        "seed": cli.seed,
        "use_ralu": not cli.no_ralu,
        "ralu_mode": cli.ralu_mode,
        "sampling_method": cli.sampling_method,
        "num_sampling_steps": cli.num_sampling_steps,
        "cfg": cli.cfg,
        "stage1_steps": cli.stage1_steps,
        "stage1_end": cli.stage1_end,
        "mixed_steps": cli.mixed_steps,
        "mixed_end": cli.mixed_end,
        "stage3_steps": cli.stage3_steps,
        "ralu_up_ratio": cli.ralu_up_ratio,
        "lift_mode": cli.lift_mode,
        "low_pos_mode": cli.low_pos_mode,
        "ralu_hf_noise": cli.ralu_hf_noise,
        "timings": timings,
    }


def save_record_files(out_dir, records):
    json_path = os.path.join(out_dir, "manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = [
        "name",
        "elapsed_sec",
        "label",
        "seed",
        "use_ralu",
        "ralu_mode",
        "sampling_method",
        "num_sampling_steps",
        "cfg",
        "stage1_steps",
        "stage1_end",
        "mixed_steps",
        "mixed_end",
        "stage3_steps",
        "ralu_up_ratio",
        "lift_mode",
        "low_pos_mode",
        "ralu_hf_noise",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})

    print("manifest:", json_path)
    print("manifest_csv:", csv_path)


def build_sweep(base_cli):
    cfg = base_cli.cfg
    variants = [
        (
            "000_baseline_50",
            dict(
                no_ralu=True,
                sampling_method=base_cli.baseline_method,
                num_sampling_steps=base_cli.baseline_steps,
            ),
        ),
        (
            "001_baseline_64_euler",
            dict(no_ralu=True, sampling_method="euler", num_sampling_steps=64),
        ),
        (
            "002_baseline_50_heun",
            dict(no_ralu=True, sampling_method="heun", num_sampling_steps=50),
        ),
        (
            "003_ralu_balanced",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                stage1_steps=20,
                stage1_end=0.30,
                mixed_steps=12,
                mixed_end=0.55,
                stage3_steps=32,
                ralu_up_ratio=0.30,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
        (
            "004_ralu_more_stage3",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                stage1_steps=20,
                stage1_end=0.30,
                mixed_steps=10,
                mixed_end=0.50,
                stage3_steps=40,
                ralu_up_ratio=0.35,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
        (
            "005_ralu_conservative_mixed",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                stage1_steps=24,
                stage1_end=0.35,
                mixed_steps=8,
                mixed_end=0.50,
                stage3_steps=36,
                ralu_up_ratio=0.45,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
        (
            "006_ralu_high_upratio",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                stage1_steps=20,
                stage1_end=0.30,
                mixed_steps=12,
                mixed_end=0.55,
                stage3_steps=32,
                ralu_up_ratio=0.55,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
        (
            "007_ralu_late_mixed",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                stage1_steps=28,
                stage1_end=0.40,
                mixed_steps=8,
                mixed_end=0.58,
                stage3_steps=32,
                ralu_up_ratio=0.40,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
        (
            "008_ralu_cfg_plus",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                cfg=round(cfg + 0.2, 3),
                stage1_steps=20,
                stage1_end=0.30,
                mixed_steps=10,
                mixed_end=0.50,
                stage3_steps=40,
                ralu_up_ratio=0.40,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
        (
            "009_ralu_cfg_minus",
            dict(
                no_ralu=False,
                sampling_method="euler",
                ralu_mode="full_mixed_full",
                cfg=round(max(1.0, cfg - 0.2), 3),
                stage1_steps=20,
                stage1_end=0.30,
                mixed_steps=10,
                mixed_end=0.50,
                stage3_steps=40,
                ralu_up_ratio=0.40,
                lift_mode="fresh_noise",
                low_pos_mode="scaled",
                ralu_hf_noise=0.0,
            ),
        ),
    ]
    return variants


def run_one(model, base_cli, device, name, overrides, save_intermediates=False):
    cli = make_cli(base_cli, **overrides)
    args = build_args(cli)
    apply_runtime_config(model, args)

    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    labels = torch.full((cli.num_samples,), cli.label, device=device, dtype=torch.long)
    print(f"\n=== {name} ===")
    print(
        "mode={} method={} steps={} cfg={} ralu_N={} ralu_e={} up_ratio={}".format(
            "ralu" if args.use_ralu else "base",
            args.sampling_method,
            args.num_sampling_steps,
            args.cfg,
            args.ralu_N,
            args.ralu_e,
            args.ralu_up_ratio,
        )
    )

    start = time.perf_counter()
    outputs, elapsed = generate_outputs(model, args, labels, device)
    elapsed = time.perf_counter() - start if elapsed is None else elapsed

    sample_path = os.path.join(cli.out_dir, f"{name}.png")
    save_sample(outputs["sample"], sample_path)
    print(f"elapsed: {elapsed:.4f}s")
    print("saved:", sample_path)

    if save_intermediates and args.use_ralu:
        stem = os.path.join(cli.out_dir, name)
        for output_name, tensor in outputs.items():
            if output_name == "sample" or not torch.is_tensor(tensor) or tensor.ndim != 4:
                continue
            save_sample(tensor, f"{stem}_{output_name}.png")

    return cli_to_record(name, cli, elapsed, outputs)


@torch.no_grad()
def main():
    parser = build_parser()
    cli = parser.parse_args()
    cli = apply_checkpoint_defaults(cli)

    os.makedirs(cli.out_dir, exist_ok=True)

    # Fill fields expected by generate_one_diag.build_args().
    cli.sampling_method = cli.baseline_method
    cli.num_sampling_steps = cli.baseline_steps
    cli.ralu_mode = "full_mixed_full"
    cli.ralu_f0 = 2
    cli.ralu_up_ratio = 0.3
    cli.low_steps = 16
    cli.low_end = 0.35
    cli.full_steps = 24
    cli.stage1_steps = 20
    cli.stage1_end = 0.30
    cli.mixed_steps = 12
    cli.mixed_end = 0.55
    cli.stage3_steps = 32
    cli.ralu_hf_noise = 0.0
    cli.lift_mode = "fresh_noise"
    cli.low_pos_mode = "scaled"
    cli.no_ralu = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model_args = build_args(cli)
    model = Denoiser(model_args)
    model = load_checkpoint(model, cli.ckpt, use_ema=not cli.no_ema)
    model.to(device)
    model.eval()

    sweep = build_sweep(cli)
    if cli.quick:
        keep = {"000_baseline_50", "003_ralu_balanced", "004_ralu_more_stage3", "005_ralu_conservative_mixed"}
        sweep = [item for item in sweep if item[0] in keep]

    records = []
    for name, overrides in sweep:
        record = run_one(
            model,
            cli,
            device,
            name,
            overrides,
            save_intermediates=cli.save_intermediates,
        )
        records.append(record)
        save_record_files(cli.out_dir, records)

    print("\nDone. Review images in:", cli.out_dir)


if __name__ == "__main__":
    main()
