import argparse
import shlex
import subprocess
import sys
from pathlib import Path


class TrainRun:
    def __init__(
        self,
        name,
        model,
        augmentation,
        seeds,
        image_size=128,
        batch_size=64,
        lr=1e-3,
        num_epochs=300,
        early_stop_patience=50,
    ):
        self.name = name
        self.model = model
        self.augmentation = augmentation
        self.seeds = tuple(seeds)
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience

    def default_output_root(self):
        return Path("archive")

    def default_run_name(self):
        return self.name


RUNS = {
    "simple_cnn_seed_105": TrainRun(
        name="simple_cnn_seed_105",
        model="simple_cnn",
        augmentation="baseline",
        seeds=(105,),
        early_stop_patience=20,
    ),
    "ensemble_1": TrainRun(
        name="ensemble_1",
        model="simple_cnn",
        augmentation="ensemble_1",
        seeds=(76, 100, 123),
        early_stop_patience=15,
    ),
    "ensemble_2": TrainRun(
        name="ensemble_2",
        model="simple_cnn",
        augmentation="ensemble_2",
        seeds=(76, 100, 123),
        early_stop_patience=50,
    ),
    "ensemble_3": TrainRun(
        name="ensemble_3",
        model="simple_cnn",
        augmentation="ensemble_2",
        seeds=(76, 100, 101, 102, 103, 123),
        early_stop_patience=50,
    ),
    "simple_cnn_v2": TrainRun(
        name="simple_cnn_v2",
        model="simple_cnn_v2",
        augmentation="ensemble_2",
        seeds=(100, 101, 102, 103, 104),
        early_stop_patience=50,
    ),
}


def _cmd_for_run(run, output_root=None, run_name=None, python_exe=None):
    python_exe = python_exe or sys.executable
    out_root = output_root or run.default_output_root()
    run_group = run_name or run.default_run_name()
    return [
        python_exe,
        "main.py",
        "--model",
        run.model,
        "--augmentation",
        run.augmentation,
        "--seeds",
        *[str(s) for s in run.seeds],
        "--image-size",
        str(run.image_size),
        "--batch-size",
        str(run.batch_size),
        "--lr",
        str(run.lr),
        "--num-epochs",
        str(run.num_epochs),
        "--early-stop-patience",
        str(run.early_stop_patience),
        "--output-root",
        str(out_root),
        "--run-name",
        run_group,
    ]


def _print_cmd(cmd):
    print(" ".join(shlex.quote(c) for c in cmd))


def _build_parser():
    parser = argparse.ArgumentParser(description="Архив конфигов запусков обучения.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Запустить обучение по архивному конфигу")
    p_run.add_argument("run", choices=sorted(RUNS.keys()))
    p_run.add_argument("--output-root", type=Path, default=None)
    p_run.add_argument("--run-name", default=None)
    p_run.add_argument("--dry-run", action="store_true", help="Только напечатать команду, не запускать")

    return parser


def main():
    args = _build_parser().parse_args()
    run = RUNS[args.run]
    cmd = _cmd_for_run(run, output_root=args.output_root, run_name=args.run_name)
    _print_cmd(cmd)
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

