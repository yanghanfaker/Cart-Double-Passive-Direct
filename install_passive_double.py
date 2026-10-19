"""Run on Spark: python3 install_passive_double.py /home/test/Robot/IsaacLab."""

import argparse
import ast
import shutil
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("isaaclab", type=Path)
args = parser.parse_args()
root = args.isaaclab.expanduser().resolve()
target_dir = root / "source/isaaclab_tasks/isaaclab_tasks/core/cartpole"
init_file = target_dir / "__init__.py"
source = Path(__file__).resolve().with_name("passive_double_cartpole.py")
if not (target_dir / "cartpole_direct_env.py").is_file():
    raise SystemExit(f"Expected Isaac Lab 3 Cartpole source not found: {target_dir}")
ast.parse(source.read_text(encoding="utf-8"))
original = init_file.read_text(encoding="utf-8")
hook = "from . import passive_double_cartpole  # single-input double pendulum"
updated = original if hook in original else original + "\n" + hook + "\n"
ast.parse(updated)
target = target_dir / source.name
stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
if target.exists():
    shutil.copy2(target, target.with_name(target.name + f".{stamp}.bak"))
if updated != original:
    shutil.copy2(init_file, init_file.with_name(f"__init__.py.{stamp}.bak"))
shutil.copy2(source, target)
if updated != original:
    init_file.write_text(updated, encoding="utf-8")
print(f"Installed: {target}")
print("Registered task: Isaac-Cart-Double-Passive-Direct")
print("Original task files preserved; __init__.py has one import added.")
