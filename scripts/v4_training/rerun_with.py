#!/usr/bin/env python3
"""Repeat a logged stress.py run with a few arguments changed, then score it.

Reads the "start" row of an earlier run's .log (its full argument list), applies the overrides,
and runs `stress.py train` followed by `stress.py eval` with the same arguments, so a model change
is compared with its baseline under an otherwise identical recipe.

    rerun_with.py runs/stress/hs_m4a.log --out runs/stress/hs_rec --set recency=True
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"command", "out"}


def cli(args):
    flags = []
    for k, v in args.items():
        if k in SKIP or v == "None":
            continue
        name = "--" + k.replace("_", "-")
        if k == "cpp_sampler":
            flags.append(name if v == "True" else "--no-cpp-sampler")
        elif v in ("True", "False"):
            if v == "True":
                flags.append(name)
        elif v.startswith("["):
            flags += [name] + [str(x) for x in json.loads(v)]
        else:
            flags += [name, v]
    return flags


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", type=Path, help="the baseline run's .log")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="argument overrides (stress.py names)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true", help="train only (choose on validation before opening test)")
    a = p.parse_args()
    start = next(json.loads(line) for line in a.log.open() if '"event": "start"' in line)
    args = dict(start["args"])
    for item in a.set:
        k, v = item.split("=", 1)
        args[k.replace("-", "_")] = v
    for command in (() if a.skip_train else ("train",)) + (() if a.skip_eval else ("eval",)):
        cmd = [sys.executable, "-u", "-W", "ignore", str(HERE / "stress.py"), command, "--out", str(a.out)] + cli(args)
        print(json.dumps({"event": "run", "command": command, "argv": cmd[4:]}), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
