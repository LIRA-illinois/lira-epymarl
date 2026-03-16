import sys
import os
import subprocess
import random


def cli_to_sacred_args(argv):
    options = []
    updates = []

    for arg in argv:
        if not arg.startswith("--"):
            continue
        key_value = arg[2:]
        key = key_value.split("=", 1)[0]
        if key in ["config", "env-config"]:
            options.append(arg)
        else:
            updates.append(key_value)

            # Backward-compatible special case used in sweeps
            if key == "fcn_hidden":
                value = key_value.split("=", 1)[1] if "=" in key_value else ""
                updates.append(f"n_embed={value}")

    return options, updates


if __name__ == "__main__":
    options, updates = cli_to_sacred_args(sys.argv[1:])
    script_path = "src/main.py"
    final_args = [sys.executable, script_path] + options + ["with"] + updates
    os.execl(sys.executable, sys.executable, script_path, *options, "with", *updates)

