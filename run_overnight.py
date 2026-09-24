import os
import pty
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_script(script_path, output_file):
    """Run a repository script and stream its output to stdout and a log file."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    )

    command = [sys.executable, str(REPO_ROOT / script_path)]

    with open(REPO_ROOT / output_file, "a") as file:
        master, slave = pty.openpty()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=slave,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            close_fds=True,
        )
        os.close(slave)

        while True:
            try:
                line = os.read(master, 1024).decode()
                if not line:
                    break
                print(line, end="")
                file.write(line)
                file.flush()
            except OSError:
                break

        process.wait()
        os.close(master)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == "__main__":
    script = Path("scripts") / "centralized_multimodal_decision.py"
    for _ in range(10):
        run_script(script, "multimodal.txt")
