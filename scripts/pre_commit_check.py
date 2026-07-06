"""Pre-commit check: forbid .env files and API key patterns in staged diffs."""

import re
import subprocess
import sys


def main() -> int:
    # 1. Check for staged .env files
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for filename in result.stdout.splitlines():
        if filename.endswith(".env") or "/.env" in filename:
            print(f"ERROR: .env file staged for commit: {filename}")
            return 1

    # 2. Check for API key patterns in staged diff
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "-U0"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    api_pattern = re.compile(r"sk-[a-zA-Z0-9]{20,}")
    if api_pattern.search(diff_result.stdout):
        print("ERROR: API key pattern (sk-xxx) found in staged diff!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
