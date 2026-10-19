"""Locate study assets in the current repository. Author: James Edward Ball."""

import os
from pathlib import Path
import re
import subprocess

PUBLIC_REPOSITORY = "JamesEBall/geofm-semantic-structure"
PACKAGE = Path(__file__).resolve().parents[2]


def repository():
    """Use a mirror, the current GitHub origin, or the public source-archive URL."""
    override = os.environ.get("GEOFM_ASSET_REPOSITORY")
    if override:
        return override
    origin = subprocess.run(
        ["git", "-C", str(PACKAGE), "remote", "get-url", "origin"],
        capture_output=True, text=True, check=False,
    )
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:)([^/\s]+/[^/\s]+?)(?:\.git)?/?",
        origin.stdout.strip(),
    ) if origin.returncode == 0 else None
    if not match:
        return PUBLIC_REPOSITORY
    result = subprocess.run(
        ["gh", "api", "repos/" + match.group(1), "--jq", ".full_name"],
        check=True, capture_output=True, text=True,
    )
    name = result.stdout.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
        raise ValueError("GitHub returned an invalid asset repository name")
    return name
