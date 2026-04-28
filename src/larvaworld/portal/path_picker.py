from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess


__all__ = ["pick_directory", "resolve_picker_initial_directory"]


def resolve_picker_initial_directory(
    initial_dir: Path | None = None,
    fallback_dir: Path | None = None,
) -> Path:
    candidate = (initial_dir or fallback_dir or Path.home()).expanduser()
    if candidate.is_dir():
        return candidate

    fallbacks = [
        candidate.parent,
        fallback_dir.expanduser() if fallback_dir is not None else None,
        Path.home() / "Documents",
        Path.home(),
    ]
    for fallback in fallbacks:
        if fallback is not None and fallback.is_dir():
            return fallback
    return Path.home()


def _pick_directory_via_windows_dialog(
    initial_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    title: str,
) -> Path | None:
    if not os.getenv("WSL_DISTRO_NAME") or shutil.which("powershell.exe") is None:
        return None

    initial_linux = str(
        resolve_picker_initial_directory(initial_dir, fallback_dir=fallback_dir)
    )
    initial_windows = ""
    converted = subprocess.run(
        ["wslpath", "-w", initial_linux],
        capture_output=True,
        text=True,
        check=False,
    )
    if converted.returncode == 0:
        initial_windows = converted.stdout.strip()
    initial_windows = initial_windows.replace("'", "''")
    picker_title = title.replace("'", "''")

    script = rf"""
Add-Type -AssemblyName PresentationFramework
$defaultRoot = [Environment]::GetFolderPath('MyDocuments')
$initialDir = '{initial_windows}'
if ([string]::IsNullOrWhiteSpace($initialDir) -or -not (Test-Path $initialDir)) {{
    $initialDir = $defaultRoot
}}
$dialog = New-Object Microsoft.Win32.OpenFileDialog
$dialog.Title = '{picker_title}'
$dialog.CheckFileExists = $false
$dialog.CheckPathExists = $true
$dialog.ValidateNames = $false
$dialog.FileName = 'Select this folder'
$dialog.InitialDirectory = $initialDir
$dialog.Filter = 'Folders|*.folder'
if ($dialog.ShowDialog() -eq $true) {{
    $selectedPath = Split-Path -Parent $dialog.FileName
    if ($selectedPath) {{
        Write-Output $selectedPath
    }}
}}
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    selected = result.stdout.strip()
    if not selected:
        return None
    converted = subprocess.run(
        ["wslpath", "-u", selected],
        capture_output=True,
        text=True,
        check=False,
    )
    linux_path = converted.stdout.strip()
    return Path(linux_path) if linux_path else None


def _pick_directory_via_osascript(
    initial_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    title: str,
) -> Path | None:
    if shutil.which("osascript") is None:
        return None

    default_dir = str(
        resolve_picker_initial_directory(initial_dir, fallback_dir=fallback_dir)
    )
    default_dir = default_dir.replace("\\", "\\\\").replace('"', '\\"')
    picker_title = title.replace("\\", "\\\\").replace('"', '\\"')
    script_lines = [
        f'set defaultLocation to POSIX file "{default_dir}"',
        "try",
        (
            "set chosenFolder to choose folder with prompt "
            f'"{picker_title}" default location defaultLocation'
        ),
        "on error number -128",
        'return ""',
        "end try",
        "POSIX path of chosenFolder",
    ]
    args: list[str] = ["osascript"]
    for line in script_lines:
        args.extend(["-e", line])

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "osascript failed"
        raise RuntimeError(message)

    selected = result.stdout.strip()
    return Path(selected).expanduser() if selected else None


def _pick_directory_via_tk(
    initial_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    title: str,
) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            title=title,
            initialdir=str(
                resolve_picker_initial_directory(initial_dir, fallback_dir=fallback_dir)
            ),
        )
    finally:
        root.destroy()
    return Path(selected).expanduser() if selected else None


def pick_directory(
    initial_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    title: str = "Select folder",
) -> tuple[Path | None, str | None]:
    if os.getenv("WSL_DISTRO_NAME") and shutil.which("powershell.exe") is not None:
        try:
            return (
                _pick_directory_via_windows_dialog(
                    initial_dir, fallback_dir=fallback_dir, title=title
                ),
                None,
            )
        except Exception as exc:
            return None, f"Windows folder picker failed: {exc}"

    if platform.system() == "Darwin" and shutil.which("osascript") is not None:
        try:
            return (
                _pick_directory_via_osascript(
                    initial_dir, fallback_dir=fallback_dir, title=title
                ),
                None,
            )
        except Exception as exc:
            return None, f"macOS folder picker failed: {exc}"

    try:
        selected = _pick_directory_via_tk(
            initial_dir, fallback_dir=fallback_dir, title=title
        )
        if selected is not None:
            return selected, None
    except Exception as exc:
        return None, f"Folder picker failed: {exc}"

    return None, "No folder picker is available in this environment."
