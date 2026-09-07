"""
Native directory picker for the portal.

Opens the operating system's own dialog, so a path can be chosen without
typing it, and falls back gracefully where none is available.
"""

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


def _ps_single_quoted_literal(value: str) -> str:
    """Escape for safe embedding in PowerShell single-quoted strings."""
    return value.replace("'", "''")


def _pick_directory_via_windows_dialog(
    initial_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    title: str,
) -> Path | None:
    """WSL2: open a Windows Forms folder picker via PowerShell (foreground / topmost owner)."""
    if not os.getenv("WSL_DISTRO_NAME"):
        return None
    if shutil.which("powershell.exe") is None or shutil.which("wslpath") is None:
        return None

    initial_linux = str(
        resolve_picker_initial_directory(initial_dir, fallback_dir=fallback_dir)
    )
    converted = subprocess.run(
        ["wslpath", "-w", initial_linux],
        capture_output=True,
        text=True,
        check=False,
    )
    if converted.returncode != 0:
        raise RuntimeError(
            converted.stderr.strip() or converted.stdout.strip() or "wslpath -w failed"
        )
    initial_windows = converted.stdout.strip()
    if not initial_windows:
        raise RuntimeError("wslpath -w returned an empty path")

    initial_sq = _ps_single_quoted_literal(initial_windows)
    title_sq = _ps_single_quoted_literal(title)

    script = rf"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$owner = New-Object System.Windows.Forms.Form
$owner.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedToolWindow
$owner.ShowInTaskbar = $false
$owner.TopMost = $true
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.StartPosition = 'CenterScreen'
$owner.Show()
$owner.Activate()
$owner.BringToFront()
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{title_sq}'
$dialog.SelectedPath = '{initial_sq}'
if ([string]::IsNullOrWhiteSpace($dialog.SelectedPath) -or -not (Test-Path -LiteralPath $dialog.SelectedPath)) {{
    $dialog.SelectedPath = [Environment]::GetFolderPath('MyDocuments')
}}
try {{
    $result = $dialog.ShowDialog($owner)
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
        Write-Output $dialog.SelectedPath
    }}
}} finally {{
    $dialog.Dispose()
    $owner.Dispose()
}}
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(detail)

    selected_win = result.stdout.strip()
    if not selected_win:
        return None

    converted_back = subprocess.run(
        ["wslpath", "-u", selected_win],
        capture_output=True,
        text=True,
        check=False,
    )
    if converted_back.returncode != 0:
        raise RuntimeError(
            converted_back.stderr.strip()
            or converted_back.stdout.strip()
            or "wslpath -u failed"
        )
    linux_path = converted_back.stdout.strip()
    return Path(linux_path).expanduser() if linux_path else None


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
        'tell application "System Events" to activate',
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


def _run_linux_gui_directory_picker(cmd: list[str]) -> Path | None:
    """Run one Linux desktop picker; return path, None on cancel, or raise RuntimeError."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0 and stdout:
        return Path(stdout).expanduser()
    if result.returncode != 0 and not stdout and not stderr:
        return None
    if result.returncode != 0:
        raise RuntimeError(stderr or stdout or f"exit code {result.returncode}")
    return None


def _pick_directory_via_linux_desktop(
    initial_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    title: str,
) -> tuple[bool, Path | None]:
    """Try zenity, then kdialog, then yad.

    Returns (used_picker, path_or_none_on_cancel). If ``used_picker`` is False,
    no desktop tool was available and the caller should fall back to Tk.
    """
    initial = resolve_picker_initial_directory(initial_dir, fallback_dir=fallback_dir)
    initial_str = str(initial)
    filename_arg = initial_str if initial_str.endswith("/") else f"{initial_str}/"

    if shutil.which("zenity") is not None:
        cmd = [
            "zenity",
            "--file-selection",
            "--directory",
            f"--title={title}",
            f"--filename={filename_arg}",
        ]
        return True, _run_linux_gui_directory_picker(cmd)

    if shutil.which("kdialog") is not None:
        cmd = [
            "kdialog",
            "--title",
            title,
            "--getexistingdirectory",
            initial_str,
        ]
        return True, _run_linux_gui_directory_picker(cmd)

    if shutil.which("yad") is not None:
        cmd = [
            "yad",
            "--file-selection",
            "--directory",
            f"--title={title}",
            f"--filename={filename_arg}",
        ]
        return True, _run_linux_gui_directory_picker(cmd)

    return False, None


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
    root.lift()
    root.focus_force()
    root.update()
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
    # 1) WSL2 → Windows native picker
    if (
        os.getenv("WSL_DISTRO_NAME")
        and shutil.which("powershell.exe") is not None
        and shutil.which("wslpath") is not None
    ):
        try:
            return (
                _pick_directory_via_windows_dialog(
                    initial_dir, fallback_dir=fallback_dir, title=title
                ),
                None,
            )
        except Exception as exc:
            return None, f"Windows folder picker failed: {exc}"

    # 2) macOS native picker
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

    # 3) Linux desktop (non-WSL) → zenity / kdialog / yad
    if platform.system() == "Linux" and not os.getenv("WSL_DISTRO_NAME"):
        try:
            used, path = _pick_directory_via_linux_desktop(
                initial_dir, fallback_dir=fallback_dir, title=title
            )
        except Exception as exc:
            return None, f"Linux folder picker failed: {exc}"
        if used:
            return path, None

    # 4) Tk fallback
    try:
        selected = _pick_directory_via_tk(
            initial_dir, fallback_dir=fallback_dir, title=title
        )
        if selected is not None:
            return selected, None
    except Exception as exc:
        return None, f"Folder picker failed: {exc}"

    # 5) Nothing available
    return None, "No folder picker is available in this environment."
