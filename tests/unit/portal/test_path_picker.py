from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.portal import path_picker


def test_pick_directory_uses_macos_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "workspace"

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        path_picker.shutil,
        "which",
        lambda command: "/usr/bin/osascript" if command == "osascript" else None,
    )
    monkeypatch.setattr(
        path_picker,
        "_pick_directory_via_osascript",
        lambda initial_dir=None, *, fallback_dir=None, title="Select folder": selected,
    )

    path, error = path_picker.pick_directory(
        tmp_path, title="Select Larvaworld workspace folder"
    )

    assert path == selected
    assert error is None


def test_pick_directory_macos_cancel_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        path_picker.shutil,
        "which",
        lambda command: "/usr/bin/osascript" if command == "osascript" else None,
    )
    monkeypatch.setattr(
        path_picker,
        "_pick_directory_via_osascript",
        lambda initial_dir=None, *, fallback_dir=None, title="Select folder": None,
    )

    path, error = path_picker.pick_directory(
        tmp_path, title="Select Larvaworld workspace folder"
    )

    assert path is None
    assert error is None


def test_pick_directory_reports_missing_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(path_picker.shutil, "which", lambda _command: None)
    monkeypatch.setattr(
        path_picker,
        "_pick_directory_via_tk",
        lambda initial_dir=None, *, fallback_dir=None, title="Select folder": None,
    )

    path, error = path_picker.pick_directory(tmp_path)

    assert path is None
    assert error == "No folder picker is available in this environment."


def test_pick_directory_wsl_invokes_folder_browser_dialog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    def which(cmd: str) -> str | None:
        if cmd in ("powershell.exe", "wslpath"):
            return f"/mock/{cmd}"
        return None

    monkeypatch.setattr(path_picker.shutil, "which", which)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["wslpath", "-w"]:
            return sp.CompletedProcess(cmd, 0, stdout="C:\\\\proj\n", stderr="")
        if cmd[0] == "powershell.exe":
            script = cmd[-1]
            assert "FolderBrowserDialog" in script
            assert "TopMost" in script
            assert "ShowDialog($owner" in script
            assert "-STA" in cmd
            assert "-NoProfile" in cmd
            return sp.CompletedProcess(
                cmd, 0, stdout="C:\\\\Users\\\\x\\\\pick\n", stderr=""
            )
        if cmd[:2] == ["wslpath", "-u"]:
            return sp.CompletedProcess(
                cmd, 0, stdout="/mnt/c/Users/x/pick\n", stderr=""
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path, title="Pick me")

    assert err is None
    assert picked == Path("/mnt/c/Users/x/pick")
    assert calls[0][:2] == ["wslpath", "-w"]
    assert calls[1][0] == "powershell.exe"
    assert calls[2][:2] == ["wslpath", "-u"]


def test_pick_directory_wsl_cancel_returns_none_tuple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    def which(cmd: str) -> str | None:
        if cmd in ("powershell.exe", "wslpath"):
            return f"/mock/{cmd}"
        return None

    monkeypatch.setattr(path_picker.shutil, "which", which)

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        if cmd[:2] == ["wslpath", "-w"]:
            return sp.CompletedProcess(cmd, 0, stdout="C:\\\\temp\n", stderr="")
        if cmd[0] == "powershell.exe":
            return sp.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path)

    assert picked is None
    assert err is None


def test_pick_directory_wsl_powershell_nonzero_reports_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    def which(cmd: str) -> str | None:
        if cmd in ("powershell.exe", "wslpath"):
            return f"/mock/{cmd}"
        return None

    monkeypatch.setattr(path_picker.shutil, "which", which)

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        if cmd[:2] == ["wslpath", "-w"]:
            return sp.CompletedProcess(cmd, 0, stdout="C:\\\\t\n", stderr="")
        if cmd[0] == "powershell.exe":
            return sp.CompletedProcess(cmd, 1, stdout="", stderr="dialog failed")
        raise AssertionError(cmd)

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path)

    assert picked is None
    assert err is not None
    assert err.startswith("Windows folder picker failed:")
    assert "dialog failed" in err


def test_pick_directory_macos_picker_uses_activate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Darwin")
    captured: list[list[str]] = []

    def which(cmd: str) -> str | None:
        return "/usr/bin/osascript" if cmd == "osascript" else None

    monkeypatch.setattr(path_picker.shutil, "which", which)

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        captured.append(cmd)
        joined = " ".join(cmd)
        assert "activate" in joined
        assert "System Events" in joined
        return sp.CompletedProcess(cmd, 0, stdout="/tmp/picked-folder\n", stderr="")

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path, title="Pick title")

    assert err is None
    assert picked == Path("/tmp/picked-folder")
    assert captured


def test_pick_directory_linux_uses_zenity_before_tk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")
    calls: list[list[str]] = []

    def which(cmd: str) -> str | None:
        return "/usr/bin/zenity" if cmd == "zenity" else None

    monkeypatch.setattr(path_picker.shutil, "which", which)

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[0] == "zenity":
            return sp.CompletedProcess(
                cmd, 0, stdout=str(tmp_path / "zenity-pick") + "\n", stderr=""
            )
        raise AssertionError(f"unexpected subprocess: {cmd}")

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path, title="Linux title")

    assert err is None
    assert picked == tmp_path / "zenity-pick"
    assert calls[0][0] == "zenity"
    assert "--directory" in calls[0]
    assert any(a.startswith("--title=") for a in calls[0])


def test_pick_directory_linux_uses_kdialog_when_zenity_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")

    def which(cmd: str) -> str | None:
        if cmd == "zenity":
            return None
        if cmd == "kdialog":
            return "/usr/bin/kdialog"
        return None

    monkeypatch.setattr(path_picker.shutil, "which", which)

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        assert cmd[0] == "kdialog"
        return sp.CompletedProcess(
            cmd, 0, stdout=str(tmp_path / "kd") + "\n", stderr=""
        )

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path, title="K title")

    assert err is None
    assert picked == tmp_path / "kd"


def test_pick_directory_linux_uses_yad_when_zenity_and_kdialog_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")

    def which(cmd: str) -> str | None:
        if cmd in ("zenity", "kdialog"):
            return None
        if cmd == "yad":
            return "/usr/bin/yad"
        return None

    monkeypatch.setattr(path_picker.shutil, "which", which)

    def fake_run(cmd: list[str], **kwargs: object) -> sp.CompletedProcess[str]:
        assert cmd[0] == "yad"
        return sp.CompletedProcess(
            cmd, 0, stdout=str(tmp_path / "yd") + "\n", stderr=""
        )

    monkeypatch.setattr(path_picker.subprocess, "run", fake_run)

    picked, err = path_picker.pick_directory(tmp_path, title="Y title")

    assert err is None
    assert picked == tmp_path / "yd"


def test_pick_directory_linux_cancel_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")

    def which(cmd: str) -> str | None:
        return "/usr/bin/zenity" if cmd == "zenity" else None

    monkeypatch.setattr(path_picker.shutil, "which", which)
    monkeypatch.setattr(
        path_picker.subprocess,
        "run",
        lambda cmd, **kwargs: sp.CompletedProcess(cmd, 1, stdout="", stderr=""),
    )

    picked, err = path_picker.pick_directory(tmp_path)

    assert picked is None
    assert err is None


def test_pick_directory_linux_picker_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess as sp

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(path_picker.platform, "system", lambda: "Linux")

    def which(cmd: str) -> str | None:
        return "/usr/bin/zenity" if cmd == "zenity" else None

    monkeypatch.setattr(path_picker.shutil, "which", which)
    monkeypatch.setattr(
        path_picker.subprocess,
        "run",
        lambda cmd, **kwargs: sp.CompletedProcess(
            cmd, 2, stdout="", stderr="zenity refused"
        ),
    )

    picked, err = path_picker.pick_directory(tmp_path)

    assert picked is None
    assert err is not None
    assert err.startswith("Linux folder picker failed:")
    assert "zenity refused" in err
