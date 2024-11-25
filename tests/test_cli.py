from cli_test_helpers import shell

cli_entry_point = "larvaworld"


def test_run_experiment_no_analysis_no_video():
    """Run an experiment without analysis, without visualization on screen."""
    result = shell(f"{cli_entry_point} Exp dish -N 4 -duration 1.0")
    assert result.exit_code == 0


def test_run_experiment_with_analysis_no_video():
    """Run an experiment with analysis, without visualization on screen."""
    result = shell(f"{cli_entry_point} Exp chemorbit -N 4 -duration 1.0 -a")
    assert result.exit_code == 0


def test_run_experiment_with_analysis_with_video():
    """Run an experiment with analysis, with visualization on screen."""
    result = shell(
        f"{cli_entry_point} Exp dispersion -N 4 -duration 2.0 -vis_mode video -a"
    )
    assert result.exit_code == 0


def test_run_experiment_no_analysis_with_image():
    """Run an experiment without analysis, with visualization on screen."""
    result = shell(
        f"{cli_entry_point} Exp dish -N 4 -duration 2.0 -vis_mode image"
    )
    assert result.exit_code == 0


def test_run_ga_without_video():
    """Run a genetic algorithm experiment without visualization on screen."""
    result = shell(
        f"{cli_entry_point} Ga realism -refID exploration.30controls -Nagents 10 -Ngenerations 5 -duration 0.5 -bestConfID GA_test_loco -init_mode model"
    )
    assert result.exit_code == 0


def test_run_ga_with_video():
    """Run a genetic algorithm experiment with visualization on screen."""
    result = shell(
        f"{cli_entry_point} Ga realism -refID exploration.30controls -Nagents 10 -Ngenerations 5 -duration 0.5 -bestConfID GA_test_loco -init_mode model -vis_mode image"
    )
    assert result.exit_code == 0


def test_run_replay_on_dataset_by_ID():
    """Run an experiment replay specifying the dataset by its reference ID."""
    result = shell(f"{cli_entry_point} Replay -refID exploration.30controls")
    assert result.exit_code == 0


def test_run_replay_on_dataset_by_dir():
    """Run an experiment replay specifying the dataset by its directory path."""
    result = shell(
        f"{cli_entry_point} Replay -refDir SchleyerGroup/processed/exploration/30controls"
    )
    assert result.exit_code == 0


def xtest_batch_run():
    """Perform an experiment batch run."""
    result = shell(f"{cli_entry_point} Batch PItest_off -N 5 -duration 0.5")
    assert result.exit_code == 0


def test_evaluation_run():
    """Perform an experiment evaluation run."""
    result = shell(
        f"{cli_entry_point} Eval -refID exploration.30controls -modelIDs RE_NEU_PHI_DEF RE_SIN_PHI_DEF -N 10"
    )
    assert result.exit_code == 0


# def xtest_can_run_as_python_module():
#     """Run the CLI as a Python module."""
#     result = subprocess.run(  # noqa: S603
#         [sys.executable, "-m", "larvaworld", "--help"],
#         check=True,
#         capture_output=True,
#     )
#     assert result.returncode == 0
#     assert b"larvaworld [OPTIONS]" in result.stdout

# from typer.testing import CliRunner

# from larvaworld.cli import main

# runner = CliRunner()


# def test_help():
#     """The help message includes the CLI name."""
#     result = runner.invoke(main, ["--help"])
#     assert result.exit_code == 0
#     assert "Add the arguments and print the result" in result.stdout
