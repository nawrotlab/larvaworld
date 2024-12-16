import subprocess
from unittest.mock import Mock
from pytest import mark

from larvaworld.cli.main import main


def cli_args(cli_str: str):
    mockfun = Mock()
    main(cli_args=cli_str.split(), mainfun=mockfun)
    run = mockfun.call_args[0][1]
    args = mockfun.call_args[0][2]
    return run, args


def test_cli_entrypoint():
    """
    Is entrypoint script installed?
    """
    result = subprocess.run("larvaworld --help", capture_output=True, shell=True)
    assert result.returncode == 0
    assert b"usage: larvaworld [-h]" in result.stdout


def test_cli_experiment_args():
    """Run an experiment without analysis, without visualization on screen."""
    run, args = cli_args("Exp dish -N 4 -duration 1.0")
    assert args.sim_mode == "Exp"
    assert args.experiment == "dish"
    assert args.Nagents == 4
    assert args.duration == 1.0
    assert args.analysis == False

    assert run.__class__.__name__ == "ExpRun"
    assert run.p.experiment == "dish"
    assert run.p.duration == 1.0


def test_cli_analysis_and_visualization_args():
    """Run an experiment with analysis, with visualization on screen."""
    run, args = cli_args("Exp dispersion -vis_mode video -a")
    assert args.vis_mode == "video"
    assert args.analysis == True

    assert run.screen_manager.vis_mode == "video"


def test_cli_GA_args():
    """Run a genetic algorithm experiment without visualization on screen."""
    run, args = cli_args(
        "Ga realism -refID exploration.30controls -Nagents 10 -Ngenerations 5 -duration 0.5 -bestConfID GA_test_loco -init_mode model"
    )
    assert args.sim_mode == "Ga"
    assert args.experiment == "realism"
    assert args.refID == "exploration.30controls"
    assert args.Nagents == 10
    assert args.Ngenerations == 5
    assert args.duration == 0.5
    assert args.bestConfID == "GA_test_loco"
    assert args.init_mode == "model"

    assert run.__class__.__name__ == "GAlauncher"
    assert run.p.experiment == "realism"
    assert run.p.duration == 0.5
    assert run.p.ga_select_kws.Nagents == 10
    assert run.p.ga_select_kws.Ngenerations == 5
    assert run.p.ga_select_kws.bestConfID == "GA_test_loco"
    assert run.p.ga_select_kws.init_mode == "model"
    assert run.p.ga_eval_kws.refID == "exploration.30controls"

    assert run.selector.Nagents == 10
    assert run.selector.Ngenerations == 5
    assert run.selector.bestConfID == "GA_test_loco"
    assert run.selector.init_mode == "model"
    assert run.evaluator.refID == "exploration.30controls"


def test_cli_replay_args():
    """Run an experiment replay specifying the dataset by its reference ID."""
    run, args = cli_args("Replay -refID exploration.30controls")
    assert args.sim_mode == "Replay"
    assert args.refID == "exploration.30controls"

    assert run.__class__.__name__ == "ReplayRun"
    assert run.p.refID == "exploration.30controls"


def test_cli_evaluation_args():
    """Perform an experiment evaluation run."""
    run, args = cli_args(
        "Eval -refID exploration.30controls -modelIDs RE_NEU_PHI_DEF RE_SIN_PHI_DEF -N 10"
    )
    assert args.sim_mode == "Eval"
    assert args.refID == "exploration.30controls"
    assert args.modelIDs == ["RE_NEU_PHI_DEF", "RE_SIN_PHI_DEF"]
    assert args.N == 10

    assert run.__class__.__name__ == "EvalRun"
    assert run.refID == "exploration.30controls"
    assert run.modelIDs == ["RE_NEU_PHI_DEF", "RE_SIN_PHI_DEF"]
    assert run.N == 10


def xtest_cli_batch_args():
    """Perform an experiment batch run."""
    run, args = cli_args("Batch PItest_off -N 5 -duration 0.5")
    assert args.sim_mode == "Batch"
    assert args.experiment == "PItest_off"
    assert args.N == 5
    assert args.duration == 0.5

    assert run.__class__.__name__ == "BatchRun"
    assert run.p.experiment == "PItest_off"
    assert run.p.conf.Nagents == 5
    assert run.p.conf.duration == 1.0
