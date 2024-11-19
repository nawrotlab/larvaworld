from larvaworld.lib import reg, sim


def test_genetic_algorithm_no_video():
    """Run a genetic algorithm experiment without visualization on screen."""
    exp = "realism"
    ga1 = sim.GAlauncher(experiment=exp, duration=1.0)
    ga1.selector.Ngenerations = 5
    best1 = ga1.run()
    print(best1)
    assert best1 is not None


def test_genetic_algorithm_with_video():
    """Run a genetic algorithm experiment with visualization on screen."""
    exp = "realism"
    p = reg.conf.Ga.expand(exp)
    p.ga_select_kws.Ngenerations = 5
    ga2 = sim.GAlauncher(parameters=p, duration=1.0, screen_kws={"show_display": True})
    best2 = ga2.run()
    print(best2)
    assert best2 is not None
