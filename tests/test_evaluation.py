from larvaworld.lib import reg
from larvaworld.lib.process.evaluation import Evaluation


def test_evaluation():
    kws = {
        "refID": reg.conf.Ref.confIDs[-1],
        "cycle_curve_metrics": ["sv", "fov", "foa", "b"],
    }
    ev = Evaluation(**kws)
    assert ev.s_pars.exist_in(ev.target.step_data)
    assert ev.e_pars.exist_in(ev.target.endpoint_data)
