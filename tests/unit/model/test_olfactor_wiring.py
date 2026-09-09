"""Every olfactor states how it reaches locomotion, rather than inheriting it.

`brute_force` selects between the two couplings of the olfactory sensor: False
feeds the turner, so the animal steers up the gradient; True suppresses that
output and only interrupts locomotion when concentration falls.

`ModuleDB.module_conf` hands out the shared per-mode default dict and updates it
in place, so an omitted argument silently kept whatever the previous caller had
passed. Every `*forager*` model in the registry was built right after the
`_brute` navigator variant and picked up its True this way - 236 models with a
sensory coupling nothing had asked for. The model factory now states the flag on
every olfactor it builds.
"""

from __future__ import annotations

import pytest

from larvaworld.lib import reg


@pytest.mark.fast
class TestOlfactorWiring:
    def test_only_models_named_brute_use_the_brute_force_coupling(self):
        brute = []
        for mid in reg.conf.Model.confIDs:
            o = reg.conf.Model.getID(mid).brain.olfactor
            if o is not None and o.brute_force:
                brute.append(mid)
        assert brute, "the deliberate _brute variants have disappeared"
        leaked = [mid for mid in brute if "_brute" not in mid]
        assert leaked == [], leaked[:10]

    def test_the_forager_and_the_navigator_share_one_olfactor(self):
        """The pair the model-comparison tutorial contrasts."""
        nav = reg.conf.Model.getID("navigator").brain.olfactor
        assert reg.conf.Model.getID("max_forager").brain.olfactor == nav
        assert reg.conf.Model.getID("forager").brain.olfactor == nav

    def test_the_factory_states_the_flag_rather_than_omitting_it(self):
        """A source-level guard: `olf_kws` must not fall back to the shared default."""
        from pathlib import Path

        import larvaworld

        src = (
            Path(larvaworld.__file__).parent
            / "lib"
            / "model"
            / "modules"
            / "module_modes.py"
        ).read_text(encoding="utf-8")
        i = src.index("def olf_kws(")
        assert "brute_force=brute_force" in src[i : i + 900]
