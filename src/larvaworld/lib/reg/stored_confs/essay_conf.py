"""
Stored configurations for multi-experiment essays.

An essay groups several related experiments into one study -- rover versus
sitter, the double-patch assay, chemotaxis -- together with the analysis that
compares their results.
"""

from __future__ import annotations
from typing import Any, Optional

import shutil

from .... import SIM_DIR
from ... import reg, util, funcs
from ...param import Larva_Distro, Life

# LarvaGroup import - deep import required due to circular dependency
from ...reg.larvagroup import LarvaGroup

__all__: list[str] = [
    "Essay_dict",
    "Essay",
    "RvsS_Essay",
    "DoublePatch_Essay",
    "Chemotaxis_Essay",
]


class Essay:
    """Base class for a multi-experiment study.

    An essay runs a set of related experiments, then analyses their
    datasets both individually and together.
    """

    def __init__(
        self,
        type: str,
        essay_id: Optional[str] = None,
        N: int = 5,
        enrichment=None,
        collections: list[str] = ["pose", "brain"],
        screen_kws: dict[str, Any] = {},
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Build the essay and resolve its output directory.

        Args:
            type: The essay's name.
            enrichment: The enrichment applied to each experiment's dataset.
            collections: The output groups recorded.
            video: Whether to record video.
            N: The number of larvae per group.
            **kwargs: Further essay settings.
        """
        if enrichment is None:
            enrichment = reg.gen.EnrichConf().nestedConf
        self.screen_kws = screen_kws
        self.N = N
        self.show = show
        self.type = type
        self.enrichment = enrichment
        self.collections = collections
        if essay_id is None:
            essay_id = f'{type}_{reg.config.next_idx(id=type, conftype="Essay")}'
        self.essay_id = essay_id
        self.path = f"{SIM_DIR}/essays/{type}/{self.essay_id}"
        self.data_dir = f"{self.path}/data"
        self.plot_dir = f"{self.path}/plots"
        self.exp_dict = {}
        self.datasets = {}
        self.figs = {}
        self.results = {}

    def conf(self, exp: str, id: str, dur: float, lgs: Any, env: Any, **kwargs: Any):
        """Build one experiment configuration for this essay.

        Args:
            exp: The experiment name.
            id: The run identifier.
            dur: The run duration in minutes.
            lgs: The larva groups to place.
            env: The environment configuration.
            **kwargs: Further experiment settings.

        Returns:
            The experiment configuration.
        """
        c = reg.gen.Exp(
            duration=dur,
            env_params=env,
            larva_groups=lgs,
            experiment=None,
            enrichment=self.enrichment,
            collections=self.collections,
            **kwargs,
        ).nestedConf
        c["essay_run_label"] = id
        return c

    def run(self):
        """Run every experiment in the essay.

        Returns:
            The datasets produced, grouped by experiment.
        """
        from ...sim import ExpRun

        print(f'Running essay "{self.essay_id}"')
        for exp, cs in self.exp_dict.items():
            print(f"Running {len(cs)} versions of experiment {exp}")
            self.datasets[exp] = [
                ExpRun(
                    id=self._generate_run_id(c.pop("essay_run_label", None)),
                    parameters=c,
                    screen_kws=self.screen_kws,
                ).simulate()
                for c in cs
            ]

        return self.datasets

    def _generate_run_id(self, label: Optional[str]) -> Optional[str]:
        """Generate a run identifier for one of the essay's experiments.

        Args:
            exp: The experiment name.

        Returns:
            The run identifier.
        """
        if label is None:
            return None
        idx = reg.config.next_idx(label, conftype="Exp")
        return f"{label}_{idx}"

    def anal(self):
        """Analyse the essay's results and render its figures.

        Returns:
            The rendered figures.
        """
        self.global_anal()
        # raise
        for exp, ds0 in self.datasets.items():
            if (
                ds0 is not None
                and len(ds0) != 0
                and all([d0 is not None for d0 in ds0])
            ):
                self.analyze(exp=exp, ds0=ds0)

        shutil.rmtree(self.path, ignore_errors=True)
        return self.figs, self.results

    def analyze(self, exp, ds0):
        """Analyse one experiment's datasets.

        Args:
            exp: The experiment name.
            ds0: The datasets it produced.

        Returns:
            The figures and any derived results.
        """
        pass
        # return {}, None

    def global_anal(self):
        """Analyse the essay's experiments together.

        Returns:
            The cross-experiment figures.
        """
        pass


class RvsS_Essay(Essay):
    """The rover-versus-sitter feeding essay.

    Compares the two foraging strategies across path length, food intake,
    starvation, substrate quality and refeeding.
    """

    def __init__(self, all_figs=False, N=1, **kwargs):
        """Build the rover-versus-sitter essay.

        Args:
            all_figs: Whether to render the full figure set.
            **kwargs: Further essay settings.
        """
        super().__init__(
            type="RvsS",
            N=N,
            enrichment=reg.gen.EnrichConf.spatial_proc(),
            collections=["pose", "brain", "gut"],
            **kwargs,
        )

        self.all_figs = all_figs
        self.qs = [1.0, 0.75, 0.5, 0.25, 0.15]
        self.hs = [0, 1, 2, 3, 4]
        self.durs = [10, 15, 20]
        self.dur = 5
        self.h_refeeding = 3
        self.dur_refeeding = 120
        self.dur_pathlength = 20

        self.substrates = ["Agar", "Yeast"]
        self.exp_dict = {
            **self.pathlength_exp(),
            **self.intake_exp(),
            **self.starvation_exp(),
            **self.quality_exp(),
            **self.refeeding_exp(),
        }

        from ...plot import diff_df

        self.mdiff_df, row_colors = diff_df(mIDs=["rover", "sitter"])

    def RvsS_env(self, on_food=True):
        """Build the essay's environment.

        Args:
            on_food: Whether food is present.

        Returns:
            The environment configuration.
        """
        grid = reg.gen.FoodGrid() if on_food else None
        return reg.gen.Env(
            arena=reg.gen.Arena(geometry="rectangular", dims=(0.02, 0.02)),
            food_params=reg.gen.FoodConf(food_grid=grid),
        ).nestedConf

    def GTRvsS(self, **kwargs):
        """Build the rover and sitter larva groups.

        Args:
            **kwargs: Group settings.

        Returns:
            The group entries.
        """
        return reg.larvagroup.GTRvsS(expand=True, N=self.N, **kwargs)

    def pathlength_exp(self):
        """Build the path-length experiment.

        Args:
            **kwargs: Experiment settings overriding the defaults.

        Returns:
            The experiment entry, comparing distance covered on and off food.
        """
        dur = self.dur_pathlength
        exp = "PATHLENGTH"
        confs = []
        for n, nb in zip(self.substrates, [False, True]):
            kws = {
                "env": self.RvsS_env(on_food=nb),
                "lgs": self.GTRvsS(),
                "id": f"{exp}_{n}_{dur}min",
                "dur": dur,
                "exp": exp,
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def intake_exp(self):
        """Build the food-intake experiment.

        Args:
            **kwargs: Experiment settings overriding the defaults.

        Returns:
            The experiment entry, comparing cumulative intake.
        """
        exp = "AD LIBITUM INTAKE"
        confs = []
        for dur in self.durs:
            kws = {
                "env": self.RvsS_env(on_food=True),
                "lgs": self.GTRvsS(),
                "id": f"{exp}_{dur}min",
                "dur": dur,
                "exp": exp,
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def starvation_exp(self):
        """Build the starvation experiment.

        Args:
            **kwargs: Experiment settings overriding the defaults.

        Returns:
            The experiment entry, comparing intake after increasing starvation.
        """
        exp = "POST-STARVATION INTAKE"
        confs = []
        for h in self.hs:
            kws = {
                "env": self.RvsS_env(on_food=True),
                "lgs": self.GTRvsS(h_starved=h),
                "id": f"{exp}_{h}h_{self.dur}min",
                "dur": self.dur,
                "exp": exp,
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def quality_exp(self):
        """Build the substrate-quality experiment.

        Args:
            **kwargs: Experiment settings overriding the defaults.

        Returns:
            The experiment entry, comparing intake across substrate qualities.
        """
        exp = "REARING-DEPENDENT INTAKE"
        confs = []
        for q in self.qs:
            kws = {
                "env": self.RvsS_env(on_food=True),
                "lgs": self.GTRvsS(q=q),
                "id": f"{exp}_{q}_{self.dur}min",
                "dur": self.dur,
                "exp": exp,
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def refeeding_exp(self):
        """Build the refeeding experiment.

        Args:
            **kwargs: Experiment settings overriding the defaults.

        Returns:
            The experiment entry, comparing intake when food is restored.
        """
        exp = "REFEEDING AFTER 3h STARVED"
        h = self.h_refeeding
        dur = self.dur_refeeding
        kws = {
            "env": self.RvsS_env(on_food=True),
            "lgs": self.GTRvsS(h_starved=h),
            "id": f"{exp}_{h}h_{dur}min",
            "dur": dur,
            "exp": exp,
        }
        return {exp: [self.conf(**kws)]}

    def get_entrylist(self, datasets, substrates, durs, qs, hs, G):
        """Build the experiment entries for one condition sweep.

        Args:
            durs: The run durations.
            qs: The substrate qualities.
            hs: The starvation durations.
            N: The number of larvae per group.
            age: The larvae's age.

        Returns:
            The experiment entries.
        """
        entrylist = []
        pathlength_ls = util.flatten_list(
            [[rf'{s} $for^{"R"}$', rf'{s} $for^{"S"}$'] for s in substrates]
        )
        ls0 = [r"$for^{R}$", r"$for^{S}$"]
        kws00 = {
            "leg_cols": ["black", "white"],
            "markers": ["D", "s"],
        }
        for exp, dds in datasets.items():
            Ndds = len(dds)
            ds = util.flatten_list(dds)
            ls = (
                pathlength_ls
                if exp == "PATHLENGTH"
                else util.flatten_list([ls0] * Ndds)
            )
            kws0 = {"datasets": ds, "labels": ls, "save_as": exp, **kws00}

            if exp == "PATHLENGTH":
                kws = {
                    "xlabel": r"time on substrate $(min)$",
                    "scaled": False,
                    "unit": "cm",
                    **kws0,
                }
                plotID = "pathlength"
            elif exp == "AD LIBITUM INTAKE":
                kws = {
                    "xlabel": r"Time spent on food $(min)$",
                    "coupled_labels": durs,
                    "ks": ["sf_am_V"],
                    **kws0,
                }
                plotID = "barplot"
            elif exp == "POST-STARVATION INTAKE":
                kws = {
                    "xlabel": r"Food deprivation $(h)$",
                    "coupled_labels": hs,
                    "ks": ["f_am"],
                    "ylabel": "Food intake",
                    "scale": 1000,
                    **kws0,
                }
                plotID = "lineplot"
            elif exp == "REARING-DEPENDENT INTAKE":
                kws = {
                    "xlabel": "Food quality (%)",
                    "coupled_labels": [int(q * 100) for q in qs],
                    "ks": ["sf_am_V"],
                    **kws0,
                }
                plotID = "barplot"
            elif exp == "REFEEDING AFTER 3h STARVED":
                kws = {"scaled": True, "filt_amount": True, **kws0}
                plotID = "food intake (timeplot)"
            else:
                raise
            entry = G.entry(ID=plotID, name=exp, **util.AttrDict(kws))
            entrylist.append(entry)
        return entrylist

    def global_anal(self):
        """Render the essay's cross-experiment figures.

        Returns:
            The rendered figures.
        """
        self.entrylist = self.get_entrylist(
            datasets=self.datasets,
            substrates=self.substrates,
            durs=self.durs,
            qs=self.qs,
            hs=self.hs,
            G=reg.graphs,
        )
        kwargs = {"save_to": self.plot_dir, "show": self.show}

        self.figs["RvsS summary"] = reg.graphs.run(
            ID="RvsS summary",
            entrylist=self.entrylist,
            title=f"ROVERS VS SITTERS ESSAY (N={self.N})",
            mdiff_df=self.mdiff_df,
            **kwargs,
        )

        for e in self.entrylist:
            self.figs[e["key"]] = reg.graphs.run(ID=e["plotID"], **e["args"], **kwargs)

    def analyze(self, exp, ds0):
        """Analyse one experiment of the essay.

        Args:
            exp: The experiment name.
            ds0: The datasets it produced.

        Returns:
            The figures and any derived results.
        """
        if self.all_figs:
            entry = [e for e in self.entrylist if e["name"] == exp][0]
            kws = entry["args"]
            RS_leg_cols = ["black", "white"]
            markers = ["D", "s"]
            ls = [r"$for^{R}$", r"$for^{S}$"]
            shorts = ["f_am", "sf_am_Vg", "sf_am_V", "sf_am_A", "sf_am_M"]
            pars = reg.getPar(shorts)

            #
            def dsNls(ds0, lls=None):
                if lls is None:
                    lls = util.flatten_list([ls] * len(ds0))
                dds = util.flatten_list(ds0)
                deb_dicts = util.flatten_list([d.load_dicts("deb") for d in dds])
                return {
                    "datasets": dds,
                    "labels": lls,
                    "deb_dicts": deb_dicts,
                    "save_to": self.plot_dir,
                    "leg_cols": RS_leg_cols,
                    "markers": markers,
                }

            if exp == "PATHLENGTH":
                pass

            elif exp == "AD LIBITUM INTAKE":
                kwargs = {
                    **dsNls(ds0),
                    "coupled_labels": self.durs,
                    "xlabel": r"Time spent on food $(min)$",
                }
                for s, p in zip(shorts, pars):
                    self.figs[f"{exp} {p}"] = reg.graphs.dict["barplot"](
                        ks=[s], save_as=f"2_AD_LIBITUM_{p}.pdf", **kwargs
                    )

            elif exp == "POST-STARVATION INTAKE":
                kwargs = {
                    **dsNls(ds0),
                    "coupled_labels": self.hs,
                    "xlabel": r"Food deprivation $(h)$",
                }
                for ii in ["feeding"]:
                    self.figs[ii] = reg.graphs.dict["deb"](
                        mode=ii,
                        save_as=f"3_POST-STARVATION_{ii}.pdf",
                        include_egg=False,
                        label_epochs=False,
                        **kwargs,
                    )
                for s, p in zip(shorts, pars):
                    self.figs[f"{exp} {p}"] = reg.graphs.dict["lineplot"](
                        par_shorts=[s], save_as=f"3_POST-STARVATION_{p}.pdf", **kwargs
                    )

            elif exp == "REARING-DEPENDENT INTAKE":
                kwargs = {
                    **dsNls(ds0),
                    "coupled_labels": [int(q * 100) for q in self.qs],
                    "xlabel": "Food quality (%)",
                }
                for s, p in zip(shorts, pars):
                    self.figs[f"{exp} {p}"] = reg.graphs.dict["barplot"](
                        ks=[s], save_as=f"4_REARING_{p}.pdf", **kwargs
                    )

            elif exp == "REFEEDING AFTER 3h STARVED":
                h = self.h_refeeding
                n = f"5_REFEEDING_after_{h}h_starvation_"
                kwargs = dsNls(ds0)
                self.figs[f"{exp} food-intake"] = reg.graphs.dict[
                    "food intake (timeplot)"
                ](scaled=True, save_as=f"{n}scaled_intake.pdf", **kwargs)
                self.figs[f"{exp} food-intake(filt)"] = reg.graphs.dict[
                    "food intake (timeplot)"
                ](
                    scaled=True,
                    filt_amount=True,
                    save_as=f"{n}scaled_intake_filt.pdf",
                    **kwargs,
                )
                for s, p in zip(shorts, pars):
                    self.figs[f"{exp} {p}"] = reg.graphs.dict["timeplot"](
                        par_shorts=[s],
                        show_first=False,
                        subfolder=None,
                        save_as=f"{n}{p}.pdf",
                        **kwargs,
                    )


class DoublePatch_Essay(Essay):
    """The double-patch foraging essay.

    Places larvae between two food patches and compares how long each
    group spends on each, across substrate qualities.
    """

    def __init__(
        self,
        substrates=["sucrose", "standard", "cornmeal"],
        N=10,
        dur=5.0,
        olfactor=True,
        feeder=True,
        arena_dims=(0.24, 0.24),
        patch_x=0.06,
        patch_radius=0.025,
        **kwargs,
    ):
        """Build the double-patch essay.

        Args:
            substrates: The substrate qualities compared.
            N: The number of larvae per group.
            dur: The run duration in minutes.
            olfactor: Whether the larvae can smell.
            feeder: Whether the larvae can feed.
            variable_habit: Whether feeding habit varies between groups.
            **kwargs: Further essay settings.
        """
        super().__init__(
            N=N,
            type="DoublePatch",
            enrichment=reg.gen.EnrichConf.patch_proc(),
            **kwargs,
        )
        self.arena_dims = arena_dims
        self.patch_x = patch_x
        self.patch_radius = patch_radius
        self.substrates = substrates
        self.dur = dur
        self.mID0s = ["rover", "sitter"]
        if olfactor:
            if feeder:
                suf = "_forager"
                self.mode = "foragers"
            else:
                suf = "_nav"
                self.mode = "navigators"
        else:
            if feeder:
                suf = ""
                self.mode = "feeders"
            else:
                suf = "_loco"
                self.mode = "locomotors"
        self.mIDs = [f"{mID0}{suf}" for mID0 in self.mID0s]

        self.ms = reg.conf.Model.getID(self.mIDs)
        self.exp_dict = self.time_ratio_exp()

        from ...plot import diff_df

        self.mdiff_df, row_colors = diff_df(mIDs=self.mID0s, ms=self.ms)

    def get_larvagroups(self, age=120.0):
        """Build the essay's larva groups.

        Args:
            age: The larvae's age.

        Returns:
            The group entries.
        """

        def lg(id, c, mID):
            l = reg.gen.LarvaGroup(
                group_id=id,
                model=mID,
                color=c,
                distribution={"N": self.N, "scale": (0.005, 0.005)},
                life_history=Life(age=age),
                sample=reg.default_refID,
            )
            return l.entry()

        return util.AttrDict.merge_dicts(
            [
                lg(id=id, c=c, mID=mID)
                for mID, c, id in zip(self.mIDs, ["blue", "red"], ["rover", "sitter"])
            ]
        )

    def get_sources(self, type="standard", q=1.0, Cpeak=2.0, Cscale=0.0002):
        """Build the two food patches.

        Args:
            type: The substrate they hold.

        Returns:
            The source entries.
        """
        kws0 = {
            "r": self.patch_radius,
            "c": "green",
            "a": 0.1,
            "sub": [q, type],
            "group": "Patch",
            "o": ["Odor", Cpeak, Cscale],
        }

        return util.AttrDict(
            {
                **reg.gen.Food(pos=(-self.patch_x, 0.0), **kws0).entry("Left_patch"),
                **reg.gen.Food(pos=(self.patch_x, 0.0), **kws0).entry("Right_patch"),
            }
        )

    def patch_env(self, type="standard", q=1.0, o="G"):
        """Build the two-patch environment.

        Args:
            type: The substrate the patches hold.

        Returns:
            The environment configuration.
        """
        if o == "G":
            odorscape = reg.gen.GaussianValueLayer()
            Cpeak, Cscale = 2.0, 0.0002
        else:
            raise

        kws = {
            "arena": reg.gen.Arena(dims=self.arena_dims, geometry="rectangular"),
            "food_params": reg.gen.FoodConf(
                source_units=self.get_sources(
                    type=type, q=q, Cpeak=Cpeak, Cscale=Cscale
                )
            ),
            "odorscape": odorscape,
        }

        return reg.gen.Env(**kws)

    def time_ratio_exp(self):
        # exp = 'double_patch'
        """Build the patch-residency experiments, one per substrate.

        Returns:
            The experiment entries.
        """
        confs = {}
        for n in self.substrates:
            kws = {
                "duration": self.dur,
                "env_params": self.patch_env(type=n),
                "larva_groups": self.get_larvagroups(),
                "experiment": "double_patch",
                # 'trials': {},
                "collections": self.collections,
                "enrichment": self.enrichment,
            }

            confs[n] = [reg.gen.Exp(**kws).nestedConf]
        return util.AttrDict(confs)

    def global_anal(self):
        """Render the essay's cross-experiment figures.

        Returns:
            The rendered figures.
        """
        kwargs = {
            "datasets": self.datasets,
            "save_to": self.plot_dir,
            "show": self.show,
            "title": f"DOUBLE PATCH ESSAY (N={self.N}, duration={self.dur}')",
            "mdiff_df": self.mdiff_df,
        }

        self.figs[f"{self.mode}_fig1"] = reg.graphs.run(
            ID="double-patch summary", name=f"{self.mode}_fig1", ks=None, **kwargs
        )
        self.figs[f"{self.mode}_fig2"] = reg.graphs.run(
            ID="double-patch summary",
            name=f"{self.mode}_fig2",
            ks=["tur_tr", "tur_N_mu", "pau_tr", "cum_d", "f_am", "on_food_tr"],
            **kwargs,
        )

    def analyze(self, exp, ds0):
        """Analyse one experiment of the essay.

        Args:
            exp: The experiment name.
            ds0: The datasets it produced.

        Returns:
            The figures and any derived results.
        """
        pass


class Chemotaxis_Essay(Essay):
    """The chemotaxis essay.

    Compares how model variants navigate an odor gradient.
    """

    def __init__(self, dur=5.0, gain=300.0, mode=1, **kwargs):
        """Build the chemotaxis essay.

        Args:
            dur: The run duration in minutes.
            **kwargs: Further essay settings.
        """
        super().__init__(
            type="Chemotaxis", enrichment=reg.gen.EnrichConf.source_proc(), **kwargs
        )
        self.time_ks = ["c_odor1", "dc_odor1"]
        self.dur = dur
        self.gain = gain
        if mode == 1:
            self.models = self.get_models1(gain)
        elif mode == 2:
            self.models = self.get_models2(gain)
        elif mode == 3:
            self.models = self.get_models3(gain)
        elif mode == 4:
            self.models = self.get_models4(gain)

        from ...plot import diff_df

        self.mdiff_df, row_colors = diff_df(
            mIDs=list(self.models.keys()), ms=[v.model for v in self.models.values()]
        )
        self.exp_dict = self.chemo_exps(self.models)

    def get_models1(self, gain):
        """Build the model set comparing olfactory gain.

        Args:
            **kwargs: Group settings.

        Returns:
            The larva groups.
        """
        m = reg.conf.Model.getID("navigator")
        o = "brain.olfactor"

        mW = m.update_nestdict_copy(
            {f"{o}.gain_dict.Odor": gain, f"{o}.perception": "log"}
        )
        mWlin = m.update_nestdict_copy(
            {f"{o}.gain_dict.Odor": gain, f"{o}.perception": "linear"}
        )
        mC = m.update_nestdict_copy({f"{o}.gain_dict.Odor": 0})
        mT = m.update_nestdict_copy(
            {
                f"{o}.gain_dict.Odor": gain,
                f"{o}.perception": "log",
                f"{o}.brute_force": True,
            }
        )
        mTlin = m.update_nestdict_copy(
            {
                f"{o}.gain_dict.Odor": gain,
                f"{o}.perception": "linear",
                f"{o}.brute_force": True,
            }
        )

        T = "Tastekin"
        W = "Wystrach"
        models = {
            f"{T} (log)": {"model": mT, "color": "red"},
            f"{T} (lin)": {"model": mTlin, "color": "darkred"},
            f"{W} (log)": {"model": mW, "color": "lightgreen"},
            f"{W} (lin)": {"model": mWlin, "color": "darkgreen"},
            "controls": {"model": mC, "color": "magenta"},
        }
        return util.AttrDict(models)

    def get_models2(self, gain):
        """Build the model set comparing turner modes.

        Args:
            **kwargs: Group settings.

        Returns:
            The larva groups.
        """
        cols = util.N_colors(6)
        i = 0
        models = {}
        for Tmod in ["NEU", "SIN"]:
            for Ifmod in ["PHI", "SQ", "DEF"]:
                m = reg.conf.Model.getID(f"RE_{Tmod}_{Ifmod}_DEF_nav")
                models[f"{Tmod}_{Ifmod}"] = {
                    "model": m.update_nestdict_copy(
                        {
                            "brain.olfactor.brute_force": True,
                            "brain.olfactor.gain_dict.Odor": gain,
                            "brain.interference.attenuation": 0.1,
                            "brain.interference.attenuation_max": 0.0,
                        }
                    ),
                    "color": cols[i],
                }
                i += 1
        return util.AttrDict(models)

    def get_models3(self, gain):
        """Build the model set comparing crawl-bend coupling.

        Args:
            **kwargs: Group settings.

        Returns:
            The larva groups.
        """
        cols = util.N_colors(6)
        i = 0
        models = {}
        for Tmod in ["NEU", "SIN"]:
            for Ifmod in ["PHI", "SQ", "DEF"]:
                m = reg.conf.Model.getID(f"RE_{Tmod}_{Ifmod}_DEF_nav")
                models[f"{Tmod}_{Ifmod}"] = {
                    "model": m.update_nestdict_copy(
                        {
                            "brain.olfactor.perception": "log",
                            "brain.olfactor.decay_coef": 0.1,
                            "brain.olfactor.gain_dict.Odor": gain,
                            "brain.interference.attenuation": 0.1,
                            "brain.interference.attenuation_max": 0.9,
                        }
                    ),
                    "color": cols[i],
                }
                i += 1

        return util.AttrDict(models)

    def get_models4(self, gain):
        """Build the model set comparing the calibrated variants.

        Args:
            **kwargs: Group settings.

        Returns:
            The larva groups.
        """
        cols = util.N_colors(4)
        i = 0
        models = {}
        for Tmod in ["NEU", "SIN"]:
            for Ifmod in ["PHI", "SQ"]:
                m = reg.conf.Model.getID(f"RE_{Tmod}_{Ifmod}_DEF_var2_nav")
                models[f"{Tmod}_{Ifmod}"] = {
                    "model": m.update_nestdict_copy(
                        {
                            "brain.olfactor.perception": "log",
                            "brain.olfactor.decay_coef": 0.1,
                            "brain.olfactor.gain_dict.Odor": gain,
                        }
                    ),
                    "color": cols[i],
                }
                i += 1

        return util.AttrDict(models)

    def chemo_exps(self, models):
        """Build the chemotaxis experiments for one model set.

        Args:
            lgs: The larva groups to place.

        Returns:
            The experiment entries, one per chemotactic assay.
        """
        exp1 = "Orbiting behavior"
        dst1 = Larva_Distro(N=self.N, mode="uniform")
        kws1 = {
            "env": reg.conf.Env.get("mid_odor_gaussian"),
            "lgs": util.AttrDict.merge_dicts(
                [
                    LarvaGroup(
                        group_id=mID,
                        distribution=dst1,
                        color=d["color"],
                        model=d["model"],
                    ).entry()
                    for mID, d in models.items()
                ]
            ),
            "id": f"{exp1}_exp",
            "dur": self.dur,
            "exp": exp1,
        }

        exp2 = "Up-gradient navigation"
        dst2 = Larva_Distro(
            N=self.N,
            mode="uniform",
            loc=(-0.04, 0.0),
            orientation_range=(-30.0, 30.0),
            scale=(0.005, 0.02),
        )
        kws2 = {
            "env": reg.conf.Env.get("odor_gradient"),
            "lgs": util.AttrDict.merge_dicts(
                [
                    LarvaGroup(
                        group_id=mID,
                        distribution=dst2,
                        color=d["color"],
                        model=d["model"],
                    ).entry()
                    for mID, d in models.items()
                ]
            ),
            "id": f"{exp2}_exp",
            "dur": self.dur,
            "exp": exp2,
        }

        return {exp1: [self.conf(**kws1)], exp2: [self.conf(**kws2)]}

    def analyze(self, exp, ds0):
        """Analyse one experiment of the essay.

        Args:
            exp: The experiment name.
            ds0: The datasets it produced.

        Returns:
            The figures and any derived results.
        """
        pass

    def global_anal(self):
        """Render the essay's cross-experiment figures.

        Returns:
            The rendered figures.
        """
        kwargs = {
            "datasets": self.datasets,
            "save_to": self.plot_dir,
            "show": self.show,
            "title": f"CHEMOTAXIS ESSAY (N={self.N})",
        }
        self.figs["chemotaxis summary"] = reg.graphs.run(
            ID="chemotaxis summary", mdiff_df=self.mdiff_df, **kwargs
        )


@funcs.stored_conf("Essay")
def Essay_dict():
    """Build the registry of stored essays.

    Returns:
        Every essay configuration, keyed by name.
    """
    d = {
        # 'roversVSsitters': rover_sitter_essay,
        # 'RvsS_essay': {}
    }
    # for E in [RvsS_Essay,Chemotaxis_Essay]:
    for E in [RvsS_Essay, DoublePatch_Essay, Chemotaxis_Essay]:
        e = E()
        d[e.type] = e.exp_dict
    return util.AttrDict(d)


def RvsSx4():
    """Build the four rover-versus-sitter model groups.

    Args:
        **kwargs: Group settings.

    Returns:
        The larva groups.
    """
    sufs = ["foragers", "navigators", "feeders", "locomotors"]
    i = 0
    for o in [True, False]:
        for f in [True, False]:
            E = DoublePatch_Essay(
                video=False,
                N=5,
                dur=5,
                olfactor=o,
                feeder=f,
                essay_id=f"RvsS_{sufs[i]}",
            )
            ds = E.run()
            figs, results = E.anal()
            i += 1
