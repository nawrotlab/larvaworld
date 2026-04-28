from __future__ import annotations

from larvaworld.portal.registry_types import (
    LaneSpec,
    LandingItem,
    LearnMore,
    QuickStartModeSpec,
)

DOCS_ROOT = "https://larvaworld.readthedocs.io/en/latest/"
GITHUB_ROOT = "https://github.com/nawrotlab/larvaworld"
GITHUB_ISSUES = f"{GITHUB_ROOT}/issues"

DOCS_WEB_APPS = f"{DOCS_ROOT}visualization/web_applications.html"
DOCS_EXPERIMENT_VIEWER = f"{DOCS_WEB_APPS}#experiment-viewer"
DOCS_TRACK_VIEWER = f"{DOCS_WEB_APPS}#track-viewer"
DOCS_MODEL_INSPECTOR = f"{DOCS_WEB_APPS}#model-inspector"
DOCS_MODULE_INSPECTOR = f"{DOCS_WEB_APPS}#module-inspector"
DOCS_LATERAL_OSCILLATOR = f"{DOCS_WEB_APPS}#lateral-oscillator-inspector"

DOCS_SINGLE_EXPERIMENTS = f"{DOCS_ROOT}working_with_larvaworld/single_experiments.html"
DOCS_EXPERIMENT_TYPES = f"{DOCS_ROOT}concepts/experiment_types.html"
DOCS_BATCH_RUNS = f"{DOCS_ROOT}working_with_larvaworld/batch_runs_advanced.html"

DOCS_REFERENCE_DATASETS = f"{DOCS_ROOT}data_pipeline/reference_datasets.html"
DOCS_DATA_PROCESSING = f"{DOCS_ROOT}data_pipeline/data_processing.html"
DOCS_PLOTTING_API = f"{DOCS_ROOT}visualization/plotting_api.html"

DOCS_ARENAS_SUBSTRATES = f"{DOCS_ROOT}agents_environments/arenas_and_substrates.html"
DOCS_AGENT_ARCHITECTURE = (
    f"{DOCS_ROOT}agents_environments/larva_agent_architecture.html"
)

DOCS_MODEL_EVALUATION = f"{DOCS_ROOT}working_with_larvaworld/model_evaluation.html"
DOCS_GA_OPTIMIZATION = (
    f"{DOCS_ROOT}working_with_larvaworld/ga_optimization_advanced.html"
)
DOCS_COMPARE_DATASETS = f"{DOCS_MODEL_EVALUATION}#statistical-comparison-plots"

NOTEBOOK_TUTORIAL_BY_ITEM_ID: dict[str, str] = {
    "wf.run_experiment": "single_simulation.ipynb",
    "wf.open_dataset": "import_datasets.ipynb",
    "track_viewer": "replay.ipynb",
    "wf.dataset_manager": "import_datasets.ipynb",
    "larva_models": "library_interface.ipynb",
    "locomotory_modules": "custom_module.ipynb",
    "wf.environment_builder": "environment_configuration.ipynb",
    "wf.model_evaluation": "model_evaluation.ipynb",
    "wf.ga_optimization": "genetic_algorithm_optimization.ipynb",
    "wf.compare_datasets": "model_evaluation.ipynb",
}


# ---- Deterministic ordering: these lists define display order. ----

PINNED_QUICK_START: list[str] = [
    "wf.run_experiment",
    "wf.export_center",
    "wf.deb_explorer",
]

QUICK_START_MODES: list[QuickStartModeSpec] = [
    QuickStartModeSpec(
        mode_id="user",
        title="User mode",
        color="#e7c575",
        item_ids=[
            "wf.run_experiment",
            "wf.export_center",
            "wf.deb_explorer",
        ],
    ),
    QuickStartModeSpec(
        mode_id="modeler",
        title="Modeler mode",
        color="#c1b0c2",
        item_ids=[
            "wf.environment_builder",
            "larva_models",
            "wf.ga_optimization",
        ],
    ),
    QuickStartModeSpec(
        mode_id="experimentalist",
        title="Experimentalist mode",
        color="#b0b4c2",
        item_ids=[
            "wf.open_dataset",
            "track_viewer",
            "wf.model_evaluation",
        ],
    ),
]

QUICK_START_DEFAULT_MODE = "user"

LANES: list[LaneSpec] = [
    LaneSpec(
        title="Simulation & Optimization",
        lane="simulate",
        item_ids=[
            "wf.run_experiment",
            "wf.model_evaluation",
            "wf.ga_optimization",
            "wf.batch_runs",
            "wf.essay",
        ],
    ),
    LaneSpec(
        title="Data & Visualization",
        lane="data",
        item_ids=[
            "wf.open_dataset",
            "track_viewer",
            "wf.dataset_manager",
            "wf.export_center",
        ],
    ),
    LaneSpec(
        title="Models & Environments",
        lane="models",
        item_ids=[
            "larva_models",
            "locomotory_modules",
            "wf.environment_builder",
            "wf.deb_explorer",
        ],
    ),
]


ITEMS: dict[str, LandingItem] = {
    # ---- Real Panel apps (id == panel_app_id) ----
    "track_viewer": LandingItem(
        id="track_viewer",
        kind="panel_app",
        status="ready",
        lane="data",
        level="core",
        title="Dataset Replay",
        subtitle=(
            "Replay larval trajectories frame-by-frame.\n"
            "Inspect motion quality and path structure.\n"
            "Quickly compare individuals in one view."
        ),
        cta="Open",
        panel_app_id="track_viewer",
        learn_more=LearnMore(docs_url=DOCS_TRACK_VIEWER),
    ),
    "experiment_viewer": LandingItem(
        id="experiment_viewer",
        kind="panel_app",
        status="hidden",
        lane="simulate",
        level="core",
        title="Experiment Viewer",
        subtitle=(
            "Step through a completed experiment run.\n"
            "Inspect state changes across time.\n"
            "Review outputs before deeper analysis."
        ),
        cta="Preview",
        panel_app_id="experiment_viewer",
        learn_more=LearnMore(docs_url=DOCS_EXPERIMENT_VIEWER),
    ),
    "larva_models": LandingItem(
        id="larva_models",
        kind="panel_app",
        status="ready",
        lane="models",
        level="core",
        title="Model Inspector",
        subtitle=(
            "Browse available larva model presets.\n"
            "Inspect key parameters and defaults.\n"
            "Compare configurations before simulation."
        ),
        cta="Inspect",
        panel_app_id="larva_models",
        learn_more=LearnMore(docs_url=DOCS_MODEL_INSPECTOR),
    ),
    "locomotory_modules": LandingItem(
        id="locomotory_modules",
        kind="panel_app",
        status="ready",
        lane="models",
        level="core",
        title="Module Inspector",
        subtitle=(
            "Inspect locomotory and sensorimotor modules.\n"
            "Review module parameters and behavior roles.\n"
            "Understand how modules combine in control."
        ),
        cta="Inspect",
        panel_app_id="locomotory_modules",
        learn_more=LearnMore(docs_url=DOCS_MODULE_INSPECTOR),
    ),
    "lateral_oscillator": LandingItem(
        id="lateral_oscillator",
        kind="panel_app",
        status="hidden",
        lane="demos",
        level="demo",
        title="Lateral Oscillator",
        subtitle=(
            "Explore the lateral oscillator controller.\n"
            "Visualize oscillation and coupling behavior.\n"
            "Inspect parameters for rhythmic motion."
        ),
        cta="Open",
        panel_app_id="lateral_oscillator",
        learn_more=LearnMore(docs_url=DOCS_LATERAL_OSCILLATOR),
    ),
    # ---- External links ----
    "link.docs": LandingItem(
        id="link.docs",
        kind="external_link",
        status="ready",
        lane="demos",
        level="demo",
        title="Docs",
        subtitle=(
            "Open the Larvaworld documentation portal.\n"
            "Browse guides, tutorials, and references.\n"
            "Find details for each workflow."
        ),
        cta="Open",
        url=DOCS_ROOT,
        learn_more=LearnMore(docs_url=DOCS_ROOT),
    ),
    "link.github": LandingItem(
        id="link.github",
        kind="external_link",
        status="ready",
        lane="demos",
        level="demo",
        title="GitHub",
        subtitle=(
            "Open the Larvaworld GitHub repository.\n"
            "Track issues, roadmap, and development.\n"
            "Follow implementation progress."
        ),
        cta="Open",
        url=GITHUB_ROOT,
        learn_more=LearnMore(docs_url=f"{DOCS_ROOT}contributing.html"),
    ),
    # ---- Planned workflows / placeholders ----
    "wf.run_experiment": LandingItem(
        id="wf.run_experiment",
        kind="panel_app",
        status="ready",
        lane="simulate",
        level="core",
        title="Single Experiment",
        subtitle=(
            "Select an experiment template and key run settings.\n"
            "Optionally apply a workspace environment preset.\n"
            "Prepare one simulation in the browser workflow."
        ),
        cta="Run",
        panel_app_id="wf.run_experiment",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_SINGLE_EXPERIMENTS,
        ),
        badges=["Developer"],
        preview_md=(
            "### Single Experiment\n"
            "- Pick an experiment template from the Larvaworld registry\n"
            "- Apply a workspace environment preset or keep the template default\n"
            "- Adjust run duration and larvae-per-group override\n"
            "- Prepare an interactive arena preview for one single-run experiment\n"
        ),
    ),
    "wf.open_dataset": LandingItem(
        id="wf.open_dataset",
        kind="panel_app",
        status="ready",
        lane="data",
        level="core",
        title="Import Experimental Datasets",
        subtitle=(
            "Import one raw experimental dataset into the active workspace.\n"
            "Discover candidate folders under a chosen raw-data root.\n"
            "Save the imported dataset in workspace-owned storage."
        ),
        cta="Import",
        panel_app_id="wf.open_dataset",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_REFERENCE_DATASETS,
        ),
        preview_md=(
            "### Import Experimental Datasets\n"
            "- Choose a lab format and raw-data root\n"
            "- Discover one candidate dataset at a time\n"
            "- Import it into the active workspace with clear status feedback\n"
        ),
    ),
    "wf.model_evaluation": LandingItem(
        id="wf.model_evaluation",
        kind="placeholder",
        status="planned",
        lane="simulate",
        level="core",
        title="Model Evaluation",
        subtitle=(
            "Compare model outputs against references.\n"
            "Compute metrics and summary scores.\n"
            "Generate plots for model validation."
        ),
        cta="Evaluate",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_MODEL_EVALUATION,
        ),
        badges=["Developer"],
        preview_md=(
            "### Model Evaluation (Planned)\n"
            "- Select simulation output + reference dataset\n"
            "- Compute metrics and summary scores\n"
            "- Generate a report with comparison plots\n"
        ),
    ),
    "wf.experiment_catalog": LandingItem(
        id="wf.experiment_catalog",
        kind="placeholder",
        status="hidden",
        lane="simulate",
        level="core",
        title="Experiment Catalog",
        subtitle=(
            "Browse the curated experiment library.\n"
            "Filter presets by purpose and complexity.\n"
            "Open one preset to start quickly."
        ),
        cta="Browse",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_EXPERIMENT_TYPES,
        ),
        preview_md=(
            "### Experiment Catalog (Planned)\n"
            "- Search and filter experiment presets\n"
            "- See a short description and default parameters\n"
            "- Open a preset in Run Experiment\n"
        ),
    ),
    "wf.batch_runs": LandingItem(
        id="wf.batch_runs",
        kind="placeholder",
        status="planned",
        lane="simulate",
        level="core",
        title="Batch Runs",
        subtitle=(
            "Launch many configurations in parallel.\n"
            "Track status, logs, and produced outputs.\n"
            "Rerun failed jobs with minimal effort."
        ),
        cta="Open",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_BATCH_RUNS,
        ),
        badges=["Developer"],
        preview_md=(
            "### Batch Runs (Planned)\n"
            "- Define a parameter sweep\n"
            "- Run many simulations and track outputs\n"
            "- Export results and re-run failed jobs\n"
        ),
    ),
    "wf.dataset_manager": LandingItem(
        id="wf.dataset_manager",
        kind="panel_app",
        status="ready",
        lane="data",
        level="core",
        title="Dataset Manager",
        panel_app_id="wf.dataset_manager",
        subtitle=(
            "Browse imported datasets stored in the active workspace.\n"
            "Inspect lightweight record details and stored artifacts.\n"
            "Refresh, copy paths, and remove imported datasets safely."
        ),
        cta="Manage",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_DATA_PROCESSING,
        ),
        preview_md=(
            "### Dataset Manager\n"
            "- Browse imported datasets in the current workspace\n"
            "- Inspect IDs, labs, paths, and stored artifacts\n"
            "- Refresh the catalog and remove imported datasets safely\n"
        ),
    ),
    "wf.export_center": LandingItem(
        id="wf.export_center",
        kind="placeholder",
        status="planned",
        lane="data",
        level="core",
        title="Analysis",
        subtitle=(
            "Inspect selected datasets through analysis views.\n"
            "Build plots and summary comparisons interactively.\n"
            "Prepare deeper post-processing workflows."
        ),
        cta="Analyze",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_PLOTTING_API,
        ),
        preview_md=(
            "### Analysis (Planned)\n"
            "- Open dataset-centric analysis tools\n"
            "- Build comparison plots and summaries\n"
        ),
    ),
    "wf.environment_builder": LandingItem(
        id="wf.environment_builder",
        kind="panel_app",
        status="ready",
        lane="models",
        level="core",
        title="Environment Builder",
        subtitle=(
            "Design arenas, borders, and obstacles.\n"
            "Compose sensory and substrate landscapes.\n"
            "Save reusable environment presets."
        ),
        cta="Create",
        panel_app_id="wf.environment_builder",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_ARENAS_SUBSTRATES,
        ),
        badges=["Developer"],
        preview_md=(
            "### Environment Builder\n"
            "- Configure arena geometry, borders, and obstacles\n"
            "- Define source, grid, and scape parameters\n"
            "- Save and reuse environment presets\n"
        ),
    ),
    "wf.deb_explorer": LandingItem(
        id="wf.deb_explorer",
        kind="placeholder",
        status="planned",
        lane="models",
        level="core",
        title="DEB Simulator",
        subtitle=(
            "Inspect DEB energetics assumptions.\n"
            "Explore metabolic parameter effects.\n"
            "Relate energy state to behavior."
        ),
        cta="Open",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=f"{DOCS_AGENT_ARCHITECTURE}#4-energy-system",
        ),
        badges=["Developer"],
        preview_md=(
            "### DEB Simulator (Planned)\n"
            "- Inspect energetics assumptions and constraints\n"
            "- Explore DEB parameter presets\n"
        ),
    ),
    "wf.ga_optimization": LandingItem(
        id="wf.ga_optimization",
        kind="placeholder",
        status="planned",
        lane="simulate",
        level="advanced",
        title="GA Optimization",
        subtitle=(
            "Tune model parameters with GA search.\n"
            "Optimize against reference objectives.\n"
            "Compare candidate solutions and fitness."
        ),
        cta="Optimize",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_GA_OPTIMIZATION,
        ),
        badges=["Developer"],
        preview_md=(
            "### GA Optimization (Planned)\n"
            "- Define a scoring objective\n"
            "- Run GA to tune parameters\n"
            "- Compare candidate solutions\n"
        ),
    ),
    "wf.compare_datasets": LandingItem(
        id="wf.compare_datasets",
        kind="placeholder",
        status="hidden",
        lane="simulate",
        level="advanced",
        title="Compare Datasets",
        subtitle=(
            "Compare multiple runs side by side.\n"
            "Inspect metric differences across conditions.\n"
            "Summarize effects with common plots."
        ),
        cta="Compare",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(
            issue_url=GITHUB_ISSUES,
            docs_url=DOCS_COMPARE_DATASETS,
        ),
        preview_md=(
            "### Compare Datasets (Planned)\n"
            "- Select multiple runs or conditions\n"
            "- Compare metrics and summary plots\n"
        ),
    ),
    "wf.essay": LandingItem(
        id="wf.essay",
        kind="placeholder",
        status="planned",
        lane="simulate",
        level="core",
        title="Essay",
        subtitle=(
            "Compose experiment essay workflows interactively.\n"
            "Combine narratives, configs, and generated outputs.\n"
            "Prototype structured reporting from one place."
        ),
        cta="Open",
        prereq_hint="Not available yet in the web UI.",
        learn_more=LearnMore(issue_url=GITHUB_ISSUES),
        badges=["Beta"],
        preview_md=(
            "### Essay (Planned)\n"
            "- Compose narrative-driven experiment workflows\n"
            "- Attach runs, plots, and summary outputs\n"
        ),
    ),
}
