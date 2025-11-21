# Figures and Tables from Paper

Source: **Larvaworld PLOS Comp.Biology Software_v05**

---

## Figures

### ✅ Fig 1: Architecture Overview ⭐
- **File**: `fig1_architecture.pdf` (vector format)
- **Caption**: "Larvaworld architecture. A schematic of the main components and functionalities of Larvaworld."
- **Documentation**: `figures/fig1_architecture.md`
- **Status**: ✅ **COMPLETED** (PDF version + comprehensive documentation)
- **Source**: `/images/architecture.pdf`

### ✅ Fig 2: Visualization Snapshots ⭐
- **Files**: 
  - `fig2a_replay_black.png`
  - `fig2b_replay_zoom.png`
  - `fig2c_sim_preference.png`
  - `fig2d_sim_game.png`
- **Caption**: "Real-time visualization of reconstructed real-animal experiments and agent simulations"
- **Documentation**: `figures/fig2_visualization_snapshots.md`
- **Status**: ✅ **COMPLETED** (4 snapshots + comprehensive documentation)
- **Source**: `/images/snapshots/`

### ✅ Fig 3: Simulation Modes ⭐
- **File**: `fig3_simulation_modes.png`
- **Caption**: "CLI simulation modes. The simulation modes available in Larvaworld"
- **Documentation**: `figures/fig3_simulation_modes.md`
- **Status**: ✅ **COMPLETED** (copied + documentation)
- **Source**: `/images/sim_modes.png`

### Fig 4: GA Configuration Panel
- **File**: `GApanel.png`
- **Caption**: "GA configuration panel"
- **Status**: 🟡 Reference
- **Location**: `/images/GApanel.png`

### Fig 5: Model Evaluation Configuration ⭐
- **File**: `EvalConf params.png`
- **Caption**: "Model Evaluation configuration parameters"
- **Status**: 🟡 From tutorial - Developer says no need to add separately
- **Location**: `/images/EvalConf params.png`

### Fig 6: Environment Configuration ⭐
- **File**: `EnvConf.png`
- **Caption**: "Virtual environment configuration parameters"
- **Status**: 🟡 From tutorial (`environment_configuration.ipynb`)
- **Location**: `/images/EnvConf.png`

### Fig 7: Larva Group Parameters ⭐
- **File**: `LarvaGroup.png`
- **Caption**: "Virtual larva group parameters"
- **Status**: 🟡 From tutorial
- **Location**: `/images/LarvaGroup.png`

### Fig 8: Web Application Screenshot ⭐
- **File**: `larvaworld_app_model_inspector.png`
- **Caption**: "Web-based Larvaworld application to inspect the modular composition of any preconfigured locomotory larva-model"
- **Status**: 🔴 Priority - Developer requested (combine with Table 5)
- **Location**: `/images/Apps/larvaworld_app_model_inspector.png`

### Fig 9: Data Import Tab
- **Files**: 
  - `import/import_tab.png`
  - `import/import_win_solo.png`
- **Caption**: "Data import tab" and "Data import window"
- **Status**: 🟢 Optional
- **Location**: `/images/import/`

### Fig 10-15: Results Figures
- **Files**: Multiple result/summary PDFs in `/images/Results/` and `/images/Results/SUMMARY/`
- **Status**: 🟢 Optional - Research results, not platform documentation

---

## Tables

### Table 1: Keyboard Controls ⭐
- **Title**: "Visualization default keyboard/mouse controls"
- **Content**: Screen, Drawing, Color, Interaction, Simulation/Storage controls
- **Status**: 🔴 Priority - Developer requested
- **Note**: Needs explanation of what each shortcut does (ideally video)

### Table 2: Simulation Modes
- **Title**: "Simulation modes"
- **Content**: Description of different simulation modes
- **Status**: 🟡 Reference

### Table 3: Simulation Configuration
- **Title**: "Simulation configuration"
- **Content**: Configuration parameters for simulations
- **Status**: 🟡 Reference

### Table 4: Experiment Types ⭐
- **Title**: "Summary of preconfigured behavioral experiments"
- **Content**: List of available experiments
- **Status**: 🔴 Priority - Developer says combine with experiment types mindmap
- **Note**: Already covered in diagrams 05_a and 05_b

### Table 5: Batch-run Configurations
- **Title**: "Batch-run configurations"
- **Content**: Parameters for batch runs
- **Status**: 🟡 Reference

### Table 6: Virtual Environment Attributes
- **Title**: "Virtual environment attributes"
- **Content**: Environment configuration options
- **Status**: 🟡 Reference

### ✅ Table 2: Preconfigured Behavioral Experiments ⭐
- **Title**: "Summary of preconfigured behavioral experiments"
- **Content**: Experiment types (exploration, chemotaxis, odor preference, foraging, growth, imitation, games)
- **Markdown File**: `tables/table2_preconfigured_experiments.md`
- **Status**: ✅ **COMPLETED** (16 experiments across 7 categories + citations)
- **Related**: Experiment types mindmap diagrams (05_a and 05_b)

### ✅ Table 3: Nutritious Arena Substrates ⭐
- **Title**: "Compound composition of established nutritious arena substrates"
- **Content**: Food substrate recipes
- **Markdown File**: `tables/table3_nutritious_substrates.md`
- **Status**: ✅ **COMPLETED** (4 substrates with compound compositions + full citations)
- **Note**: Practical information for experiments

### ✅ Table 4: Larva Group Placement Parameters ⭐
- **Title**: "Larva group initial spatial placement parameters"
- **Content**: Initial position configuration
- **Markdown File**: `tables/table4_larva_placement.md`
- **Status**: ✅ **COMPLETED** (6 parameters with examples)

### ✅ Table 5: Web-based Applications ⭐
- **Title**: "Web-based applications"
- **Content**: List of available web apps
- **Markdown File**: `tables/table5_web_applications.md`
- **Status**: ✅ **COMPLETED** (4 apps + dev note about status)
- **Note**: Developer mentions some don't work properly (future fix) - included in doc

### ✅ Table 6: Data Processing Methods ⭐
- **Title**: "Data processing methods"
- **Content**: Available processing/analysis methods
- **Markdown File**: `tables/table6_data_processing.md`
- **Status**: ✅ **COMPLETED** (3 pipelines: preprocessing, processing, annotation)

### ✅ Table 7: Lab-specific Data Formats ⭐
- **Title**: "Lab-specific experimental data-formats"
- **Content**: Supported lab data formats for import
- **Markdown File**: `tables/table7_lab_formats.md`
- **Status**: ✅ **COMPLETED** (4 labs: Schleyer, Jovanic, Berni, Arguello)

### Table 8: Larva Group Attributes
- **Title**: "Larva group attributes"
- **Content**: Group configuration parameters
- **Status**: 🟡 Reference

### Table 13: Dataset Import Parameters
- **Title**: "Dataset import parameters"
- **Content**: Parameters for importing datasets
- **Status**: 🟡 Reference

### Tables 14-17: Evaluation Results
- **Title**: Various "Locomotory model evaluation" and "Average vs group-variability model" tables
- **Content**: KS_D values and statistical results
- **Status**: 🟢 Optional - Research results

---

## Priority Summary

### 🔴 High Priority (Developer Requested):
1. ✅ **Fig 1**: Architecture (central placement - PDF version)
2. ✅ **Fig 2**: Snapshots (nice pictures)
3. ✅ **Fig 3**: Simulation modes
4. ✅ **Fig 8**: Web app screenshot
5. ✅ **Table 1**: Keyboard controls (+ explanations) - **CORRECTED FORMAT**
6. ✅ **Table 2**: Preconfigured behavioral experiments (combine with mindmap - **DONE** in diagrams)
7. ✅ **Table 3**: Nutritious substrates (with full citations)
8. ✅ **Table 4**: Larva placement parameters
9. ✅ **Table 5**: Web applications list
10. ✅ **Table 6**: Data processing methods (CORRECTED FORMAT)
11. ✅ **Table 7**: Lab data formats (with full citations)

### 🟡 Medium Priority (From Tutorials):
- **Figs 5,6,7**: Configuration panels (developer says from notebooks, no need separate)

### 🟢 Low Priority:
- Results figures/tables (research results, not platform docs)
- Internal configuration tables (covered in tutorials)

---

## Next Steps

1. ✅ Copy high-priority figures
2. ✅ Extract and format high-priority tables
3. ✅ Create markdown documentation for each
4. ✅ Integrate into ReadTheDocs structure

---

**Note**: Figures 5, 6, 7 are from tutorial notebooks (`environment_configuration.ipynb`) and developer says they don't need to be added separately as `.md` files.

