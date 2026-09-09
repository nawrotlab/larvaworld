# DEB Model Examples and Educational Materials

This folder contains examples and educational materials for working with the Dynamic Energy Budget (DEB) model in Larvaworld.

## Overview

The DEB model (`src/larvaworld/lib/model/deb/deb_equations.py`) is a ground-truth transcription of the holometabolous insect model from the AmP (Add-my-Pet) database. This folder provides tools for:

1. **Validating predictions** against AmP reference data
2. **Understanding the model** through worked examples
3. **Testing integration engines** (closed-form ODE vs. stepped Euler)
4. **Educational demonstrations** of DEB theory applied to _Drosophila melanogaster_

## Files

### `test_amp_predictions.py`

**Purpose**: Complete pipeline for validating AmP zero-variate predictions against Larvaworld simulations.

**Features**:

- Load AmP JSON exports (parameters + observed/predicted data)
- Run life cycle simulations with two engines:
  - Closed-form ODE integration (`scipy.integrate.solve_ivp`)
  - Fixed-step Euler method (numerical stepping)
- Extract and compare predictions
- Generate formatted comparison tables
- Visualize results (observed vs. AmP vs. Larvaworld)
- Compute and report error statistics

**Usage**:

```bash
# Run as standalone script
python examples/deb/test_amp_predictions.py

# Import for interactive use (Jupyter, IPython)
from examples.deb.test_amp_predictions import AmPPredictionTester

# Create a tester instance
tester = AmPPredictionTester("path/to/Drosophila_melanogaster.json", f=1.0)

# Run full pipeline
tester.run_full_pipeline(plot=True)

# Or use individual methods
tester.load_json()
tester.run_simulations()
tester.extract_predictions()
comparison_df = tester.build_comparison()
tester.print_comparison()
tester.print_error_summary()
tester.plot_predictions()
```

**Key Classes and Methods**:

- `AmPPredictionTester`: Main class orchestrating the pipeline
  - `load_json()`: Parse JSON export and parameters
  - `run_simulations()`: Simulate life cycle with both engines
  - `extract_predictions()`: Extract zero-variate predictions
  - `build_comparison()`: Assemble side-by-side comparison table
  - `print_comparison()`: Display formatted table
  - `print_error_summary()`: Show accuracy metrics
  - `plot_predictions()`: Visualize all predictions
  - `run_full_pipeline()`: Execute all steps end-to-end

**Educational Value**:

This script demonstrates:

- How to load and parse AmP parameter exports
- The structure of the DEB model's life history
- Differences between numerical integration methods
- How to measure model accuracy against reference data
- Best practices for scientific software validation

### AmP JSON Structure

The script expects JSON files in the format exported by the AmP (Add-my-Pet) database:

```json
{
  "metadata": {
    "species": "Drosophila_melanogaster",
    "author": "...",
    "date_subm": [2026, 1, 1],
    ...
  },
  "parameters": [
    {"symbol": "z", "value": 0.049897, "free": 1, "unit": "-"},
    {"symbol": "v", "value": 0.04503, "free": 1, "unit": "cm/d"},
    ...
  ],
  "results": [
    {"symbol": "ab", "observed": 0.7, "predicted": 0.716, "RE": 0.02284, "unit": "d", "description": "age at birth"},
    {"symbol": "tj", "observed": 7.8, "predicted": 6.146, "RE": 0.212, "unit": "d", "description": "time since birth at pupation"},
    ...
  ]
}
```

The script is currently configured to find `Drosophila_melanogaster.json` in:

```
src/larvaworld/lib/model/deb/AmP_models/Drosophila_melanogaster/
```

### Zero-Variate Predictions Tested

**Currently Implemented** (6 symbols):

- `ab`: age at birth (d)
- `tj`: time since birth at pupation (d)
- `tje`: time since pupation at emergence (d)
- `Lb`: physical length at birth (cm) — _mapped from structural length via del_M_
- `Lj`: physical length at pupation (cm)
- `Wd_e_f`: dry weight at emergence, female (mg)

**Deferred** (ground truth located, requires additional porting):

- `t1`, `t2`: instar 1/2 durations — requires `get_tj_habp` instar sub-model
- `L1`, `L2`: instar lengths — requires `get_tj_habp` integration
- `Ri`: fecundity (eggs/female/day) — requires ultimate-state reproduction formula
- `am_BD_LD`: mean adult lifespan (d) — requires aging ODE (`get_tm_mod_habp`)

See `plans/deb_amp_predictions_and_univariate_plots.md` for full scope and porting roadmap.

## DEB Model Reference

**Key Documents**:

- `src/larvaworld/lib/model/deb/deb_equations.py` — Ground-truth implementation
- `src/larvaworld/lib/model/deb/deb.py` — Larvaworld integration and utilities
- Kooijman (2010) "Dynamic Energy Budget theory for metabolic organisation" — Theoretical foundation

**Parameter Meanings** (Google-style docstrings):

- `z` — zoom factor (scaling parameter)
- `v` — energy conductance (cm/d)
- `kap` — allocation fraction to soma (-)
- `p_M` — volume-specific somatic maintenance (J/d·cm³)
- `E_G` — specific cost for structure (J/cm³)
- `E_Hb`, `E_Hp`, `E_He` — maturity thresholds (J) at birth, pupation, emergence
- `del_M` — shape coefficient for larval head capsule (-)
- `del_Mw` — shape coefficient for imago wing length (-)
- `kap_V` — conversion efficiency E→V→E at pupation (-)
- `h_a` — Weibull aging acceleration (1/d²) — _currently unused, carried for future aging model_
- `s_G` — Gompertz stress coefficient (-) — _currently unused, carried for future aging model_

See `src/larvaworld/lib/model/deb/deb_equations.py` class `DEBPars` for complete parameter documentation.

## Course Integration

This example is designed for conversion to a Jupyter notebook for teaching:

1. **Load and explore**: Import `AmPPredictionTester`, load an AmP JSON export
2. **Run simulations**: Compare the two integration engines
3. **Validate model**: Check predictions against reference data
4. **Analyze errors**: Examine where and why the model diverges
5. **Visualize**: Generate comparison plots
6. **Extend**: Modify `f` or `T` to explore parameter sensitivity

The modular design allows instructors to:

- Highlight individual methods for teaching concepts
- Explore life-history events step by step
- Compare numerical methods (ODE vs. stepping)
- Discuss error sources and model limitations
- Extend to univariate data (mass over time, fecundity curves)

## Dependencies

- `numpy`, `pandas`, `matplotlib` — standard scientific Python
- `scipy.integrate.solve_ivp` — closed-form ODE solver
- `larvaworld` — installed from this repository

## Future Work

Remaining univariate data reproduction (see `plans/deb_amp_predictions_and_univariate_plots.md`):

- Port `dget_ELH` ODE for larval growth trajectory
- Port `dget_ELHE_imago` ODE for imago reproduction curve
- Add weight conversion formulas and plotting
- Implement alternate-diet (`f_F424`, `f_JAZZ`, `f_DR`, `f_HS`) simulations
- Extend to instar moults and aging/survival curves

## References

- **AmP (Add-my-Pet)**: https://www.bio.vu.nl/thb/deb/deblab/
- **DEBtool_M**: The original MATLAB implementation
- **Kooijman, S.A.L.M.** (2010). _Dynamic Energy Budget theory for metabolic organisation_. Cambridge University Press.

---

**Author**: Claude Haiku 4.5
**License**: Same as Larvaworld (see LICENSE file in repo root)
