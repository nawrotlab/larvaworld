# Univariate Data Reproduction Roadmap

**Status**: Partially implemented (framework ready, ODE solvers deferred)
**Date**: 2026-08-25
**Files**: `univariate_data_plots.py` (framework), MATLAB sources in `G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\`

## Overview

The goal is to reproduce four univariate datasets showing observed data, DEBtool-predicted curves, and Larvaworld simulations (both closed-form ODE and stepped Euler engines):

1. **tWw_f** — Time vs wet weight (larval development)
2. **tR_C** — Time vs reproductive output (standard diet, f=1.0)
3. **tR_DR** — Time vs reproductive output (dietary restriction, f_DR=0.9)
4. **tR_HS** — Time vs reproductive output (high-sugar diet, f_HS=0.9)

## Completed ✓

- Framework structure in `univariate_data_plots.py` with `UnivariatePlotter` class
- MATLAB sources located and available locally
- Observed data hard-coded (from `mydata_Drosophila_melanogaster.m`)
- DEB parameter loading via JSON working correctly
- Larvaworld life-cycle simulation infrastructure ready (both engines)
- Plotting templates prepared (matplotlib)

## Implementation Path

### Phase 1: Use Existing Larvaworld Infrastructure (No New ODE Solvers)

**Current status**: Ready to implement now

The simplest approach is to leverage what `run_life_cycle()` already does and avoid implementing complex new ODE solvers:

#### Option 1A: Direct Trajectory Conversion (Recommended for Phase 1)

Extract time-series data from existing `Trajectory` objects returned by `run_life_cycle()`:

```python
def plot_tWw_f(self):
    # Run existing Larvaworld life-cycle
    lh = de.run_life_cycle(pars, engine="closed", f=1.0)

    # Extract trajectory time-series
    for point in lh.trajectory:
        age = point.age
        L = point.L
        E = point.E
        Ww = (L^3 + E*w_E/mu_E/d_E)*1000  # wet weight in mg
        plot(age, Ww)
```

**Pros**: Uses existing, tested code; no new solvers needed
**Cons**: Assumes Larvaworld's Table S1/S2 dynamics match MATLAB's `dget_ELH`; may differ

#### Option 1B: Parameter-Only Predictions (Without Full ODE)

Compute zero-variate predictions from DEBPars alone (no trajectory):

```python
# From DEBPars, compute at f=1.0:
a_b = time to birth
t_1, t_2 = instar durations
L_1, L_2 = instar lengths
R_i = fecundity rate
a_m = mean lifespan (requires solving get_tm_mod_habp)
```

**Pros**: Ground-truthed to MATLAB; no integration required for many
**Cons**: Requires porting `get_tj_habp`, `get_tm_mod_habp` (complex ODE solvers)

### Phase 2: Port MATLAB ODE Solvers (Deferred to Later Session)

**Scope**: Substantial (~500-800 lines of new code)

Three interconnected functions need porting from DEBtool_M and local MATLAB code:

#### Step A: Port `get_tj.m` → Python (`_get_tj.py`)

**Source**: `c:\Users\Panos\AmP Drosophila matlab workspace\DEBtool_M\animal\get_tj.m`
**Purpose**: Solve for scaled ages/lengths at birth, pupation, emergence

- Uses `fzero` (root-finder equivalent: `scipy.optimize.brentq`)
- Returns: `(tau_j, tau_p, tau_b, l_j, l_p, l_b, l_i, rho_j, rho_B, info)`
- Drosophila case: `v_Hj = v_Hp + 1e-8` (instar collapse → `l_j ≈ l_p`)

**Estimated effort**: 100-150 lines

#### Step B: Port `get_tj_habp.m` → Python (`_get_tj_habp.py`)

**Source**: Local `G:\Το Drive μου\DEB projects\...\get_tj_habp.m` (lines 1-75)
**Purpose**: Wrap `get_tj` + integrate pupal ODE (`dget_tj_hex`)

- Calls `get_tj` from Step A
- Integrates 3-state ODE: `(l, u_E, v_H)` from pupation → emergence
- Uses `scipy.integrate.solve_ivp` with terminal event at `v_H = v_He`
- Returns: `(tau_p, tau_e, tau_b, l_p, l_e, l_b, rho_j, u_Ee, info)`

**Estimated effort**: 80-120 lines (mostly ODE setup)

#### Step C: Port `get_tm_mod_habp.m` → Python (`_get_tm_mod_habp.py`)

**Source**: Local `get_tm_mod_habp.m`, `'hax'` branch only (lines 141-148)
**Purpose**: Compute scaled mean lifespan via aging/survival ODE

- Calls `get_tj_habp` from Step B
- Integrates 4-state ODE: `(q, h_A, S, t)` representing aging, hazard, survival, cumulative time
- Terminal event: `S <= 1e-6` (dead for sure)
- Returns: `tau_m` (scaled mean lifespan)
- ODE: `dget_qhSt_hex_ji` (lines 454-477)

**Estimated effort**: 120-180 lines (complex aging model)

**Critical**: Use **only the `'hax'` branch** (lines 141-148). Skip `'std'`, `'abj'`, `'hex'`, etc.

#### Step D: Add Zero-Variate Accessors to `deb_equations.py`

Once Steps A-C are done, implement:

```python
# Add to DEBPars or LifeHistory
def predict_instars(f: float = 1.0) -> dict:
    """Return t1, t2, L1, L2 as dict."""
    # Uses get_tj_habp from Step B

def predict_Ri(f: float = 1.0) -> float:
    """Ultimate fecundity rate."""
    # Formula from predict_Drosophila_melanogaster.m:65-70

def predict_am(f: float = 1.0) -> float:
    """Mean lifespan as imago."""
    # Uses get_tm_mod_habp from Step C
```

Update `_PREDICTION_ACCESSORS` mapping to use these.

**Estimated effort**: 40-60 lines

### Phase 3: Generate Full Univariate Plots

Once ODE solvers are ready:

#### For tWw_f (Larval wet weight):

1. Extract trajectory from `run_life_cycle()` (Phase 1)
2. Compute wet weight: `Ww = (L^3 + E*w_E/mu_E/d_E)*1000` mg
3. Compare against observed data + DEBtool curve (from MATLAB `predict_*.m`)

#### For tR_C, tR_DR, tR_HS (Fecundity):

1. Run adult stage (`Stage.IMAGO`) with different `f` values
2. Integrate reproduction buffer `E_R` to get cumulative eggs
3. Convert: `EN = kap_R * E_R / E_0`
4. Plot vs observed

## Current Code Structure

```
src/larvaworld/lib/model/deb/
  deb_equations.py     ← Main DEB reference implementation (ground truth)
                          - run_life_cycle() ← Use this, already works
  _get_tj.py           ← [TODO] Port get_tj.m
  _get_tj_habp.py      ← [TODO] Port get_tj_habp.m
  _get_tm_mod_habp.py  ← [TODO] Port get_tm_mod_habp.m

examples/deb/
  univariate_data_plots.py   ← Framework (ready for Phase 1)
  UNIVARIATE_ROADMAP.md      ← This file
  comprehensive_deb_pipeline.py  ← Analysis framework (existing)
  test_behavioral_rvss.py        ← Behavioral tests (existing)
```

## Strategy Recommendation

### For This Session:

- Implement Phase 1 (`univariate_data_plots.py` using existing trajectories)
- Generate and verify tWw_f plot (closest to being ready)
- Leave ODE solver porting to next session (it's orthogonal to behavioral work)

### For Next Session:

- Port `get_tj.m` → `_get_tj.py` (start with this, build up)
- Port `get_tj_habp.m` → `_get_tj_habp.py`
- Port `get_tm_mod_habp.m` → `_get_tm_mod_habp.py` (most complex)
- Add `predict_*` methods to DEBPars
- Complete tR_C/tR_DR/tR_HS plots
- Generate educational Jupyter notebook

## Testing & Verification

Once ODE solvers are ported:

```bash
# Check against MATLAB reference values
python -c "
from larvaworld.lib.model.deb._get_tj_habp import get_tj_habp
from larvaworld.lib.model.deb.generalized_animal import deb_generalized_animal

pars = deb_generalized_animal()
tau_p, tau_e, tau_b, l_p, l_e, l_b = get_tj_habp([...], f=1.0)[:6]
print(f't_e = {tau_e / k_M} d (expect ~18.6 d from predict_*.m)')
"

# Verify plots visually
python examples/deb/univariate_data_plots.py
```

Cross-check numeric outputs against:

- `Drosophila_melanogaster_res.html` (printed predictions)
- `mass_vs_time.png` (visual reference)
- `fecundity_vs_time.png` (visual reference)

## Files to Reference

**MATLAB sources** (read-only, ground truth):

- `G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\predict_Drosophila_melanogaster.m` — Main predictions
- `G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\get_tj_habp.m` — Pupal ODE
- `G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\get_tm_mod_habp.m` — Lifespan ODE (use lines 141-148 only)
- `c:\Users\Panos\AmP Drosophila matlab workspace\DEBtool_M\animal\get_tj.m` — Base solver
- `G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\mydata_Drosophila_melanogaster.m` — Observed data

**Larvaworld Python**:

- `src/larvaworld/lib/model/deb/deb_equations.py` — Reference implementation (use as model for porting style)
- `src/larvaworld/lib/model/deb/generalized_animal.py` — Species-specific parameter loading

## Next Steps

1. **Immediate (this session)**: Finish Phase 1 implementation in `univariate_data_plots.py`
2. **Generate tWw_f plot**: Verify it matches visually against `mass_vs_time.png`
3. **Document findings**: Note any divergences between Larvaworld Table S1/S2 and MATLAB `dget_ELH`
4. **Commit**: Phase 1 framework + single working plot
5. **Later session**: Implement Phase 2 (ODE solvers) + complete tR\_\* plots

---

**Author**: Claude Haiku 4.5
**Status**: Framework ready; ODE solvers deferred (substantial but straightforward porting)
