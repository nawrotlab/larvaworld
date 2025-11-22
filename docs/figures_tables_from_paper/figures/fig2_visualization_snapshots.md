# Figure 2: Real-time Visualization Snapshots

## 📊 Images

### (a) Real-animal behavior - Black background

![Replay Black Background](fig2a_replay_black.png)

**Description**: Real-animal behavior displayed on a black background with visible larval IDs and reconstructed locomotion trajectories.

---

### (b) Close-up - Midline tracking

![Replay Zoom](fig2b_replay_zoom.png)

**Description**: A close-up showing the 12-point midline tracking of individual larvae.

---

### (c) Simulated odor preference experiment

![Simulation Preference](fig2c_sim_preference.png)

**Description**: A simulated odor preference experiment with an appetitive odor source placed on the left.

---

### (d) Simulated olfactory landscape

![Simulation Game](fig2d_sim_game.png)

**Description**: A simulated olfactory sensory landscape ("odorscape") attracting two competing larval groups toward a central odor source.

---

## Full Caption

**Real-time visualization of reconstructed real-animal experiments and agent simulations.**

Snapshots from the experimental arena illustrate:

- **(a)** real-animal behavior displayed on a black background with visible larval IDs and reconstructed locomotion trajectories
- **(b)** a close-up showing the 12-point midline tracking of individual larvae
- **(c)** a simulated odor preference experiment with an appetitive odor source placed on the left
- **(d)** a simulated olfactory sensory landscape ("odorscape") attracting two competing larval groups toward a central odor source

---

## Purpose

These "nice pictures" (developer request) showcase:

- ✅ **Real data visualization**: Import and replay of real experimental data
- ✅ **High-resolution tracking**: 12-point midline representation
- ✅ **Simulation capabilities**: Virtual experiments with odor sources
- ✅ **Multi-agent interactions**: Competitive scenarios with multiple groups
- ✅ **Visual appeal**: Demonstrates the platform's visualization features

---

## Usage in ReadTheDocs

**Placement**: Visualization section / Gallery

```rst
Real-time Visualization
~~~~~~~~~~~~~~~~~~~~~~~

Larvaworld provides real-time visualization of both real experimental data
and simulated experiments.

.. figure:: _static/images/fig2_combined.png
   :alt: Visualization Snapshots
   :align: center
   :width: 100%

   **Figure 2**: Real-time visualization capabilities.
   **(a)** Real-animal trajectories on black background with IDs.
   **(b)** Close-up of 12-point midline tracking.
   **(c)** Simulated odor preference assay.
   **(d)** Multi-group competition with odorscape.

**Key Features:**

- **Real Data Replay**: Import and visualize experimental trajectories
- **Midline Tracking**: High-resolution 12-point body representation
- **Odor Visualization**: Dynamic odorscape rendering
- **Multi-agent Display**: Multiple larvae with distinct colors/IDs
- **Interactive Controls**: See :ref:`table-keyboard-controls` for full list

For keyboard shortcuts and real-time controls, see Table 1.
```

---

## Keyboard Controls

These visualizations are controlled in real-time using keyboard shortcuts
(see **Table 1: Keyboard Controls**).

Key features shown:

- **Larval IDs**: Toggle with TAB key
- **Trajectories**: Toggle with 'p', adjust duration with +/-
- **Midline/Contour**: Toggle with 'm' and 'c'
- **Background**: Change with 'g'
- **Zoom**: Mouse scroll or 'M' button
- **Selection**: Click larvae to select/track

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **Files**:
  - `/images/snapshots/1.replay_black.png` → `fig2a_replay_black.png`
  - `/images/snapshots/2.replay_zoom.png` → `fig2b_replay_zoom.png`
  - `/images/snapshots/3.sim_pref.png` → `fig2c_sim_preference.png`
  - `/images/snapshots/4.sim_game.png` → `fig2d_sim_game.png`
- **Caption** (LaTeX lines 384-391)
- **Label**: `fig:snapshots`
