# Table 1: Visualization Keyboard Controls

## 📋 Table

### Visualization Default Keyboard/Mouse Controls

| **Screen**    |      | **Drawing**    |     | **Color**  |     | **Interaction** |     | **Simulation/Storage** |       |
| ------------- | ---- | -------------- | --- | ---------- | --- | --------------- | --- | ---------------------- | ----- |
| State text    | s    | midline        | m   | random     | r   | select          | L\* | snapshot               | i     |
| Timer         | t    | contour        | c   | behavior   | b   | lock screen     | f   | odorscape overlay      | o     |
| IDs           | TAB  | head           | h   | background | g   | delete          | del | pause                  | space |
| Scale bar     | n    | centroid       | e   | odorscape  | 0-9 | add             | L\* |                        |       |
| Screen Motion | ↑↓↔ | trail          | p   |            |     | inspect         | R\* |                        |       |
|               |      | trail duration | +/- |            |     | dynamic graph   | q   |                        |       |

**Legend**:

- `L*` = Left mouse button
- `R*` = Right mouse button
- `M*` = Mouse scroll wheel (center button)

---

## Detailed Explanations

### Screen Controls

**State text (`s`)**

- Toggles a status overlay with simulation state information
- Helpful for checking whether the simulation is running or paused
- Useful when stepping through simulations interactively

**Timer (`t`)**

- Shows elapsed simulation/replay time
- Displays in minutes:seconds format
- Helps track temporal dynamics

**IDs (`TAB`)**

- Shows unique identifier for each larva
- Critical for multi-agent experiments
- Useful for tracking specific individuals

**Scale bar (`n`)**

- Toggles a scale bar showing real-world dimensions
- Useful for understanding spatial context (arena size, larva size)
- Essential when comparing different arena configurations

**Zoom (`M*` - Mouse wheel)**

- Scroll up to zoom in, down to zoom out
- Allows close inspection of individual behavior
- Can zoom from full arena to single segment detail

**Screen Motion (Arrow keys `↑↓←→`)**

- Pan the viewport without zooming
- Useful for large arenas that don't fit in one view
- Can follow action while larvae disperse

---

### Drawing Controls

**Midline (`m`)**

- Toggles 12-point midline representation
- Shows body posture and bending
- Essential for analyzing crawling kinematics

**Contour (`c`)**

- Shows full body outline/contour
- More detailed than midline
- Useful for collision detection visualization

**Head (`h`)**

- Highlights the head segment
- Shows orientation clearly
- Helps identify directional movement

**Centroid (`e`)**

- Highlights body center of mass
- Useful for trajectory analysis
- Simpler view for navigation studies

**Trail (`p`)**

- Shows past trajectory as a colored line
- Duration adjustable with `+`/`-` keys
- Essential for understanding path patterns

**Trail Duration (`+` / `-`)**

- `+` increases trail length (show more history)
- `-` decreases trail length
- Adjust based on arena size and larva speed

---

### Color Controls

**Random (`r`)**

- Assigns random colors to each larva
- Helps distinguish individuals
- Useful for multi-agent visualization

**Behavior (`b`)**

- Colors larvae based on current behavioral state
- Examples: crawling (green), paused (red), turning (blue)
- Real-time indication of behavioral dynamics

**Background (`g`)**

- Toggles between light and dark background
- Dark background often preferred for trajectories
- Light background better for printing/presentations

**Odorscape (`0-9`)**

- Number keys select which odor to visualize
- Shows concentration heatmap
- Dynamic visualization of chemical landscape

---

### Interaction Controls

**Select (`L*` - Left-click)**

- Click on a larva to select it
- Selected larva highlighted
- Enables larva-specific actions

**Lock screen (`f`)**

- Locks camera to follow selected larva
- Camera automatically pans to keep larva centered
- Excellent for single-larva detailed analysis

**Delete (`DEL`)**

- Removes selected larva or object
- Useful for testing scenarios
- Can remove food sources, borders, etc.

**Add (`L*` in empty space)**

- Click empty arena to add new larva
- Position determined by click location
- Allows dynamic experiment modification

**Inspect (`R*` - Right-click)**

- Right-click object for detailed information
- Shows parameters, state, position
- Debugging and analysis tool

**Dynamic graph (`q`)**

- Toggles real-time behavior plots
- Shows time-series of selected variables
- Overlay on visualization screen

---

### Simulation/Storage Controls

**Snapshot (`i`)**

- Captures current frame as PNG image
- Saved to experiment directory
- Useful for figures and presentations

**Odorscape overlay (`o`)**

- Toggles visualization of odorscape layers
- Shows concentration heatmap in the arena
- Combine with snapshots to capture odorscapes in exported images

**Pause (`space`)**

- Pause/resume simulation or replay
- Allows inspection of specific moments
- Simulation state preserved

---

## Usage in ReadTheDocs

```rst
Keyboard Controls
~~~~~~~~~~~~~~~~~

Real-time visualization in Larvaworld supports extensive keyboard and mouse
controls for interactive exploration.

.. list-table:: Visualization Controls
   :header-rows: 1
   :widths: 15 25 10 50

   * - Category
     - Action
     - Key
     - Description
   * - **Screen**
     - State text
     - ``s``
     - Toggle simulation state overlay (running/paused, etc.)
   * - **Screen**
     - Timer
     - ``t``
     - Show/hide simulation timer
   * - **Screen**
     - IDs
     - ``TAB``
     - Toggle larva unique identifiers
   * - **Screen**
     - Scale bar
     - ``n``
     - Toggle display of real-world scale bar
   * - **Drawing**
     - Midline
     - ``m``
     - Toggle 12-point midline display
   * - **Drawing**
     - Contour
     - ``c``
     - Show/hide body contour
   * - **Drawing**
     - Trail
     - ``p``
     - Toggle trajectory trails
   * - **Drawing**
     - Trail duration
     - ``+`` / ``-``
     - Adjust trail length
   * - **Color**
     - Random
     - ``r``
     - Random larva colors
   * - **Color**
     - Behavior
     - ``b``
     - Color by current behavior
   * - **Interaction**
     - Select
     - Left-click
     - Select a larva
   * - **Interaction**
     - Lock screen
     - ``f``
     - Follow selected larva
   * - **Simulation**
     - Snapshot
     - ``i``
     - Capture current frame
   * - **Simulation**
     - Pause
     - ``space``
     - Pause/resume

**Mouse Controls:**

- **Left-click**: Select larvae or add new larva
- **Right-click**: Inspect object details
- **Scroll wheel**: Zoom in/out
- **Arrow keys**: Pan viewport

.. note::
   A video tutorial demonstrating these controls is available at
   :ref:`visualization-tutorial-video`.

For visual examples of the visualization capabilities, see
:ref:`figure-visualization-snapshots`.
```

---

## Developer Note

**Requested**: Include with explanation of what each shortcut does.
**Ideal addition**: Video tutorial showing shortcuts in action.

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **LaTeX**: Lines 398-420 in main.tex
- **Label**: `tab:visualization`
- **Related**: Figure 2 (Visualization Snapshots) shows results of using these controls
