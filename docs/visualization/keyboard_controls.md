# Keyboard Controls

Larvaworld's visualization window supports interactive keyboard and mouse controls for real-time exploration.

---

## Quick Reference

| **Screen**    |      | **Drawing**    |     | **Color**  |     | **Interaction** |     | **Simulation/Storage** |       |
| ------------- | ---- | -------------- | --- | ---------- | --- | --------------- | --- | ---------------------- | ----- |
| State text    | s    | midline        | m   | random     | r   | select          | L\* | snapshot               | i     |
| Timer         | t    | contour        | c   | behavior   | b   | lock screen     | f   | odorscape overlay      | o     |
| IDs           | TAB  | head           | h   | background | g   | delete          | del | pause                  | space |
| Scale bar     | n    | centroid       | e   | odorscape  | 0-9 | add             | L\* |                        |       |
| Screen Motion | ↑↓↔ | trail          | p   |            |     | inspect         | R\* |                        |       |
| Zoom          | M\*  | trail duration | +/- |            |     | dynamic graph   | q   |                        |       |

**Legend**:

- `L*` = Left mouse button
- `R*` = Right mouse button
- `M*` = Mouse scroll wheel

---

## Screen Controls

| Key      | Action     | Description                                  |
| -------- | ---------- | -------------------------------------------- |
| **s**    | State text | Toggle status overlay (simulation time, fps) |
| **t**    | Timer      | Show/hide elapsed time                       |
| **TAB**  | IDs        | Show/hide larva IDs                          |
| **n**    | Scale bar  | Toggle scale bar                             |
| **↑↓←→** | Pan        | Move viewport                                |
| **M\***  | Zoom       | Scroll to zoom in/out                        |

---

## Drawing Controls

| Key     | Action         | Description              |
| ------- | -------------- | ------------------------ |
| **m**   | Midline        | Toggle 12-point midline  |
| **c**   | Contour        | Toggle body contour      |
| **h**   | Head           | Highlight head segment   |
| **e**   | Centroid       | Show body centroid       |
| **p**   | Trail          | Toggle trajectory trails |
| **+/-** | Trail duration | Adjust trail length      |

---

## Color Modes

| Key     | Action     | Description                                    |
| ------- | ---------- | ---------------------------------------------- |
| **r**   | Random     | Random colors per larva                        |
| **b**   | Behavior   | Color by behavior state (run/pause/turn)       |
| **g**   | Background | Toggle background color                        |
| **0-9** | Odorscape  | Show odor concentration (0=off, 1-9=intensity) |

---

## Interaction

| Key     | Action        | Description                    |
| ------- | ------------- | ------------------------------ |
| **L\*** | Select/Add    | Left-click to select/add larva |
| **R\*** | Inspect       | Right-click for detailed info  |
| **del** | Delete        | Remove selected larva          |
| **f**   | Lock screen   | Lock camera to follow larva    |
| **q**   | Dynamic graph | Show real-time metrics plot    |

---

## Simulation Control

| Key       | Action    | Description                 |
| --------- | --------- | --------------------------- |
| **space** | Pause     | Pause/resume simulation     |
| **i**     | Snapshot  | Save current frame as image |
| **o**     | Odorscape | Toggle odorscape overlay    |

---

## Usage Examples

### Following a Larva

1. **Left-click** on a larva to select it
2. Press **f** to lock camera
3. Press **m** to show midline
4. Press **p** to show trail

### Creating a Video

1. Run experiment with `screen_kws={'vis_mode': 'video'}`
2. Adjust drawing options (**m**, **c**, **p**)
3. Select color mode (**r** or **b**)
4. Simulation auto-exports to MP4

### Inspecting Behavior

1. **Right-click** on a larva
2. View detailed metrics in popup
3. Press **q** for real-time plots

---

## Related Documentation

- {doc}`visualization_snapshots` - Visualization examples
- {doc}`web_applications` - Web-based dashboards
- {doc}`../working_with_larvaworld/replay` - Replay mode
