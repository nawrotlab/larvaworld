# Table 4: Larva Group Initial Spatial Placement Parameters

## 📋 Table

### Larva Group Initial Spatial Placement Parameters

| Parameter | Description |
|-----------|-------------|
| **N** | Number of virtual larvae in the group |
| **location** | Centre of the spatial distribution in the arena |
| **scale** | Spatial extent of the distribution |
| **shape** | Shape of the distribution (circular/rectangular/oval) |
| **placement** | Placement within the distribution's shape (uniform/normal/periphery) |
| **orientation** | Range of initial spatial body orientations |

---

## Detailed Descriptions

### N (Number of Larvae)
- **Type**: Integer
- **Range**: 1 to hundreds
- **Description**: Total number of virtual larvae in the group
- **Examples**:
  - `N=1`: Single larva experiments
  - `N=10`: Small group dynamics
  - `N=100`: Large population studies

### location (Center Point)
- **Type**: (x, y) tuple in meters
- **Range**: Within arena bounds
- **Description**: Center point of the spatial distribution
- **Examples**:
  - `location=(0.0, 0.0)`: Arena center
  - `location=(-0.04, 0.0)`: Left side, 4cm from center
  - `location=(0.0, 0.03)`: Top side, 3cm from center
- **Note**: Coordinates in meters, relative to arena center

### scale (Distribution Size)
- **Type**: (x_scale, y_scale) tuple in meters
- **Range**: 0 to arena dimensions
- **Description**: Spatial extent of the distribution in x and y directions
- **Examples**:
  - `scale=(0.005, 0.005)`: Tight 5mm × 5mm cluster
  - `scale=(0.02, 0.02)`: 2cm × 2cm area
  - `scale=(0.04, 0.01)`: Elongated 4cm × 1cm strip
- **Note**: Defines the "spread" of larvae around location center

### shape (Distribution Shape)
- **Type**: String enum
- **Options**: 
  - `"circular"`: Round distribution
  - `"rectangular"`: Square/rectangular distribution
  - `"oval"`: Elliptical distribution
- **Description**: Geometric shape of the spatial distribution
- **Usage**:
  - `circular`: Isotropic experiments, natural clustering
  - `rectangular`: Arena-aligned placement, strip assays
  - `oval`: Elongated distributions

### placement (Distribution Mode)
- **Type**: String enum
- **Options**:
  - `"uniform"`: Evenly distributed throughout shape
  - `"normal"`: Gaussian/bell curve distribution (dense center)
  - `"periphery"`: Placed on edge/border of shape
- **Description**: How larvae are positioned within the distribution shape
- **Examples**:
  - `uniform`: Random positions, all equally likely
  - `normal`: Most larvae near center, fewer at edges
  - `periphery`: All larvae start at boundary (e.g., arena edge)

### orientation (Initial Heading)
- **Type**: (min_angle, max_angle) tuple in degrees
- **Range**: 0-360° or (-180, 180)°
- **Description**: Range of initial body orientations
- **Examples**:
  - `orientation=(0, 360)`: Random all directions
  - `orientation=(90, 90)`: All facing up (90°)
  - `orientation=(-30, 30)`: Mostly facing right, ±30° variation
- **Note**: 0° is right, 90° is up, angles in degrees

---

## Configuration Examples

### Example 1: Central Cluster
```python
larva_group = {
    'N': 10,
    'location': (0.0, 0.0),
    'scale': (0.01, 0.01),
    'shape': 'circular',
    'placement': 'uniform',
    'orientation': (0, 360)
}
```
**Result**: 10 larvae clustered in 1cm radius at arena center, random orientations

### Example 2: Peripheral Ring
```python
larva_group = {
    'N': 20,
    'location': (0.0, 0.0),
    'scale': (0.04, 0.04),
    'shape': 'circular',
    'placement': 'periphery',
    'orientation': (0, 360)
}
```
**Result**: 20 larvae arranged in a ring at 4cm radius from center

### Example 3: Oriented Line
```python
larva_group = {
    'N': 5,
    'location': (-0.04, 0.0),
    'scale': (0.005, 0.02),
    'shape': 'rectangular',
    'placement': 'uniform',
    'orientation': (90, 90)
}
```
**Result**: 5 larvae in a vertical strip on the left, all facing up

### Example 4: Chemotaxis Start
```python
larva_group = {
    'N': 8,
    'location': (-0.04, 0.0),
    'scale': (0.005, 0.02),
    'shape': 'oval',
    'placement': 'normal',
    'orientation': (-30, 30)
}
```
**Result**: 8 larvae starting from left side, Gaussian distribution, mostly facing right toward potential odor source

---

## Usage in ReadTheDocs

```rst
Larva Group Placement
~~~~~~~~~~~~~~~~~~~~~

Initial spatial placement of larva groups is controlled by six key parameters.

.. list-table:: Spatial Placement Parameters
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - **N**
     - Number of virtual larvae in the group (integer, 1+)
   * - **location**
     - Center of distribution in arena (x, y) in meters
   * - **scale**
     - Spatial extent of distribution (x_scale, y_scale) in meters
   * - **shape**
     - Distribution shape: ``circular``, ``rectangular``, or ``oval``
   * - **placement**
     - Distribution mode: ``uniform``, ``normal``, or ``periphery``
   * - **orientation**
     - Initial heading range (min_deg, max_deg) in degrees

**Configuration Example:**

.. code-block:: python

   from larvaworld import reg
   
   # Create larva group with custom placement
   group = reg.gen.LarvaGroup(
       model='explorer',          # Larva model ID
       N=10,                      # 10 larvae
       location=(0.0, 0.0),      # Arena center
       scale=(0.02, 0.02),       # 2cm × 2cm area
       shape='circular',          # Circular distribution
       placement='uniform',       # Uniformly distributed
       orientation=(0, 360),      # Random orientations
       color='blue'               # Visualization color
   )

**Common Patterns:**

**Chemotaxis Experiments:**
   Start larvae on one side, facing potential odor source
   
   .. code-block:: python
   
      location=(-0.04, 0.0)      # Left side
      orientation=(-30, 30)       # Facing right ±30°

**Dispersion Studies:**
   Tight initial cluster to measure spreading
   
   .. code-block:: python
   
      location=(0.0, 0.0)        # Center
      scale=(0.005, 0.005)       # Tight 5mm cluster
      placement='normal'          # Gaussian distribution

**Competition/Games:**
   Two groups on opposite sides
   
   .. code-block:: python
   
      # Group 1 (Left)
      location=(-0.03, 0.0)
      
      # Group 2 (Right)
      location=(0.03, 0.0)

For visual examples, see :ref:`figure-larva-group-params` (Figure 7 from paper).
```

---

## Code Implementation

### Location in Codebase
`/src/larvaworld/lib/reg/larvagroup.py`

### LarvaGroup Class
```python
class LarvaGroup:
    """Virtual larva group with spatial distribution parameters."""
    
    N: int  # Number of larvae
    location: tuple  # (x, y) center in meters
    scale: tuple  # (x_scale, y_scale) in meters
    shape: str  # 'circular' | 'rectangular' | 'oval'
    placement: str  # 'uniform' | 'normal' | 'periphery'
    orientation: tuple  # (min_deg, max_deg)
    
    def generate_positions(self):
        """Generate initial positions based on parameters."""
        if self.shape == 'circular':
            # Generate within circle
            ...
        elif self.placement == 'periphery':
            # Place on boundary
            ...
```

---

## Related Content

- **Figure 7** (paper): Visual representation of LarvaGroup parameters panel
- **Tutorial**: `library_interface.ipynb` - Creating custom larva groups
- **Experiments**: All experiments use these parameters for initial setup

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **LaTeX**: Lines 806-828 in main.tex
- **Label**: `tab:distribution`
- **Related Figure**: Fig 7 (`LarvaGroup.png`)

