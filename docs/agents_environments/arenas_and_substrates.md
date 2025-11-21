# Arenas and Substrates

This page covers environment configuration: arena geometry, food sources, odorscapes, and obstacles.

---

## Arena Types

### Circular Arena (Petri Dish)

**Default**: 90mm diameter

```python
env_params = {
    "arena": {"geometry": 0.09}  # 9cm diameter (circular)
}
```

### Rectangular Arena

```python
env_params = {
    "arena": {
        "geometry": [0.15, 0.10]  # 15cm x 10cm (width, height)
    }
}
```

### Preconfigured Arenas

```python
from larvaworld.lib import reg

# List available arenas
env_ids = reg.conf.Env.confIDs
print(env_ids)

# Load arena
env_conf = reg.conf.Env.getID("arena_200mm")
```

---

## Food Sources

Larvaworld supports three types of food distributions:

### 1. Discrete Patches

**Purpose**: Localized food sources

```python
env_params = {
    "food_params": {
        "source_groups": [
            {
                "group": "patches",
                "amount": 3,              # 3 patches
                "radius": 0.005,          # 5mm radius
                "distribution": "uniform" # Random placement
            }
        ]
    }
}
```

### 2. Food Grid

**Purpose**: Regular grid of patches

```python
env_params = {
    "food_params": {
        "source_groups": [
            {
                "group": "grid",
                "N": [4, 4],  # 4x4 grid
                "radius": 0.003
            }
        ]
    }
}
```

### 3. Uniform Substrate

**Purpose**: Continuous nutritious substrate

```python
env_params = {
    "food_params": {
        "source_groups": [
            {
                "group": "uniform",
                "quality": 1.0  # Full nutrition
            }
        ]
    }
}
```

---

## Nutritious Substrates

Larvaworld implements real experimental substrates:

| Substrate | Glucose (μg/ml) | Yeast (μg/ml) | Agar (μg/ml) | Source |
|-----------|----------------|---------------|--------------|--------|
| **standard-medium** | 100 | 50 | 16 | Kaun et al. (2007) |
| **PED-tracker** | 10* | 187.5 | 5000 | Schumann et al. (2020) |
| **cornmeal** | 70.3** | 14.1 | 6.6 | Wosniack et al. (2021) |
| **sucrose** | 17.1 | 0 | 4 | Wosniack et al. (2021) |

*Saccharose instead of glucose  
**Dextrose instead of glucose

**Usage**:

```python
env_params = {
    "food_params": {
        "substrate": "standard-medium"
    }
}
```

---

## Odorscapes

### Gaussian Plume

```python
env_params = {
    "odorscape": {
        "odor_layers": [
            {
                "id": "apple",
                "peak": [0.05, 0],    # Position (x, y)
                "spread": 0.02,        # Gaussian width (σ)
                "intensity": 1.0       # Peak concentration
            }
        ]
    }
}
```

### Linear Gradient

```python
env_params = {
    "odorscape": {
        "odor_layers": [
            {
                "id": "banana",
                "gradient": True,
                "direction": [1, 0],   # Along x-axis
                "value_range": [0, 1]  # Concentration range
            }
        ]
    }
}
```

### Multiple Odors

```python
env_params = {
    "odorscape": {
        "odor_layers": [
            {"id": "odorA", "peak": [-0.03, 0], "spread": 0.02},
            {"id": "odorB", "peak": [0.03, 0], "spread": 0.02}
        ]
    }
}
```

---

## Obstacles and Borders

### Arena Borders

```python
env_params = {
    "arena": {
        "geometry": [0.15, 0.10],
        "border_list": [
            {"vertices": [[0, 0], [0.15, 0], [0.15, 0.10], [0, 0.10]]}
        ]
    }
}
```

### Internal Obstacles

```python
env_params = {
    "arena": {
        "obstacles": [
            {
                "vertices": [[0.05, 0.04], [0.10, 0.04], [0.10, 0.06], [0.05, 0.06]],
                "type": "wall"
            }
        ]
    }
}
```

---

## Larva Initial Placement

Control where larvae start:

```python
larva_groups = [
    {
        "model": "explorer",
        "N": 20,
        "distribution": {
            "mode": "uniform",       # Distribution mode
            "loc": [0, 0],           # Center position
            "s": 0.01,               # Spread radius (1cm)
            "shape": "circle"        # Shape
        }
    }
]
```

### Distribution Modes

| Mode | Description |
|------|-------------|
| `"uniform"` | Uniform random within shape |
| `"periphery"` | Ring around center |
| `"line"` | Linear arrangement |
| `"grid"` | Regular grid |

See [Table 4](../figures_tables_from_paper/tables/table4_larva_placement.md) for details.

---

## Complete Example

```python
from larvaworld.lib.sim import ExpRun

run = ExpRun(
    experiment="custom_foraging",
    env_params={
        "arena": {"geometry": [0.2, 0.2]},
        "food_params": {
            "source_groups": [
                {"group": "patches", "amount": 5, "radius": 0.008}
            ],
            "substrate": "standard-medium"
        },
        "odorscape": {
            "odor_layers": [
                {"id": "food_odor", "peak": [0.05, 0.05], "spread": 0.03}
            ]
        }
    },
    larva_groups=[
        {
            "model": "forager",
            "N": 30,
            "distribution": {"mode": "periphery", "s": 0.08}
        }
    ],
    duration=15.0
)
run.simulate()
```

---

## Related Documentation

- {doc}`larva_agent_architecture` - Agent models
- {doc}`../concepts/experiment_types` - Preconfigured experiments
- {doc}`../concepts/experiment_configuration_pipeline` - Configuration system

