# Larva Agent Architecture

The `LarvaSim` class represents a complete simulated _Drosophila_ larva, combining physical morphology, sensorimotor control, energetics, and physics.

---

## LarvaSim Structure

```{mermaid}
graph TB
    subgraph LARVASIM ["LarvaSim"]
        AGENT[LarvaSim<br/>Simulated Larva]:::larva

        subgraph PHYSICAL ["Physical Body"]
            BODY[Segmented Body<br/>11-13 segments]:::body
            CONTOUR[Body Contour<br/>Collision detection]:::body
        end

        subgraph BRAIN_SYS ["Brain System"]
            BRAIN[Brain<br/>DefaultBrain or NengoBrain]:::brain
            SENSORS[Sensors<br/>Olfactor, Toucher<br/>Wind, Thermo]:::sensors
            MEMORY[Memory<br/>RL or MB<br/>optional]:::neural
        end

        subgraph LOCOMOTOR_SYS ["Locomotor System"]
            LOCOMOTOR[Locomotor<br/>Motor coordination]:::motor
            CRAWLER[Crawler<br/>Forward motion]:::behavior
            TURNER[Turner<br/>Directional changes]:::behavior
            FEEDER[Feeder<br/>Feeding behavior]:::behavior
            INTERMITTER[Intermitter<br/>State switching]:::behavior
        end

        subgraph ENERGY_SYS ["Energy System"]
            DEB[DEB Model<br/>Metabolism]:::energy
            GUT[Gut<br/>Digestion]:::energy
            RESERVES[Reserves<br/>E, E_R, E_H]:::energy
        end

        subgraph PHYSICS_SYS ["Physics Control (BaseController)"]
            PHYSICS[Physics<br/>Damping, Spring]:::physics
            MOTION[Motion<br/>Velocity/Force modes]:::physics
        end
    end

    AGENT --> BODY
    AGENT --> BRAIN
    AGENT --> LOCOMOTOR
    AGENT --> DEB
    AGENT --> PHYSICS

    BRAIN --> SENSORS
    BRAIN -.-> MEMORY

    LOCOMOTOR --> CRAWLER
    LOCOMOTOR --> TURNER
    LOCOMOTOR --> FEEDER
    LOCOMOTOR --> INTERMITTER

    DEB --> GUT
    DEB --> RESERVES

    SENSORS -.->|Sensing| BODY
    LOCOMOTOR -.->|Commands| MOTION
    MOTION -.->|Apply| BODY
    PHYSICS -.->|Mechanics| BODY
    FEEDER -.->|Eating| GUT
    DEB -.->|Metabolism| BODY

    classDef larva fill:#2c3e50,stroke:#34495e,stroke-width:3px,color:#ffffff
    classDef body fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#ffffff
    classDef sensors fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#ffffff
    classDef brain fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#ffffff
    classDef neural fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#ffffff
    classDef behavior fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    classDef energy fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:#000000
    classDef motor fill:#1abc9c,stroke:#16a085,stroke-width:2px,color:#ffffff
    classDef physics fill:#e91e63,stroke:#c2185b,stroke-width:2px,color:#ffffff
```

---

## Inheritance Structure

`LarvaSim` inherits from two parent classes:

```python
class LarvaSim(LarvaMotile, BaseController):
    pass
```

### LarvaMotile

**Purpose**: Core agent behavior

**Key Methods**:

- `sense()`: Sensory input
- `think()`: Brain processing
- `act()`: Locomotor output
- `move()`: Physics update

### BaseController

**Purpose**: Physics control (Box2D integration)

**Key Methods**:

- `apply_motion()`: Velocity/force application
- `apply_damping()`: Resistance
- `apply_spring()`: Body segment coupling

---

## Component Details

### 1. Physical Body

**Class**: `LarvaSegmented` (base for `LarvaMotile`)

**Attributes**:

- `Nsegs`: Number of segments (11-13)
- `length`: Body length (mm)
- `radius`: Body radius (mm)
- `segs`: Collection of body segments

**Purpose**: Morphology, segments, and sensor placement

---

### 2. Brain System

**Class**: `Brain` (DefaultBrain or NengoBrain)

See {doc}`brain_module_architecture` for details.

**Components**:

- **Sensors**: Olfactor, Toucher, Windsensor, Thermo
- **Modalities**: Sensory processing channels
- **Memory** (optional): RL or MB learning

---

### 3. Locomotor System

**Class**: `Locomotor`

**Modules**:

- **Crawler**: Forward peristaltic waves
- **Turner**: Reorientation maneuvers
- **Feeder**: Food intake
- **Intermitter**: State switching (run/pause)
- **Interference**: Crawl-turn coupling

---

### 4. Energy System

**Class**: `DEB` (Dynamic Energy Budget)

**Reserves**:

- `E`: Energy reserves
- `E_R`: Reproduction buffer
- `E_H`: Maturity

**Processes**:

- Assimilation from gut
- Growth of structure
- Maturation
- Starvation

---

### 5. Physics Control

**Class**: `BaseController`

**Options**:

- `physics_model`: Box2D multisegment body
- `body_spring`: Segment coupling stiffness
- `body_damping`: Resistance coefficient

---

## Configuration

### Model Selection

```python
from larvaworld.lib import reg

# List available models
model_ids = reg.conf.Model.confIDs
print(model_ids)

# Load model configuration
model_conf = reg.conf.Model.getID("explorer")
```

### Custom Agent

```python
from larvaworld.lib.sim import ExpRun

run = ExpRun(
    experiment="dish",
    larva_groups=[
        {
            "model": "custom",
            "N": 10,
            "brain": {"modalities": ["olfaction", "touch"]},
            "body": {"Nsegs": 11, "length": 3.5},
            "locomotor": {"crawler": {"f": 1.2}, "turner": {"ang_v": 0.5}}
        }
    ]
)
```

---

## Related Documentation

- {doc}`brain_module_architecture` - Brain details
- {doc}`arenas_and_substrates` - Environments
- {doc}`../concepts/module_interaction` - Runtime interactions
