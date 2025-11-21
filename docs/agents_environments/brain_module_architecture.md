# Brain Module Architecture

The Brain module integrates sensory input, memory systems, and locomotor control. Larvaworld provides two implementations: `DefaultBrain` (standard) and `NengoBrain` (neural network-based).

---

## Brain architecture overview

```{mermaid}
graph TD
    %% Brain implementations
    subgraph Brain Implementations
        BrainBase[\"Brain\"<br/>Base class]:::base
        DefaultBrain[\"DefaultBrain\"<br/>Standard implementation]:::impl
        NengoBrain[\"NengoBrain\"<br/>Nengo-based implementation]:::impl
    end

    BrainBase --> DefaultBrain
    BrainBase --> NengoBrain

    %% Modalities
    subgraph Modalities
        OlfMod[\"olfaction\"<br/>odor sensing]:::modality
        TouchMod[\"touch\"<br/>mechanoreception/feeding]:::modality
        ThermoMod[\"thermosensation\"<br/>temperature]:::modality
        WindMod[\"windsensation\"<br/>wind/airflow]:::modality
    end

    %% Sensors
    OlfSensor[\"Olfactor\"<br/>odor sensor]:::sensor
    TouchSensor[\"Toucher\"<br/>contact/food sensor]:::sensor
    ThermoSensor[\"Thermo\"<br/>temperature sensor]:::sensor
    WindSensor[\"Windsensor\"<br/>wind sensor]:::sensor

    %% Memory modules
    RLmem[\"RLmemory\"<br/>reinforcement learning]:::memory
    NullMem[\"No memory\"]:::memory

    %% Locomotor interface
    subgraph Locomotor System
        Locomotor[\"Locomotor\"<br/>crawl/turn/feed]:::locomotor
        A_in[\"A_in\"<br/>sensory drive]:::signal
        MotorCmds[\"motor commands\"<br/>crawl/turn/feed rates]:::output
    end

    %% Wiring: Brain -> Modalities
    DefaultBrain --> OlfMod
    DefaultBrain --> TouchMod
    DefaultBrain --> ThermoMod
    DefaultBrain --> WindMod

    %% Wiring: Modalities -> Sensors
    OlfMod --> OlfSensor
    TouchMod --> TouchSensor
    ThermoMod --> ThermoSensor
    WindMod --> WindSensor

    %% Wiring: Memory attachment (example: olfaction)
    OlfMod --> RLmem
    TouchMod --> NullMem
    ThermoMod --> NullMem
    WindMod --> NullMem

    %% Wiring: Modalities -> Locomotor
    OlfMod --> A_in
    TouchMod --> A_in
    ThermoMod --> A_in
    WindMod --> A_in

    A_in --> Locomotor
    Locomotor --> MotorCmds

    %% Color definitions
    classDef base fill:#2c3e50,stroke:#34495e,stroke-width:3px,color:#ffffff
    classDef impl fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#ffffff
    classDef modality fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#ffffff
    classDef sensor fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#ffffff
    classDef signal fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#ffffff
    classDef memory fill:#e91e63,stroke:#c2185b,stroke-width:2px,color:#ffffff
    classDef locomotor fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    classDef output fill:#f1c40f,stroke:#f39c12,stroke-width:3px,color:#000000
```

---

## Sensory Modalities

The brain organizes sensors into **modalities**, each processing a specific sensory channel:

| Modality | Sensor Class | Signal | Memory | Purpose |
|----------|--------------|--------|--------|---------|
| **olfaction** | `Olfactor` | `A` (float) | Optional | Odor detection |
| **touch** | `Toucher` | `A` (float) | Optional | Contact/food sensing |
| **thermosensation** | `Thermosensor` | `A` (float) | Optional | Temperature sensing |
| **windsensation** | `Windsensor` | `A` (float) | Optional | Wind/airflow sensing |

### Modality Structure

```python
modality = {
    "sensor": Olfactor(),           # Sensor instance
    "func": sense_odors,            # Processing function
    "A": 0.0,                       # Sensory signal (output)
    "mem": Memory() or None         # Optional memory
}
```

---

## Sensor Modules

### Olfactor

**Purpose**: Odor detection

**Key Attributes**:
- `gain`: Sensory gain (modulated by memory)
- `sensed_odor`: Processed odor concentration

**Processing**:
1. Query odorscape at larva position
2. Apply Weber-Fechner law
3. Modulate by gain (if learning enabled)

**Code Location**: `/lib/model/modules/sensor.py` (class `Olfactor`)

---

### Toucher

**Purpose**: Contact and food sensing

**Key Attributes**:
- `contacts`: List of detected contacts
- `on_food`: Boolean (food contact)

**Processing**:
1. Check collisions with obstacles
2. Detect food patches at position

**Code Location**: `/lib/model/modules/sensor.py` (class `Toucher`)

---

### Windsensor

**Purpose**: Wind/airflow detection

**Key Attributes**:
- `wind_direction`: Vector (x, y)
- `wind_speed`: Magnitude

**Code Location**: `/lib/model/modules/sensor.py` (class `Windsensor`)

---

### Thermo

**Purpose**: Temperature sensing

**Key Attributes**:
- `temperature`: Current temperature (°C)

**Code Location**: `/lib/model/modules/sensor.py` (class `Thermosensor`)

---

## Memory Modules

Memory modules **attach to sensory modalities** and modulate their gain through learning.

### RLmemory

**Algorithm**: Q-learning (reinforcement learning)

**Mechanism**:
- Increase gain for rewarded stimuli
- Decrease gain for non-rewarded stimuli

**Update Rule**:

```python
if reward > 0:
    gain[odor] += α * reward
else:
    gain[odor] -= β * punishment
```

**Code Location**: `/lib/model/modules/memory.py` (`RLmemory` class)

---

### RemoteBrianModelMemory (MB memory)

**Algorithm**: Mushroom Body model (Hebbian learning)

**Mechanism**:
- KC-MBON synaptic plasticity
- Reward-modulated learning

**Code Location**: `/lib/model/modules/memory.py` (`RemoteBrianModelMemory` class)

---

## Locomotor Integration

The brain coordinates locomotor modules through the `Locomotor` class:

```{mermaid}
graph LR
    BRAIN[Brain] --> LOCOMOTOR[Locomotor]
    LOCOMOTOR --> CRAWLER[Crawler]
    LOCOMOTOR --> TURNER[Turner]
    LOCOMOTOR --> FEEDER[Feeder]
    LOCOMOTOR --> INTERMITTER[Intermitter]
```

### Brain → Locomotor Flow

1. **Brain.step()**: Process sensory input
2. **Brain.compute()**: Generate motor commands
3. **Locomotor.compute()**: Coordinate modules
4. **Modules.step()**: Execute behaviors

---

## DefaultBrain

**Implementation**: Rule-based sensorimotor control

**Step Sequence**:

```python
def step(self):
    # 1. Collect sensory input
    for modality in self.modalities.values():
        modality["func"]()  # e.g., sense_odors()
    
    # 2. Update memory (if present)
    for modality in self.modalities.values():
        if modality["mem"] is not None:
            modality["mem"].step(reward=self.feeder.amount_eaten)
    
    # 3. Generate locomotor commands
    self.locomotor.compute()
```

---

## NengoBrain

**Implementation**: Spiking neural network (Nengo)

**Architecture**:
- **Input**: Sensory neurons (olfactory, tactile, etc.)
- **Hidden**: Processing layers
- **Output**: Motor neurons (forward, turn)

**Step Sequence**:

```python
def step(self):
    # 1. Map sensors to input neurons
    self.sim.data[self.input_neurons] = self.get_sensory_input()
    
    # 2. Run Nengo simulation (1 timestep)
    self.sim.run_steps(1)
    
    # 3. Map output neurons to locomotor
    self.locomotor.crawler.fov = self.sim.data[self.forward_neuron]
    self.locomotor.turner.ang_v = self.sim.data[self.turn_neuron]
```

---

## Configuration

### Enable Specific Modalities

```python
larva_groups = [
    {
        "model": "custom",
        "N": 10,
        "brain": {
            "modalities": ["olfaction", "touch"]  # Only these two
        }
    }
]
```

### Attach Memory

```python
larva_groups = [
    {
        "model": "RL_model",
        "N": 10,
        "brain": {
            "modalities": {
                "olfaction": {"memory": "RL"}  # RL memory on olfaction
            }
        }
    }
]
```

### Use NengoBrain

```python
larva_groups = [
    {
        "model": "nengo",
        "N": 10,
        "brain": {"brain_class": "NengoBrain"}
    }
]
```

---

## Related Documentation

- {doc}`larva_agent_architecture` - Complete agent structure
- {doc}`../concepts/module_interaction` - Runtime interactions
- {doc}`../working_with_larvaworld/single_experiments` - Olfactory learning example
