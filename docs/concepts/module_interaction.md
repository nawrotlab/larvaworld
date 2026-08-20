# Module Interaction

This page describes how Larvaworld's modules interact **at runtime** during simulation execution. Understanding these interactions is crucial for extending the platform or debugging behavior.

---

## High-Level Interaction Flow

```{mermaid}
sequenceDiagram
    participant User as User
    participant CLI as CLI
    participant SimEngine as SimEngine
    participant LarvaAgent as LarvaAgent
    participant Brain as Brain
    participant Crawler as Crawler
    participant Feeder as Feeder
    participant Sensor as Sensor
    participant Environment as Environment

    User->>CLI: Run simulation command
    CLI->>SimEngine: Initialize simulation
    SimEngine->>Environment: Create arena
    SimEngine->>LarvaAgent: Create larva agents

    loop Simulation Loop
        SimEngine->>LarvaAgent: Update timestep
        LarvaAgent->>Sensor: Read environment
        Sensor->>Environment: Get sensory input
        Environment-->>Sensor: Return sensory data
        Sensor-->>LarvaAgent: Sensory feedback

        LarvaAgent->>Brain: Process sensory input
        Brain->>Crawler: Generate crawling commands
        Brain->>Feeder: Generate feeding commands

        Crawler-->>LarvaAgent: Movement commands
        Feeder-->>LarvaAgent: Feeding commands

        LarvaAgent->>Environment: Execute actions
        Environment-->>LarvaAgent: Action results

        LarvaAgent-->>SimEngine: Update state
        SimEngine->>SimEngine: Record data
    end

    SimEngine-->>CLI: Simulation complete
    CLI-->>User: Results available
```

---

## Detailed Phase-by-Phase Breakdown

### Phase 1: Initialization

**Sequence**:

1. **User Command**: User runs `larvaworld Exp chemotaxis -N 20`
2. **CLI Parsing**: `argparser.py` parses arguments
3. **SimEngine Setup**: `ExpRun.__init__()` configures the run object (parameters, runtime, screen manager)
4. **Environment Creation**: during `ExpRun.setup()`, `BaseRun.build_env(...)` builds arena, sources, and sensory landscapes
5. **Agent Creation**: during `ExpRun.setup()`, `ExpRun.build_agents(...)` places `LarvaSim` agents (N=20)
6. **Module Initialization**: each `LarvaSim` initializes its brain/locomotor/sensors from the selected model configuration

**Code Path**:

```python
# cli/main.py
main() → SimModeParser.parse_args() → SimModeParser.configure()

# sim/single_run.py
ExpRun.simulate() → ABModel.run() → ExpRun.setup()
ExpRun.setup() → BaseRun.build_env() → ExpRun.build_agents()

# sim/base_run.py
BaseRun.build_env() → envs.Arena/Border/FoodGrid + create_odor_layers()

# model/agents/_larva_sim.py
LarvaSim.__init__() → LarvaMotile.__init__() → build_brain() → DefaultBrain()
```

---

### Phase 2: Simulation Loop

The core execution loop runs for `Nsteps` timesteps (computed as `duration * 60 / dt`, where `duration` is in minutes). For `dt=0.1s`, `Nsteps ≈ duration * 600`.

#### 2.1 Timestep Update

**SimEngine → LarvaAgent**: "Update to step t"

**Code**:

```python
# sim/ABM_model.py (ABModel.run)
self.sim_setup(steps, seed)
while self.running:
    self.sim_step()  # BaseRun.sim_step: step_env → step → screen_manager.step → update
```

#### 2.2 Sensory Input

**LarvaAgent → Sensors → Environment**

**Sequence**:

1. Larva queries sensors
2. Sensors read environment state (odor, food, obstacles)
3. Environment returns sensory data
4. Sensors process and return to larva

**Code (excerpt)**:

```python
# model/agents/_larva.py (LarvaMotile.step)
self.cum_dur += m.dt
self.sense()                      # placeholder; subclasses/robots can override
pos = self.olfactor_pos
if m.space.accessible_sources:
    self.food_detected = m.space.accessible_sources[self]
elif self.brain.locomotor.feeder or self.brain.toucher:
    self.food_detected = util.sense_food(
        pos, sources=m.sources, grid=m.food_grid, radius=self.radius
    )
self.resolve_carrying(self.food_detected)
lin, ang, self.feeder_motion = self.brain.step(
    pos, length=self.length, on_food=self.on_food
)
```

**Olfactory Example**:

```python
# model/modules/sensor.py (Olfactor)
from larvaworld.lib.model.modules.sensor import Olfactor

olf = Olfactor(decay_coef=0.15, perception="log")
olf.step({"odor_A": 0.6, "odor_B": 0.2})
print("First odor:", olf.first_odor_concentration)
```

#### 2.3 Neural Processing

**LarvaAgent → Brain**: "Process sensory input"

**Brain Responsibilities**:

- Integrate multi-sensory information
- Update memory (reinforcement learning, gain adaptation)
- Generate locomotor commands

**Code**:

```python
# model/modules/brain.py (DefaultBrain.step)
def step(self, pos, on_food: bool = False, **kwargs):
    # 1. Sense environment and update neural drive
    self.sense(pos=pos, reward=on_food)

    # 2. Compute locomotor commands (crawler/turner/feeder)
    return self.locomotor.step(A_in=self.A_in, on_food=on_food, **kwargs)
```

#### 2.4 Motor Command Generation

**Brain → Crawler/Turner/Feeder**: "Generate actions"

**Locomotor Coordination**:

```python
# model/modules/locomotor.py (Locomotor.step)
def step(self, A_in=0, length=1, on_food=False):
    C, F, T, If = self.crawler, self.feeder, self.turner, self.interference
    if If:
        If.cur_attenuation = 1
    if F:
        F.step()
        if F.active and If:
            If.check_module(F, "Feeder")
    if C:
        lin = C.step() * length
        if C.active and If:
            If.check_module(C, "Crawler")
    else:
        lin = 0

    # Run/pause/feed state machine
    self.step_intermitter(
        stride_completed=self.stride_completed,
        feed_motion=self.feed_motion,
        on_food=on_food,
    )

    # Turning with optional crawl–turn interference
    if T:
        if If:
            cur_att_in, cur_att_out = If.apply_attenuation(If.cur_attenuation)
        else:
            cur_att_in, cur_att_out = 1, 1
        ang = T.step(A_in=A_in * cur_att_in) * cur_att_out
    else:
        ang = 0
    return lin, ang, self.feed_motion
```

#### 2.5 Action Execution

**LarvaAgent → Environment**: "Execute actions"

**Physics Update**:

```python
# model/agents/_larva.py (LarvaMotile.step)
lin, ang, self.feeder_motion = self.brain.step(
    pos, length=self.length, on_food=self.on_food
)
self.prepare_motion(lin=lin, ang=ang)
```

**Feeding Action**:

```python
# model/agents/_larva.py (LarvaMotile.feed)
def feed(self, source, motion):
    def get_max_V_bite():
        return self.brain.locomotor.feeder.V_bite * self.V * 1000

    if motion and source is not None:
        grid = self.model.food_grid
        a_max = get_max_V_bite()
        if grid:
            V = -grid.add_cell_value(source, -a_max)
        else:
            V = source.subtract_amount(a_max)
        return V
    return 0
```

**DEB Update** (energetics):

```python
# model/agents/_larva.py (LarvaMotile.run_energetics)
def run_energetics(self, V_eaten):
    self.deb.run_check(dt=self.model.dt, X_V=V_eaten)
    self.length = self.deb.Lw * 10 / 1000
    self.mass = self.deb.Ww
    self.V = self.deb.V
```

#### 2.6 State Recording

**SimEngine**: Record data

**Data Collection**:

```python
# sim/base_run.py (BaseRun.set_collectors)
self.collectors = reg.par.get_reporters(cs=cs, agents=self.agents)

# sim/single_run.py (update/end)
self.agents.nest_record(self.collectors["step"])  # per-step variables
...
self.agents.nest_record(self.collectors["end"])   # endpoint variables

# sim/single_run.py (simulate)
self.data_collection = LarvaDatasetCollection.from_agentpy_output(self.output)
self.datasets = self.data_collection.datasets
```

---

### Phase 3: Finalization

**Sequence**:

1. **SimEngine**: Simulation loop completes
2. **Data Processing**: Convert raw data to `LarvaDataset`
3. **Storage**: Save to HDF5
4. **Visualization**: Generate plots (optional)
5. **CLI**: Return control to user

**Code**:

```python
# sim/single_run.py (ExpRun.simulate)
def simulate(self, **kwargs):
    self.run(**kwargs)  # AgentPy run loop
    if getattr(self, "aborted", False):
        self.datasets = []
        return self.datasets

    # Collect into datasets
    self.data_collection = LarvaDatasetCollection.from_agentpy_output(self.output)
    self.datasets = self.data_collection.datasets

    # Optional enrichment and storage
    if self.p.enrichment:
        for d in self.datasets:
            d.enrich(**self.p.enrichment, is_last=False)
    if self.store_data and not getattr(self, "aborted", False):
        self.store()
```

---

## Module Dependencies

### Larva Agent Dependencies

```
LarvaSim
├── LarvaMotile (parent)
│   ├── Brain
│   │   ├── Olfactor (sensor)
│   │   ├── Toucher (sensor)
│   │   ├── Memory (learning)
│   │   └── Locomotor
│   │       ├── Crawler
│   │       ├── Turner
│   │       ├── Feeder
│   │       ├── Intermitter
│   │       └── Interference
│   ├── DEB (energetics)
│   └── Body (morphology)
└── BaseController (parent)
    └── Visualization methods
```

### Environment Dependencies

```
Environment (BaseRun.build_env)
├── Space/Arena (geometry + collision detection)
├── FoodGrid (food sources; optional)
├── Odor layers (GaussianValueLayer / DiffusionValueLayer; optional)
├── Thermoscape (thermal gradients; optional)
└── Windscape (wind fields; optional)
```

---

## Communication Patterns

### 1. Sensor → Environment (Pull)

**Pattern**: Sensors **pull** data from environment on each timestep.

```python
# model/modules/brain.py (Brain.sense_odors)
odor_input = {id: layer.get_value(pos) for id, layer in self.agent.model.odor_layers.items()}
A_olf = self.olfactor.step(odor_input)
```

### 2. Brain → Locomotor (Command)

**Pattern**: Brain **commands** locomotor modules.

```python
# model/modules/brain.py (DefaultBrain.step)
def step(self, pos, on_food=False, **kwargs):
    self.sense(pos=pos, reward=on_food)
    return self.locomotor.step(A_in=self.A_in, on_food=on_food, **kwargs)
```

### 3. Locomotor → Agent (Update)

**Pattern**: Locomotor modules **update** agent state.

```python
# model/agents/_larva.py (LarvaMotile.step)
lin, ang, self.feeder_motion = self.brain.step(
    pos, length=self.length, on_food=self.on_food
)
self.prepare_motion(lin=lin, ang=ang)  # applies lin/ang to body pose
```

### 4. Agent → Environment (Modify)

**Pattern**: Agents **modify** environment state (feeding, collisions).

```python
# model/agents/_larva.py (LarvaMotile.feed)
if motion and source is not None:
    a_max = self.brain.locomotor.feeder.V_bite * self.V * 1000
    grid = self.model.food_grid
    V = -grid.add_cell_value(source, -a_max) if grid else source.subtract_amount(a_max)
```

### 5. SimEngine → Agent (Broadcast)

**Pattern**: SimEngine **broadcasts** timestep update to all agents.

```python
# sim/single_run.py (ExpRun.step)
self.agents.step()  # AgentPy: steps all agents synchronously
```

## Extending the Platform

### Adding a New Sensor

**Steps**:

1. Create a subclass of an existing sensor (e.g., `Olfactor`, `Thermosensor`) in your own module.
2. Register it as a new `mode` in `BrainModuleDB.BrainModuleModes`.
3. Select that mode in a model configuration (e.g., `mm.brain.olfactor.mode = "custom"`).
4. Run a simulation using that model configuration.

**Example**:

```python
from larvaworld.lib import reg
from larvaworld.lib.model.modules.module_modes import BrainModuleDB
from larvaworld.lib.model.modules.sensor import Olfactor

class MyOlfactor(Olfactor):
    def update(self):
        super().update()
        self.output *= 0.5  # example custom processing

# Register a new sensor mode (global for the current Python session)
BrainModuleDB.BrainModuleModes["olfactor"]["custom"] = MyOlfactor

# Select the new mode in a model configuration
mm = reg.conf.Model.getID("explorer").get_copy()
mm.brain.olfactor.mode = "custom"
```

### Adding a New Behavioral Module

**Steps**:

1. Create a subclass of the appropriate module type (e.g., `Crawler`, `Turner`, `Feeder`).
2. Register it as a new `mode` in `BrainModuleDB.BrainModuleModes`.
3. Select that mode in a model configuration (e.g., `mm.brain.turner.mode = "custom"`).
4. Run a simulation using that model configuration.

**Example**:

```python
from larvaworld.lib import reg
from larvaworld.lib.model.modules.module_modes import BrainModuleDB
from larvaworld.lib.model.modules.turner import ConstantTurner

class MyTurner(ConstantTurner):
    def update(self):
        super().update()
        self.output *= 0.0  # example: disable turning

BrainModuleDB.BrainModuleModes["turner"]["custom"] = MyTurner

mm = reg.conf.Model.getID("explorer").get_copy()
mm.brain.turner.mode = "custom"
```

For detailed tutorial, see {doc}`../tutorials/5_extending_larvaworld/custom_brain_modules`.

---

## Related Documentation

- {doc}`architecture_overview` - Platform layers
- {doc}`../agents_environments/larva_agent_architecture` - Agent architecture
- {doc}`../agents_environments/brain_module_architecture` - Brain module details
- {doc}`simulation_modes` - Simulation execution modes
- {doc}`../tutorials/5_extending_larvaworld/custom_brain_modules` - Adding custom modules
