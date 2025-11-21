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
3. **SimEngine Setup**: `ExpRun.__init__()` initializes
4. **Environment Creation**: `Env` object created with arena, food, odorscape
5. **Agent Creation**: `LarvaSim` objects created (N=20)
6. **Module Initialization**: Each larva initializes Brain, Locomotor, Sensors

**Code Path**:

```python
# cli/main.py
main() → SimModeParser.parse_args()

# sim/single_run.py
ExpRun.__init__() → self.build_env() → self.build_agents()

# model/envs/env.py
Env.__init__() → create arena, food_grid, odorscape

# model/agents/larva_robot.py
LarvaSim.__init__() → Brain(), Locomotor(), Sensors()
```

---

### Phase 2: Simulation Loop

The core execution loop runs for `Nsteps` timesteps (typically `duration * 600` for 0.1s timestep).

#### 2.1 Timestep Update

**SimEngine → LarvaAgent**: "Update to step t"

**Code**:

```python
# sim/base_run.py (BaseRun.simulate)
for step in range(self.Nsteps):
    self.model.step()  # Agentpy ABM step
```

#### 2.2 Sensory Input

**LarvaAgent → Sensors → Environment**

**Sequence**:
1. Larva queries sensors
2. Sensors read environment state (odor, food, obstacles)
3. Environment returns sensory data
4. Sensors process and return to larva

**Code**:

```python
# model/agents/_larva.py (LarvaMotile.sense)
def sense(self):
    # Olfactory sensing
    if self.olfactor is not None:
        self.olfactor.step()  # Read odorscape

    # Touch sensing
    if self.toucher is not None:
        self.toucher.step()  # Detect contacts

    # Feeding sensing
    if self.feeder is not None:
        self.feeder.sense()  # Detect food
```

**Olfactory Example**:

```python
# model/modules/sensors/olfactor.py
def step(self):
    # Get odor concentration at larva position
    odor_value = self.model.odorscape.get_value(self.pos)
    
    # Apply sensory processing (Weber-Fechner law)
    self.sensed_odor = self.process_olfaction(odor_value)
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
def step(self):
    # 1. Collect sensory input
    olf = self.olfactor.sensed_odor if self.olfactor else 0
    touch = self.toucher.contacts if self.toucher else []
    
    # 2. Update memory/learning
    if self.memory is not None:
        self.memory.step(reward=self.feeder.amount_eaten)
    
    # 3. Generate locomotor commands
    self.locomotor.compute()  # Activates crawler, turner, feeder
```

**NengoBrain** (neural network alternative):

```python
# model/modules/brain.py (NengoBrain.step)
def step(self):
    # Run Nengo neural simulation
    self.sim.run_steps(1)
    
    # Extract motor commands from output neurons
    self.locomotor.compute()
```

#### 2.4 Motor Command Generation

**Brain → Crawler/Turner/Feeder**: "Generate actions"

**Locomotor Coordination**:

```python
# model/modules/locomotor.py (Locomotor.compute)
def compute(self):
    # Crawling
    if self.crawler is not None:
        self.crawler.step()  # Generate stride
    
    # Turning
    if self.turner is not None:
        self.turner.step()  # Compute angular velocity
    
    # Feeding
    if self.feeder is not None:
        self.feeder.step()  # Attempt to feed
    
    # Interference (crawl-turn coupling)
    if self.interference is not None:
        self.interference.step()  # Modulate crawler based on turner
```

**Crawler Step**:

```python
# model/modules/locomotor.py (Crawler.step)
def step(self):
    # Check if stride should initiate
    if self.ready_to_stride():
        self.initiate_stride()
    
    # Update stride phase
    if self.striding:
        self.update_stride_phase()
        
        # Compute forward velocity
        self.fov = self.compute_velocity()
```

**Turner Step**:

```python
# model/modules/locomotor.py (Turner.step)
def step(self):
    # Compute angular velocity based on:
    # - Sensory input (odor gradient)
    # - Interference from crawler
    # - Random exploration
    
    self.ang_v = self.compute_angular_velocity()
```

#### 2.5 Action Execution

**LarvaAgent → Environment**: "Execute actions"

**Physics Update**:

```python
# model/agents/_larva.py (LarvaMotile.move)
def move(self):
    # Get locomotor output
    fov = self.locomotor.crawler.fov  # Forward velocity
    ang_v = self.locomotor.turner.ang_v  # Angular velocity
    
    # Update orientation
    self.orientation += ang_v * self.dt
    
    # Update position
    dx = fov * np.cos(self.orientation) * self.dt
    dy = fov * np.sin(self.orientation) * self.dt
    self.pos[0] += dx
    self.pos[1] += dy
    
    # Check collisions
    self.model.space.check_collisions(self)
```

**Feeding Action**:

```python
# model/modules/locomotor.py (Feeder.step)
def step(self):
    if self.on_food():
        # Consume food
        amount = self.intake_rate * self.dt
        self.model.food_grid.consume(self.pos, amount)
        
        # Update DEB model
        self.deb.feed(amount)
```

**DEB Update** (energetics):

```python
# model/modules/energetics.py (DEB.step)
def step(self):
    # Dynamic Energy Budget model
    # Update reserves, structure, maturity
    self.update_reserves()
    self.update_structure()
    self.update_maturity()
    
    # Compute body mass
    self.body_mass = self.compute_mass()
```

#### 2.6 State Recording

**SimEngine**: Record data

**Data Collection**:

```python
# sim/base_run.py (BaseRun.store_data)
def store_data(self):
    for larva in self.model.agents:
        # Record pose
        self.data["position"].append(larva.pos)
        self.data["orientation"].append(larva.orientation)
        
        # Record velocities
        self.data["linear_velocity"].append(larva.locomotor.crawler.fov)
        self.data["angular_velocity"].append(larva.locomotor.turner.ang_v)
        
        # Record brain state
        if larva.brain.memory:
            self.data["gain"].append(larva.brain.memory.gain)
        
        # Record energetics
        if larva.deb:
            self.data["body_mass"].append(larva.deb.body_mass)
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
# sim/base_run.py (BaseRun.simulate)
def simulate(self):
    # Run simulation loop
    self.model.setup()
    for step in range(self.Nsteps):
        self.model.step()
        self.store_data()
    
    # Finalize
    self.finalize()  # Convert to datasets
    self.store()     # Save HDF5
    self.plot()      # Generate plots (if requested)
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
│   │       └── Interference
│   ├── DEB (energetics)
│   └── Body (morphology)
└── BaseController (parent)
    └── Visualization methods
```

### Environment Dependencies

```
Env
├── Arena (geometry)
├── FoodGrid (food sources)
├── Odorscape (odor gradients)
├── Thermoscape (thermal gradients, optional)
├── Windscape (wind fields, optional)
└── Space (collision detection)
```

---

## Communication Patterns

### 1. Sensor → Environment (Pull)

**Pattern**: Sensors **pull** data from environment on each timestep.

```python
# Sensor queries environment
odor_value = self.model.odorscape.get_value(self.pos)
```

### 2. Brain → Locomotor (Command)

**Pattern**: Brain **commands** locomotor modules.

```python
# Brain activates locomotor
self.locomotor.compute()  # Brain triggers locomotor computation
```

### 3. Locomotor → Agent (Update)

**Pattern**: Locomotor modules **update** agent state.

```python
# Crawler updates forward velocity
self.locomotor.crawler.fov = computed_velocity

# Turner updates angular velocity
self.locomotor.turner.ang_v = computed_angular_velocity
```

### 4. Agent → Environment (Modify)

**Pattern**: Agents **modify** environment state (feeding, collisions).

```python
# Agent consumes food
self.model.food_grid.consume(self.pos, amount)
```

### 5. SimEngine → Agent (Broadcast)

**Pattern**: SimEngine **broadcasts** timestep update to all agents.

```python
# Agentpy ABM loop
for agent in self.model.agents:
    agent.step()  # All agents step synchronously
```

## Extending the Platform

### Adding a New Sensor

**Steps**:
1. Create subclass of `Sensor` in `/lib/model/modules/sensor.py`
2. Implement `update()` method to process sensory input
3. Register sensor in `Brain` initialization
4. Add sensory data to `Brain.step()` integration

**Example**:

```python
# model/modules/sensor.py
from larvaworld.lib.model.modules.sensor import Sensor

class MySensor(Sensor):
    def update(self):
        # Query environment
        value = self.brain.agent.model.environment.get_my_stimulus(self.brain.agent.pos)
        
        # Process sensory input
        input_dict = {'my_stimulus': value}
        self.step(input=input_dict)
        
        # Output is automatically available via self.output
```

### Adding a New Behavioral Module

**Steps**:
1. Create subclass of `Effector` in `/lib/model/modules/`
2. Implement `update()` method
3. Register module in `Locomotor` or `Brain`
4. Add module output to agent actions

For detailed tutorial, see {doc}`../tutorials/custom_module`.

---

## Related Documentation

- {doc}`architecture_overview` - Platform layers
- {doc}`../agents_environments/larva_agent_architecture` - Agent architecture
- {doc}`../agents_environments/brain_module_architecture` - Brain module details
- {doc}`simulation_modes` - Simulation execution modes
- {doc}`../tutorials/custom_module` - Adding custom modules
