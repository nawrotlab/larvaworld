# LarvaSim Architecture (Simplified)

This diagram shows a **simplified view** of the `LarvaSim` class architecture, focusing on the main functional groups rather than exact code structure.

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'primaryBorderColor': '#000000', 'lineColor': '#000000', 'secondaryColor': '#f8f9fa', 'tertiaryColor': '#e9ecef'}}}%%
graph TB
    subgraph LARVASIM ["LarvaSim"]
        AGENT[LarvaSim<br/>Simulated Larva]:::larva
        
        subgraph PHYSICAL ["Physical Body"]
            BODY[Segmented Body<br/>11-13 segments]:::body
            CONTOUR[Body Contour<br/>Collision detection]:::body
        end
        
        subgraph BRAIN_SYS ["Brain System"]
            BRAIN[Brain<br/>DefaultBrain or NengoBrain]:::brain
            SENSORS[Sensors<br/>Olfactor, Toucher<br/>Windsensor, Thermosensor]:::sensors
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

    %% Color definitions
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

## Description

This simplified view groups `LarvaSim` components by **functional purpose** rather than strict code hierarchy.

### Color Legend

- **⚫ Dark Blue**: LarvaSim agent
- **🔴 Red**: Physical body
- **🟠 Orange**: Sensory systems
- **🔵 Blue**: Brain module
- **🟣 Purple**: Memory (learning)
- **🟢 Green**: Behavioral modules
- **🟡 Yellow**: Energy/metabolism
- **🔷 Cyan**: Locomotor coordination
- **🔴 Pink**: Physics control

### Functional Groups

#### Physical Body (Red)
- **Segmented Body**: 11-13 segments with realistic biomechanics
- **Body Contour**: Shape representation for collision detection

#### Brain System (Blue/Orange/Purple)
- **Brain**: Central control (DefaultBrain or NengoBrain)
- **Sensors**: Environmental perception (olfaction, touch, temperature, wind)
- **Memory**: Optional learning system (RL or model-based)

#### Locomotor System (Cyan/Green)
- **Locomotor**: Coordinates all motor modules
- **Crawler**: Generates forward peristaltic motion
- **Turner**: Controls body bending for direction changes
- **Feeder**: Manages head-sweeping feeding behavior
- **Intermitter**: Switches between run/pause/turn states

#### Energy System (Yellow)
- **DEB Model**: Dynamic Energy Budget for metabolism
- **Gut**: Digestive system and food processing
- **Reserves**: Energy storage (E, E_R, E_H)

#### Physics Control (Pink)
- **Physics**: Body mechanics (damping, torsional spring)
- **Motion**: Motion control modes (velocity/force/torque)

### Information Flow

1. **Perception**: Sensors detect environmental stimuli
2. **Processing**: Brain integrates sensory information
3. **Memory**: Optional learning modulates responses
4. **Motor Generation**: Locomotor coordinates behavioral modules
5. **Physics Application**: BaseController applies realistic motion physics
6. **Execution**: Final motion applied to segmented body
7. **Metabolism**: DEB regulates energy and growth

### Key Differences from Detailed View

This simplified view:
- Groups components by function rather than inheritance
- Hides internal brain structure (modalities dict)
- Shows high-level data flow instead of method calls
- Emphasizes system integration over implementation details

For the **detailed, code-accurate architecture**, see `13_a_larvasim_architecture_verified.md`.

### Biological Fidelity

- **Segmented body**: Matches real larva anatomy
- **Sensory integration**: Realistic multi-modal perception
- **Behavioral modules**: Based on neural oscillators
- **Energy constraints**: Metabolic limits on behavior
- **Physics realism**: Damping, inertia, body mechanics

### Design Principles

1. **Modularity**: Independent, replaceable components
2. **Biological Realism**: Structure mirrors real larvae
3. **Multiple Inheritance**: Combines behavior (LarvaMotile) + physics (BaseController)
4. **Hierarchical Control**: Brain → Locomotor → Modules → Body
5. **Energy Constraints**: DEB limits behavioral capacity
