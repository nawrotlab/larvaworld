# Figure 1: Larvaworld Architecture

## 📊 Image

**File:** `fig1_architecture.pdf` (vector format for high-quality printing and zooming)

> **Note**: This is a PDF file for maximum quality. In Sphinx/ReadTheDocs, PDFs can be embedded directly. For markdown viewers, convert to PNG if needed.

## Description

**A schematic of the main components and functionalities of Larvaworld.**

This figure provides a comprehensive overview of the Larvaworld platform architecture, showing:

### Main Components:

1. **User Interfaces**

   - Command Line Interface (CLI)
   - Web-based applications
   - GUI (deprecated/legacy)

2. **Core Modules**

   - Simulation engine
   - Agent modeling (larvae)
   - Environment modeling
   - Data processing & analysis

3. **Functionality**
   - Experiment configuration
   - Behavioral simulation
   - Data import/export
   - Visualization
   - Model evaluation

### Purpose

This architecture diagram is intended to:

- ✅ **Central placement** in documentation (developer request)
- ✅ Give newcomers a high-level understanding of the platform
- ✅ Show how different components interact
- ✅ Illustrate the modular design philosophy

---

## Usage in ReadTheDocs

**Placement**: Home page / Getting Started section (central/prominent position)

**Context**: Place at the beginning as an overview figure before diving into specific components.

```rst
Platform Architecture
~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/images/fig1_architecture.png
   :alt: Larvaworld Architecture Overview
   :align: center
   :width: 100%

   **Figure 1**: Larvaworld architecture. A schematic of the main components
   and functionalities of the platform. The architecture demonstrates the
   modular design with clear separation between user interfaces, core engine,
   modeling layers, and data processing pipelines.

Larvaworld is built around a modular architecture that separates concerns and
enables flexible extension. The platform can be accessed through multiple
interfaces (CLI, web apps, Python library) while maintaining a unified core
engine for simulation and analysis.
```

---

## Source

- **Paper**: Larvaworld PLOS Comp.Biology Software_v05
- **File**: `/images/architecture.png`
- **Caption** (LaTeX line 320): "Larvaworld architecture. A schematic of the main components and functionalities of Larvaworld."
- **Label**: `fig:architecture`
