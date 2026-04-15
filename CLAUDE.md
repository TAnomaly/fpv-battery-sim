# Build an FPV Drone Battery Performance Simulator in Python

## Project Overview
Create a comprehensive Python-based simulation tool that models and predicts battery performance, flight time, and power consumption for FPV drones. The simulator should allow users to input drone specifications and receive detailed analysis of battery behavior under various conditions.

---

## Core Features Required

### 1. Battery Model (LiPo)
Implement realistic LiPo battery physics including:
- **Voltage characteristics**: Nominal voltage per cell (3.7V), fully charged (4.2V), cutoff voltage (3.0V-3.5V)
- **Capacity degradation under load** using Peukert's Law approximation
- **Internal resistance effects** causing voltage sag under high current draw
- **Temperature compensation** for performance variations
- **Discharge curve modeling** showing realistic voltage drop over time

### 2. Motor & ESC Parameters
Model the power consumption of:
- **Motor KV rating** (RPM per volt)
- **Propeller size and pitch** affecting load
- **ESC efficiency** (typically 90-95%)
- **Current draw calculation** based on throttle input

### 3. Flight Dynamics Model
Include basic flight scenarios:
- **Hover mode**: ~60% throttle average
- **Aggressive flying**: ~80-100% throttle bursts with recovery periods
- **Cruising**: ~50-70% throttle sustained
- **Custom throttle profiles** (user-defined time-series)

### 4. Simulation Engine
Build a real-time simulation that:
- Runs in discrete time steps (configurable, default 0.1s)
- Calculates instantaneous current draw based on throttle and battery voltage
- Updates remaining capacity using Coulomb counting with efficiency factors
- Tracks voltage sag due to internal resistance
- Detects low-voltage cutoff conditions

### 5. Output & Visualization
Generate comprehensive reports including:
- **Flight time prediction** (minutes:seconds)
- **Voltage vs. Time graph** showing discharge curve
- **Current draw vs. Time graph**
- **Remaining capacity percentage over time**
- **Average power consumption** in watts
- **Peak current events** and their duration

---

## Technical Requirements

### Libraries to Use
```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
