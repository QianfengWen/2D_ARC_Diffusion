# Experiment Naming Issue and Fix

## 🐛 **What Happened**

The experiment contents were placed directly in the date folder (`experiments/2025-09/09/`) instead of in a properly named experiment subfolder like `1_experiment_name/`.

**Structure that occurred:**
```
experiments/2025-09/09/
├── config.yaml        # ❌ Should be in experiment folder
├── data/              # ❌ Should be in experiment folder  
├── logs/              # ❌ Should be in experiment folder
├── models/            # ❌ Should be in experiment folder
└── results/           # ❌ Should be in experiment folder
```

**Structure that should occur:**
```
experiments/2025-09/09/
└── 1_experiment_name/
    ├── config.yaml    # ✅ Properly contained
    ├── data/          # ✅ Properly contained
    ├── logs/          # ✅ Properly contained
    ├── models/        # ✅ Properly contained
    └── results/       # ✅ Properly contained
```

## 🔍 **Root Cause**

1. **Empty experiment name**: When the default experiment name was changed to empty string, if an experiment ran with that empty name, the path template resolved incorrectly:
   ```
   Template: "experiments/{time.month}/{time.day}/{experiment.name}/data"
   With empty name: "experiments/2025-09/09//data" 
   Filesystem simplified: "experiments/2025-09/09/data"
   ```

2. **Missing safety checks**: The system didn't prevent empty experiment names from causing path resolution issues.

## ✅ **Fixes Applied**

### 1. **Added Safety Check in Path Resolution**
```python
def _resolve_paths(self):
    # Safety check: prevent empty experiment names
    if not self.experiment.name or self.experiment.name.strip() == "":
        raise ValueError("Experiment name cannot be empty. Please provide a name via --name argument.")
```

### 2. **Deferred Path Resolution**
```python
def __post_init__(self):
    # Only resolve paths if experiment name is set (will be set by CLI)
    if self.experiment.name:
        self._resolve_paths()
```

### 3. **Cleaned Up Problematic Directory**
Moved the problematic experiment contents to `UNNAMED_experiment/` folder for proper organization.

## 🚀 **How to Use Going Forward**

### ✅ **Correct Usage**
```bash
# This will create: experiments/2025-09/09/1_my_test/
python main.py --name my_test

# This will create: experiments/2025-09/09/2_baseline/
python main.py --name baseline
```

### ❌ **What Will Now Fail Safely**
```bash
# This will fail with clear error message
python main.py  # Missing --name parameter

# Internal errors will also be caught
# "Experiment name cannot be empty. Please provide a name via --name argument."
```

## 🔧 **System Status**
- ✅ **Indexed naming**: Working (1_name, 2_name, etc.)
- ✅ **Dataset management**: Working (central storage + reuse)  
- ✅ **Safety checks**: Added to prevent empty names
- ✅ **Error handling**: Clear error messages for missing names
- ✅ **Path resolution**: Fixed and tested

## 📁 **Current Directory Structure**
```
experiments/2025-09/09/
└── UNNAMED_experiment/     # The problematic experiment (fixed)
    ├── config.yaml
    ├── data/
    ├── logs/
    ├── models/
    └── results/

# Future experiments will be properly named:
# ├── 1_my_next_experiment/
# ├── 2_another_test/
# └── 3_final_run/
```

**All future experiments will be properly indexed and contained within their own folders.** 🎯
