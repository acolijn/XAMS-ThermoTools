# XAMS-ThermoTools
calculate thermodynamics of the XAMS setup

## Installation

Follow these steps to install the package locally.

### Prerequisites
- Python >= 3.9
- Git (if installing from a local clone)

### 1) Clone (optional, if you already have the folder, skip)
```bash
git clone https://github.com/acolijn/XAMS-ThermoTools.git
cd XAMS-ThermoTools
```

### 2) (Recommended) Create and activate a virtual environment
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install
- Editable install (recommended for development):
```bash
pip install -e .
```

- Regular install (no source editing):
```bash
pip install .
```

### 4) Verify
```bash
python -c "import xams; print(xams.__version__)"
```

If you see a version printed without errors, the installation succeeded.


## Usage
Import the library in Python:
```python
import xams
# Use the available modules and functions
```

## Development: pre-commit for notebooks
This repository uses `pre-commit` with the `nbstripout` hook to automatically strip Jupyter notebook outputs before they are committed. This keeps the Git history clean and avoids large diffs.

### Setup
Install and enable the hooks (run inside your virtual environment):
```bash
pip install pre-commit
pre-commit install
```

The configuration lives in `.pre-commit-config.yaml`. After `pre-commit install`, the hooks run automatically on `git commit`.

### Clean existing files (optional)
To strip outputs in all existing notebooks once:
```bash
pre-commit run --all-files
```

If a commit is blocked by a hook, review the changes that `pre-commit` made, stage them, and commit again.

