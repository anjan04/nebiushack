#!/usr/bin/env bash
# =============================================================================
# AutoRobot — Nebius GPU VM Setup Script
# =============================================================================
#
# Automates the full environment setup for the AutoRobot project on a Nebius
# GPU VM (covers tasks T0.1 through T0.4 from docs/TASKS.md):
#
#   T0.1  VM provisioning assumed done (script validates GPU is available)
#   T0.2  Install Isaac Gym Preview 4 (Miniconda + conda env + pip install)
#   T0.3  Clone Eureka repo, install isaacgymenvs and rl_games
#   T0.4  Verify Ant training (headless, 200 iterations)
#   (+)   Clone nebiushack repo, install project Python deps
#
# Usage:
#   chmod +x scripts/setup_vm.sh
#   ./scripts/setup_vm.sh
#
# Prerequisites:
#   - Ubuntu 20.04 or 22.04 with NVIDIA GPU drivers already installed
#   - Internet access (downloads Miniconda, clones repos)
#   - Isaac Gym tarball at ~/IsaacGym_Preview_4_Package.tar.gz
#     (download from https://developer.nvidia.com/isaac-gym)
#
# The script is idempotent — safe to re-run. Already-completed steps are
# skipped automatically. All output is logged to ~/autorobot_setup.log.
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOGFILE="$HOME/autorobot_setup.log"
MINICONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="autorobot"
CONDA_PYTHON_VERSION="3.8"
ISAAC_GYM_TARBALL="$HOME/IsaacGym_Preview_4_Package.tar.gz"
ISAAC_GYM_DIR="$HOME/isaacgym"
EUREKA_DIR="$HOME/Eureka"
EUREKA_REPO="https://github.com/eureka-research/Eureka.git"
NEBIUSHACK_DIR="$HOME/nebiushack"
NEBIUSHACK_REPO="https://github.com/anjan04/nebiushack.git"
ANT_VERIFY_ITERATIONS=200

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_success() { echo -e "${GREEN}[OK]${NC}    $*" | tee -a "$LOGFILE"; }
log_fail()    { echo -e "${RED}[FAIL]${NC}  $*" | tee -a "$LOGFILE"; }
log_skip()    { echo -e "${YELLOW}[SKIP]${NC}  $*" | tee -a "$LOGFILE"; }
log_info()    { echo -e "${CYAN}[INFO]${NC}  $*" | tee -a "$LOGFILE"; }
log_header()  { echo -e "\n${BOLD}=== $* ===${NC}" | tee -a "$LOGFILE"; }

# ---------------------------------------------------------------------------
# Trap — cleanup on failure
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_fail "Setup did not complete successfully (exit code $exit_code)."
        log_info "Review the log at $LOGFILE for details."
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Summary tracking
# ---------------------------------------------------------------------------
declare -a SUMMARY_LINES=()
summary_add() {
    SUMMARY_LINES+=("$1")
}

VERIFICATION_RESULT="NOT RUN"

# ---------------------------------------------------------------------------
# Initialize log
# ---------------------------------------------------------------------------
echo "" >> "$LOGFILE"
echo "============================================================" >> "$LOGFILE"
echo "AutoRobot setup started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOGFILE"
echo "============================================================" >> "$LOGFILE"

log_header "AutoRobot GPU VM Setup"
log_info "Log file: $LOGFILE"
log_info "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ===================================================================
# Step 0: Validate GPU availability (T0.1 check)
# ===================================================================
log_header "Step 0: GPU Validation"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        log_success "GPU detected: $GPU_INFO"
        summary_add "GPU: $GPU_INFO"
    else
        log_fail "nvidia-smi found but no GPU reported. Check NVIDIA drivers."
        exit 1
    fi
else
    log_fail "nvidia-smi not found. NVIDIA drivers must be installed before running this script."
    exit 1
fi

# ===================================================================
# Step 1: Install Miniconda (if not present)
# ===================================================================
log_header "Step 1: Miniconda"

if [ -d "$MINICONDA_DIR" ] && [ -x "$MINICONDA_DIR/bin/conda" ]; then
    log_skip "Miniconda already installed at $MINICONDA_DIR"
    summary_add "Miniconda: already installed"
else
    log_info "Downloading Miniconda..."
    MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        wget -q -O "$MINICONDA_INSTALLER" \
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
            2>>"$LOGFILE"
    fi
    log_info "Installing Miniconda to $MINICONDA_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR" >> "$LOGFILE" 2>&1
    rm -f "$MINICONDA_INSTALLER"
    log_success "Miniconda installed"
    summary_add "Miniconda: freshly installed"
fi

# Activate conda for this shell session
eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"

# Ensure conda-forge channel (some deps need it)
conda config --add channels conda-forge --force >> "$LOGFILE" 2>&1 || true

# ===================================================================
# Step 2: Create / activate conda environment
# ===================================================================
log_header "Step 2: Conda Environment '$CONDA_ENV_NAME'"

if conda env list 2>/dev/null | grep -qw "$CONDA_ENV_NAME"; then
    log_skip "Conda env '$CONDA_ENV_NAME' already exists"
    summary_add "Conda env: already existed (python $CONDA_PYTHON_VERSION)"
else
    log_info "Creating conda env '$CONDA_ENV_NAME' with Python $CONDA_PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$CONDA_PYTHON_VERSION" -y >> "$LOGFILE" 2>&1
    log_success "Conda env '$CONDA_ENV_NAME' created"
    summary_add "Conda env: created (python $CONDA_PYTHON_VERSION)"
fi

conda activate "$CONDA_ENV_NAME"
log_info "Active Python: $(python --version 2>&1) at $(which python)"

# ===================================================================
# Step 3: Install Isaac Gym Preview 4 (T0.2)
# ===================================================================
log_header "Step 3: Isaac Gym Preview 4"

ISAAC_GYM_INSTALLED=false
if python -c "import isaacgym" 2>/dev/null; then
    log_skip "Isaac Gym already importable in current env"
    summary_add "Isaac Gym: already installed"
    ISAAC_GYM_INSTALLED=true
fi

if [ "$ISAAC_GYM_INSTALLED" = false ]; then
    # Check for tarball
    if [ ! -f "$ISAAC_GYM_TARBALL" ]; then
        log_fail "Isaac Gym tarball not found at $ISAAC_GYM_TARBALL"
        echo ""
        echo -e "${YELLOW}Please download Isaac Gym Preview 4 from:${NC}"
        echo "  https://developer.nvidia.com/isaac-gym"
        echo ""
        echo "Then place the tarball at:"
        echo "  $ISAAC_GYM_TARBALL"
        echo ""
        echo "Re-run this script after the tarball is in place."
        exit 1
    fi

    # Extract if not already extracted
    if [ ! -d "$ISAAC_GYM_DIR" ]; then
        log_info "Extracting Isaac Gym tarball..."
        tar -xzf "$ISAAC_GYM_TARBALL" -C "$HOME" >> "$LOGFILE" 2>&1
        log_success "Extracted to $ISAAC_GYM_DIR"
    else
        log_skip "Isaac Gym directory already exists at $ISAAC_GYM_DIR"
    fi

    # Install via pip in editable mode
    log_info "Installing Isaac Gym Python package..."
    (cd "$ISAAC_GYM_DIR/python" && pip install -e . >> "$LOGFILE" 2>&1)

    # Verify import
    if python -c "import isaacgym; print('Isaac Gym OK')" >> "$LOGFILE" 2>&1; then
        log_success "Isaac Gym installed and importable"
        summary_add "Isaac Gym: freshly installed"
    else
        log_fail "Isaac Gym installation failed — could not import isaacgym"
        log_info "Check $LOGFILE for pip output"
        exit 1
    fi
fi

# ===================================================================
# Step 4: Clone Eureka and install isaacgymenvs + rl_games (T0.3)
# ===================================================================
log_header "Step 4: Eureka (isaacgymenvs + rl_games)"

# Clone Eureka repo
if [ -d "$EUREKA_DIR/.git" ]; then
    log_skip "Eureka repo already cloned at $EUREKA_DIR"
    summary_add "Eureka repo: already cloned"
else
    log_info "Cloning Eureka repo..."
    git clone "$EUREKA_REPO" "$EUREKA_DIR" >> "$LOGFILE" 2>&1
    log_success "Eureka repo cloned to $EUREKA_DIR"
    summary_add "Eureka repo: freshly cloned"
fi

# Install isaacgymenvs
ISAACGYMENVS_DIR="$EUREKA_DIR/isaacgymenvs"
if python -c "import isaacgymenvs" 2>/dev/null; then
    log_skip "isaacgymenvs already installed"
    summary_add "isaacgymenvs: already installed"
else
    log_info "Installing isaacgymenvs..."
    (cd "$ISAACGYMENVS_DIR" && pip install -e . >> "$LOGFILE" 2>&1)
    if python -c "import isaacgymenvs" 2>/dev/null; then
        log_success "isaacgymenvs installed"
        summary_add "isaacgymenvs: freshly installed"
    else
        log_fail "isaacgymenvs installation failed"
        exit 1
    fi
fi

# Install rl_games
RL_GAMES_DIR="$EUREKA_DIR/rl_games"
if python -c "import rl_games" 2>/dev/null; then
    log_skip "rl_games already installed"
    summary_add "rl_games: already installed"
else
    log_info "Installing rl_games..."
    (cd "$RL_GAMES_DIR" && pip install -e . >> "$LOGFILE" 2>&1)
    if python -c "import rl_games" 2>/dev/null; then
        log_success "rl_games installed"
        summary_add "rl_games: freshly installed"
    else
        log_fail "rl_games installation failed"
        exit 1
    fi
fi

# ===================================================================
# Step 5: Install project Python dependencies
# ===================================================================
log_header "Step 5: Project Python Dependencies"

PROJECT_DEPS=("openai" "pyyaml" "matplotlib" "gitpython")
DEPS_INSTALLED=()
DEPS_SKIPPED=()

for dep in "${PROJECT_DEPS[@]}"; do
    # Map package names to import names
    case "$dep" in
        pyyaml)     import_name="yaml" ;;
        gitpython)  import_name="git" ;;
        *)          import_name="$dep" ;;
    esac

    if python -c "import $import_name" 2>/dev/null; then
        DEPS_SKIPPED+=("$dep")
    else
        pip install "$dep" >> "$LOGFILE" 2>&1
        if python -c "import $import_name" 2>/dev/null; then
            DEPS_INSTALLED+=("$dep")
        else
            log_fail "Failed to install $dep"
            exit 1
        fi
    fi
done

if [ ${#DEPS_INSTALLED[@]} -gt 0 ]; then
    log_success "Installed: ${DEPS_INSTALLED[*]}"
    summary_add "Python deps installed: ${DEPS_INSTALLED[*]}"
fi
if [ ${#DEPS_SKIPPED[@]} -gt 0 ]; then
    log_skip "Already present: ${DEPS_SKIPPED[*]}"
fi

# ===================================================================
# Step 6: Clone nebiushack repo (if not present)
# ===================================================================
log_header "Step 6: nebiushack Repository"

if [ -d "$NEBIUSHACK_DIR/.git" ]; then
    log_skip "nebiushack repo already present at $NEBIUSHACK_DIR"
    summary_add "nebiushack repo: already cloned"
else
    log_info "Cloning nebiushack repo..."
    git clone "$NEBIUSHACK_REPO" "$NEBIUSHACK_DIR" >> "$LOGFILE" 2>&1
    log_success "nebiushack repo cloned to $NEBIUSHACK_DIR"
    summary_add "nebiushack repo: freshly cloned"
fi

# ===================================================================
# Step 7: Verify Ant training — headless, 200 iterations (T0.4)
# ===================================================================
log_header "Step 7: Isaac Gym Verification (Ant headless, $ANT_VERIFY_ITERATIONS iterations)"

log_info "Running Ant training verification (this may take a few minutes)..."
log_info "Command: python train.py task=Ant headless=True max_iterations=$ANT_VERIFY_ITERATIONS"

VERIFY_LOG="/tmp/autorobot_ant_verify_$$.log"

set +e  # temporarily allow failure so we can capture the result
(
    cd "$ISAACGYMENVS_DIR" && \
    python train.py \
        task=Ant \
        headless=True \
        max_iterations="$ANT_VERIFY_ITERATIONS" \
        graphics_device_id=-1 \
        2>&1
) > "$VERIFY_LOG" 2>&1
VERIFY_EXIT_CODE=$?
set -e

# Append verification output to main log
echo "--- Ant Verification Output ---" >> "$LOGFILE"
cat "$VERIFY_LOG" >> "$LOGFILE"
echo "--- End Ant Verification ---" >> "$LOGFILE"

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    # Try to extract some metrics from the output
    FPS_LINE=$(grep -i "fps" "$VERIFY_LOG" | tail -1 || true)
    REWARD_LINE=$(grep -i "reward" "$VERIFY_LOG" | tail -1 || true)

    log_success "Ant training verification completed successfully (exit code 0)"
    VERIFICATION_RESULT="PASS"
    if [ -n "$FPS_LINE" ]; then
        log_info "Last FPS line: $FPS_LINE"
    fi
    if [ -n "$REWARD_LINE" ]; then
        log_info "Last reward line: $REWARD_LINE"
    fi
    summary_add "Ant verification: PASSED"
else
    log_fail "Ant training verification failed (exit code $VERIFY_EXIT_CODE)"
    log_info "Check $LOGFILE for full output"
    VERIFICATION_RESULT="FAIL (exit code $VERIFY_EXIT_CODE)"
    summary_add "Ant verification: FAILED (exit $VERIFY_EXIT_CODE)"
    # Show last 10 lines of error output for quick debugging
    echo -e "${RED}Last 10 lines of verification output:${NC}"
    tail -10 "$VERIFY_LOG" | tee -a "$LOGFILE"
fi

rm -f "$VERIFY_LOG"

# ===================================================================
# Summary
# ===================================================================
log_header "Setup Summary"

echo -e "${BOLD}Timestamp:${NC}        $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOGFILE"
echo -e "${BOLD}Conda env:${NC}        $CONDA_ENV_NAME (Python $CONDA_PYTHON_VERSION)" | tee -a "$LOGFILE"
echo -e "${BOLD}Isaac Gym dir:${NC}    $ISAAC_GYM_DIR" | tee -a "$LOGFILE"
echo -e "${BOLD}Eureka dir:${NC}       $EUREKA_DIR" | tee -a "$LOGFILE"
echo -e "${BOLD}nebiushack dir:${NC}   $NEBIUSHACK_DIR" | tee -a "$LOGFILE"
echo -e "${BOLD}Verification:${NC}     $VERIFICATION_RESULT" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo -e "${BOLD}Actions taken:${NC}" | tee -a "$LOGFILE"
for line in "${SUMMARY_LINES[@]}"; do
    echo -e "  - $line" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"

if [ "$VERIFICATION_RESULT" = "PASS" ]; then
    echo -e "${GREEN}${BOLD}Setup complete. Environment is ready for AutoRobot.${NC}" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
    echo -e "Next steps:" | tee -a "$LOGFILE"
    echo -e "  1. Set your Nebius API key:  ${CYAN}export NEBIUS_API_KEY=\"your-key-here\"${NC}" | tee -a "$LOGFILE"
    echo -e "  2. Activate the environment: ${CYAN}conda activate $CONDA_ENV_NAME${NC}" | tee -a "$LOGFILE"
    echo -e "  3. Run the agent:            ${CYAN}cd $NEBIUSHACK_DIR && python agent.py${NC}" | tee -a "$LOGFILE"
else
    echo -e "${YELLOW}${BOLD}Setup completed with issues. Review verification results above.${NC}" | tee -a "$LOGFILE"
    echo -e "Full log: $LOGFILE" | tee -a "$LOGFILE"
fi

echo ""
log_info "Full log saved to $LOGFILE"
