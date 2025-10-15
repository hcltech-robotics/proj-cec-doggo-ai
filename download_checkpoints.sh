#!/usr/bin/env bash
set -euo pipefail

# ---- Color + styling (fallback to no color on dumb terminals) ----
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
  BOLD="$(tput bold)"; RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; MAGENTA="$(tput setaf 5)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

info () { printf "%b➤%b %b%s%b\n" "$BLUE" "$RESET" "$BOLD" "$*" "$RESET"; }
ok   () { printf "%b✔%b %s\n" "$GREEN" "$RESET" "$*"; }
warn () { printf "%b⚠%b %s\n" "$YELLOW" "$RESET" "$*"; }
err  () { printf "%b✖%b %s\n" "$RED" "$RESET" "$*" >&2; }

trap 'err "Script failed at: ${BASH_COMMAND}"' ERR

info "Starting model download and setup..."

# Create the checkpoints directory if it doesn't exist
info "Creating 'checkpoints' directory if needed..."
mkdir -p checkpoints
ok "OK"

# Download helper (wget or curl)
download () {
  local out="$1"; shift
  local url="$1"; shift
  info "Downloading $(basename "$out")..."
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --progress-bar -o "$out" "$url"
  else
    err "Neither wget nor curl found."
    return 1
  fi
  ok "Saved to $out"
}

download "checkpoints/gauge_detect.pth" \
  "https://huggingface.co/hcltech-robotics/gauge_detect/resolve/main/gauge_detect3_hcl.pt.state_dict.pth"

download "checkpoints/gauge_reader.pt" \
  "https://huggingface.co/hcltech-robotics/gauge_reader/resolve/main/gauge_net3v4_hcl.pt"

download "checkpoints/hey-artie.tflite" \
  "https://huggingface.co/targabor/hey-artie-precise-lite/resolve/main/hey-artie.tflite"

# Ensure the target directories exist
info "Ensuring target directory 'ros.ws/src/gauge_net/gauge_net/models' exists..."
mkdir -p ros.ws/src/gauge_net/gauge_net/models
ok "OK"

info "Ensuring target directory 'ros.ws/src/artie_audio/models' exists..."
mkdir -p ros.ws/src/artie_audio/models
ok "OK"

# Copy the models to the ROS workspace
info "Copying gauge_detect.pth to ROS workspace..."
cp -f checkpoints/gauge_detect.pth ros.ws/src/gauge_net/gauge_net/models/
ok "Copied"

info "Copying gauge_reader.pt to ROS workspace..."
cp -f checkpoints/gauge_reader.pt ros.ws/src/gauge_net/gauge_net/models/
ok "Copied"

info "Copying hey-artie.tflite to ROS workspace..."
cp -f checkpoints/hey-artie.tflite ros.ws/src/artie_audio/models
ok "Copied"

printf "\n%b✔%b %bModel setup completed successfully!%b\n" "$GREEN" "$RESET" "$BOLD" "$RESET"
