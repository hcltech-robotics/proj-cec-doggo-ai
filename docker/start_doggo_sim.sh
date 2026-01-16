#!/bin/bash


ENVIRONMENT="default"
QUADRUPED="spot"
JOY_ENABLE_BUTTON=5
JOY_LINEAR_AXIS=4
JOY_ANGULAR_AXIS=3

# --------------------
# Usage
# --------------------
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --environment VALUE        Environment name (default: $ENVIRONMENT)
  --quadruped VALUE          Quadruped model (default: $QUADRUPED)
  --joy-enable-button VALUE  Joystick enable button index (default: $JOY_ENABLE_BUTTON)
  --joy-linear-axis VALUE    Joystick linear axis index (default: $JOY_LINEAR_AXIS)
  --joy-angular-axis VALUE   Joystick angular axis index (default: $JOY_ANGULAR_AXIS)
  -h, --help                 Show this help and exit

Examples:
  $0
  $0 --environment real --quadruped a1
  $0 --joy-enable-button 6 --joy-linear-axis 3 --joy-angular-axis 2
  $0 --environment=sim --quadruped=go2
EOF
}

# --------------------
# Argument parsing
# --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --environment=*)
      ENVIRONMENT="${1#*=}"
      shift
      ;;
    --quadruped)
      QUADRUPED="$2"
      shift 2
      ;;
    --quadruped=*)
      QUADRUPED="${1#*=}"
      shift
      ;;
    --joy-enable-button)
      JOY_ENABLE_BUTTON="$2"
      shift 2
      ;;
    --joy-enable-button=*)
      JOY_ENABLE_BUTTON="${1#*=}"
      shift
      ;;
    --joy-linear-axis)
      JOY_LINEAR_AXIS="$2"
      shift 2
      ;;
    --joy-linear-axis=*)
      JOY_LINEAR_AXIS="${1#*=}"
      shift
      ;;
    --joy-angular-axis)
      JOY_ANGULAR_AXIS="$2"
      shift 2
      ;;
    --joy-angular-axis=*)
      JOY_ANGULAR_AXIS="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo
      usage
      exit 1
      ;;
  esac
done


mkdir -p isaac-sim/cache/main/ov
mkdir -p isaac-sim/cache/main/warp
mkdir -p isaac-sim/cache/computecache
#mkdir -p isaac-sim/config
mkdir -p isaac-sim/data/documents
mkdir -p isaac-sim/data/Kit
mkdir -p isaac-sim/logs
mkdir -p isaac-sim/pkg

if [[ "$(stat -c '%u:%g' isaac-sim)" != "1234:1234" ]]; then
  sudo chown -R 1234:1234 isaac-sim
fi

export ENVIRONMENT QUADRUPED JOY_ENABLE_BUTTON JOY_LINEAR_AXIS JOY_ANGULAR_AXIS

docker compose up
