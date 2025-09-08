#!/usr/bin/env bash
set -euo pipefail

# Defaults
NAME=""
PYVER=""

# Parse args
for arg in "$@"; do
  case $arg in
    --name=*)
      NAME="${arg#*=}"
      shift
      ;;
    --python=*)
      PYVER="${arg#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 --name=<env_name> --python=<version>"
      exit 1
      ;;
  esac
done

# Validate
if [[ -z "$NAME" || -z "$PYVER" ]]; then
  echo "Both --name and --python are required."
  echo "Usage: $0 --name=<env_name> --python=<version>"
  exit 1
fi

# Create environment
echo "Creating uv venv: $NAME with Python $PYVER"
uv venv "$NAME" --python "$PYVER"
