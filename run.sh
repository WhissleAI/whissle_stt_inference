#!/usr/bin/env bash
# Backward-compatible wrapper — delegates to setup.sh
exec "$(dirname "$0")/setup.sh" "$@"
