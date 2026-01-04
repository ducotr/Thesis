#!/usr/bin/env bash
#
# format_python.sh
#
# Format all Python files in this directory and all subdirectories.
# Edit the settings in the CONFIG section below.

set -euo pipefail

#########################
# CONFIG
#########################

# Max line length for formatters
LINE_LENGTH=88

# Whether to run isort (true/false)
RUN_ISORT=true

# Whether to run black (true/false)
RUN_BLACK=true

# isort profile: "black" is a good default
ISORT_PROFILE="black"

# Show diff instead of modifying files (true/false)
DRY_RUN=false

#########################
# HELPER FUNCTIONS
#########################

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

#########################
# CHECK TOOLS
#########################

if [ "$RUN_ISORT" = true ] && ! command_exists isort; then
    echo "Error: isort is not installed (pip install isort)" >&2
    exit 1
fi

if [ "$RUN_BLACK" = true ] && ! command_exists black; then
    echo "Error: black is not installed (pip install black)" >&2
    exit 1
fi

#########################
# COLLECT FILES
#########################

# Find all .py files under current dir
PY_FILES=()
while IFS= read -r -d '' file; do
    PY_FILES+=("$file")
done < <(find . -type f -name "*.py" -print0)

if [ ${#PY_FILES[@]} -eq 0 ]; then
    echo "No Python files found."
    exit 0
fi

#########################
# RUN FORMATTERS
#########################

echo "Formatting ${#PY_FILES[@]} Python files..."
echo "  Line length: $LINE_LENGTH"
echo "  Run isort:   $RUN_ISORT"
echo "  Run black:   $RUN_BLACK"
echo "  Dry run:     $DRY_RUN"
echo

if [ "$RUN_ISORT" = true ]; then
    echo "Running isort..."
    if [ "$DRY_RUN" = true ]; then
        isort --profile "$ISORT_PROFILE" --line-length "$LINE_LENGTH" --diff "${PY_FILES[@]}"
    else
        isort --profile "$ISORT_PROFILE" --line-length "$LINE_LENGTH" "${PY_FILES[@]}"
    fi
    echo
fi

if [ "$RUN_BLACK" = true ]; then
    echo "Running black..."
    if [ "$DRY_RUN" = true ]; then
        black --line-length "$LINE_LENGTH" --diff "${PY_FILES[@]}"
    else
        black --line-length "$LINE_LENGTH" "${PY_FILES[@]}"
    fi
    echo
fi

echo "Done."
