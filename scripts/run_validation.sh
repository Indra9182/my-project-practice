#!/bin/bash
# One-command validation runner
# Usage: ./scripts/run_validation.sh [profile]
# Example: ./scripts/run_validation.sh strict

set -e

PROFILE=${1:-standard}
INPUT="data/gpu_metrics.csv"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo " NVIDIA GPU Validation Framework"
echo " Profile  : $PROFILE"
echo " Timestamp: $TIMESTAMP"
echo "========================================"

# Run validation pipeline
python3 src/main.py \
  --input "$INPUT" \
  --profile "$PROFILE" \
  --output "results/results_${TIMESTAMP}.json" \
  --verbose

# Run analysis if history exists
if [ -f "results/history.csv" ]; then
  echo "Running trend analysis..."
  python src/analyze.py --input results/history.csv
fi

echo "Done. Check results/ for output files."