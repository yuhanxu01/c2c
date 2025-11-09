#!/bin/bash
# Quick test script to verify a config works before running full experiments

CONFIG=${1:-"configs/cross_no_skip.json"}
EPOCHS=${2:-2}

echo "========================================"
echo "🧪 Quick Test"
echo "========================================"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS (quick test)"
echo "========================================"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Starting training..."
python train.py --config "$CONFIG" --epochs "$EPOCHS" --no-wandb

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
    echo "   You can now run full experiments with:"
    echo "   python run_experiments.py --config $CONFIG"
else
    echo "❌ Test failed with exit code: $EXIT_CODE"
    echo "   Please check the error messages above."
fi
echo "========================================"

exit $EXIT_CODE
