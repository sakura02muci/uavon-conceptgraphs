#!/bin/bash
# Batch evaluation script for UAV-ON benchmark
# Usage: ./run_uavon_evaluation.sh [strategy] [num_episodes]

STRATEGY=${1:-clip}  # Default: clip
NUM_EPISODES=${2:-5}  # Default: 5 episodes per scene

echo "========================================="
echo "UAV-ON Benchmark Evaluation"
echo "Strategy: $STRATEGY"
echo "Episodes per scene: $NUM_EPISODES"
echo "========================================="

# Dataset scenes
SCENES=(
    "Barnyard"
    "BrushifyRoad"
    "BrushifyUrban"
    "CabinLake"
    "CityPark"
    "CityStreet"
    "DownTown"
    "Neighborhood"
    "NYC"
    "Slum"
    "UrbanJapan"
    "Venice"
    "WesternTown"
    "WinterTown"
)

# Create results directory
mkdir -p results/uavon_${STRATEGY}

cd "$(dirname "$0")"

# Evaluate each scene
for scene in "${SCENES[@]}"; do
    echo ""
    echo "=========================================== "
    echo "Evaluating scene: $scene"
    echo "=========================================="
    
    DATASET_PATH="../UAV-ON-dataset/valset/${scene}.json"
    OUTPUT_PATH="results/uavon_${STRATEGY}/${scene}.json"
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo "⚠️  Dataset not found: $DATASET_PATH"
        continue
    fi
    
    echo "📁 Dataset: $DATASET_PATH"
    echo "💾 Output: $OUTPUT_PATH"
    echo ""
    
    # NOTE: Environment needs to be manually started!
    echo "⚠️  MANUAL STEP REQUIRED:"
    echo "   1. Start UE4 environment in another terminal:"
    echo "      cd ../TEST_ENVS/${scene}/"
    echo "      ./${scene}_*.sh -windowed"
    echo "   2. Wait for environment to load (10-20 seconds)"
    echo "   3. Press ENTER to continue..."
    read -p ""
    
    # Run evaluation
    python scripts/eval_simple_uavon.py \
        --dataset "$DATASET_PATH" \
        --num_episodes "$NUM_EPISODES" \
        --strategy "$STRATEGY" \
        --output "$OUTPUT_PATH"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ $scene completed"
    else
        echo "❌ $scene failed"
    fi
    
    # Optional: Kill environment
    echo ""
    echo "Press ENTER to close environment and continue to next scene..."
    read -p ""
    pkill -f "Linux-Shipping"
    sleep 2
done

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved in: results/uavon_${STRATEGY}/"
echo ""
echo "To generate summary report:"
echo "  python scripts/aggregate_uavon_results.py --strategy $STRATEGY"
