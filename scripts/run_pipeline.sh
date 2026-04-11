#!/bin/bash
# End-to-end evaluation pipeline for Push-Pull Attention-Guided Jailbreaking
#
# Usage:
#   bash scripts/run_pipeline.sh --model llava --model_path /path/to/model
#
# Options:
#   --model       Model type: llava, qwen, internvl (default: llava)
#   --model_path  Path to model (required)
#   --benchmark   Benchmark: advbench, harmbench, etc. (default: advbench)
#   --eps         Perturbation budget /255 (default: 16)
#   --num_iter    Number of iterations (default: 2000)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default arguments
MODEL="llava"
MODEL_PATH=""
BENCHMARK="advbench"
EPS=16
NUM_ITER=2000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --eps) EPS="$2"; shift 2 ;;
        --num_iter) NUM_ITER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: bash scripts/run_pipeline.sh --model llava --model_path /path/to/model"
    exit 1
fi

ATTACK_DIR="$PROJECT_DIR/results/${MODEL}_eps${EPS}_iter${NUM_ITER}"
RESPONSE_DIR="$PROJECT_DIR/results/responses"
EVAL_DIR="$PROJECT_DIR/results/evaluation"

mkdir -p "$ATTACK_DIR" "$RESPONSE_DIR" "$EVAL_DIR"

echo "=========================================="
echo "Push-Pull Attack Pipeline"
echo "=========================================="
echo "Model:       $MODEL ($MODEL_PATH)"
echo "Benchmark:   $BENCHMARK"
echo "Epsilon:     $EPS"
echo "Iterations:  $NUM_ITER"
echo "=========================================="
echo ""

# Step 1: Run adversarial attack
echo "[Step 1/3] Running adversarial attack..."
if [ -f "$ATTACK_DIR/adversarial.png" ]; then
    echo "  [SKIP] Adversarial image already exists: $ATTACK_DIR/adversarial.png"
else
    python "$PROJECT_DIR/attack/attack.py" \
        --model "$MODEL" \
        --model_path "$MODEL_PATH" \
        --image_path "$PROJECT_DIR/images/clean.jpeg" \
        --use_corpus \
        --num_iter "$NUM_ITER" \
        --alpha 1 --eps "$EPS" \
        --constrained \
        --alpha_suppress 10.0 \
        --beta_amplify 5.0 \
        --save_dir "$ATTACK_DIR"
    echo "  [OK] Attack complete"
fi

# Step 2: Generate responses
echo ""
echo "[Step 2/3] Generating model responses..."
RESPONSE_FILE="$RESPONSE_DIR/${MODEL}_${BENCHMARK}_eps${EPS}.json"

if [ -f "$RESPONSE_FILE" ]; then
    echo "  [SKIP] Responses already exist: $RESPONSE_FILE"
else
    python "$PROJECT_DIR/evaluation/generate_responses.py" \
        --model "$MODEL" \
        --input_file "$PROJECT_DIR/harmful_corpus/input_${BENCHMARK}.json" \
        --image_path "$ATTACK_DIR/adversarial.png" \
        --output_file "$RESPONSE_FILE" \
        --model_path "$MODEL_PATH"
    echo "  [OK] Responses generated"
fi

# Step 3: Evaluate safety
echo ""
echo "[Step 3/3] Evaluating safety..."
EVAL_FILE="$EVAL_DIR/${MODEL}_${BENCHMARK}_eps${EPS}_eval.csv"

python "$PROJECT_DIR/evaluation/evaluate.py" \
    --input_file "$RESPONSE_FILE" \
    --output_csv "$EVAL_FILE" \
    --benchmark "$BENCHMARK" \
    --condition "PushPull" \
    --model_name "$MODEL"

echo ""
echo "=========================================="
echo "[DONE] Pipeline complete"
echo "=========================================="
echo "Results:"
echo "  Adversarial Image: $ATTACK_DIR/adversarial.png"
echo "  Responses:         $RESPONSE_FILE"
echo "  Evaluation:       $EVAL_FILE"
echo ""
