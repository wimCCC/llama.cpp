#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

BIN=${BIN:-build-cuda/bin/llama-mtmd-cli}
MODEL=${MODEL:-Om_AI_Lab--VLM-FO1-3B-v01/vlm-fo1-text-f16-00001-of-00002.gguf}
MMPROJ=${MMPROJ:-Om_AI_Lab--VLM-FO1-3B-v01/mmproj-vlm-fo1-f16.gguf}
IMAGE=${IMAGE:-tools/mtmd/tests/test-1-positive.png}
PROMPT=${PROMPT:-Please briefly describe the image.}
GPU=${GPU:-6}
RUNS=${RUNS:-3}
N_PREDICT=${N_PREDICT:-128}
CONTEXT=${CONTEXT:-4096}
BATCH=${BATCH:-1024}
UBATCH=${UBATCH:-512}
THREADS=${THREADS:-16}
COLD=0
OUTPUT_DIR=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/vlm-fo1-eval.sh [options] [-- extra llama-mtmd-cli arguments]

Options:
  --runs N         Number of measured runs (default: 3)
  --gpu N          Physical GPU index (default: 6; GPU 7 is intentionally blocked)
  --tokens N       Fixed generated token count (default: 128)
  --output-dir DIR Directory for logs and metrics.csv
  --cold           Disable startup warmup
  -h, --help       Show this help

Paths and workload can also be overridden with environment variables:
  BIN, MODEL, MMPROJ, IMAGE, PROMPT, GPU, RUNS, N_PREDICT,
  CONTEXT, BATCH, UBATCH, THREADS

Reported metrics:
  TTFT_e2e, prefill, TTFT_decode, TPOT_decode, TPOT_e2e, generation

The script uses --ignore-eos so every run has the same decode length.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --runs)
            RUNS=$2
            shift 2
            ;;
        --gpu)
            GPU=$2
            shift 2
            ;;
        --tokens)
            N_PREDICT=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --cold)
            COLD=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$GPU" == "7" ]]; then
    echo "GPU 7 is blocked for this benchmark. Select another physical GPU with --gpu." >&2
    exit 2
fi

for path in "$BIN" "$MODEL" "$MMPROJ" "$IMAGE"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
done

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$N_PREDICT" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS and N_PREDICT must be positive integers." >&2
    exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="benchmark-results/vlm-fo1-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"
CSV="$OUTPUT_DIR/metrics.csv"
printf 'run,tokens,vision_encode_ms,ttft_e2e_ms,prefill_ms,ttft_decode_ms,tpot_decode_ms,tpot_e2e_ms,generation_ms\n' > "$CSV"

extract_metric() {
    local line=$1
    local key=$2
    printf '%s\n' "$line" | sed -n "s/.*${key}=\\([0-9.][0-9.]*\\).*/\\1/p"
}

for (( run = 1; run <= RUNS; run++ )); do
    log="$OUTPUT_DIR/run-${run}.log"
    cmd=(
        "$BIN"
        -m "$MODEL"
        --mmproj "$MMPROJ"
        --image "$IMAGE"
        -p "$PROMPT"
        -n "$N_PREDICT"
        --ignore-eos
        -c "$CONTEXT"
        -b "$BATCH"
        -ub "$UBATCH"
        -t "$THREADS"
        -tb "$THREADS"
        -ngl all
        --device CUDA0
        --seed 1
    )
    if (( COLD )); then
        cmd+=(--no-warmup)
    fi
    cmd+=("${EXTRA_ARGS[@]}")

    echo "Running sample $run/$RUNS on physical GPU $GPU..."
    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" > "$log" 2>&1

    line=$(grep -o 'generation: tokens=.*' "$log" | tail -1 || true)
    if [[ -z "$line" ]]; then
        echo "Run $run did not produce a generation metrics line. See $log" >&2
        exit 1
    fi

    tokens=$(extract_metric "$line" tokens)
    ttft_e2e=$(extract_metric "$line" TTFT_e2e)
    prefill=$(extract_metric "$line" prefill)
    ttft_decode=$(extract_metric "$line" TTFT_decode)
    tpot_decode=$(extract_metric "$line" TPOT_decode)
    tpot_e2e=$(extract_metric "$line" TPOT_e2e)
    generation=$(extract_metric "$line" generation)
    vision_encode=$(sed -n 's/.*mtmd batch encoding done in \([0-9][0-9]*\) ms.*/\1/p' "$log" | tail -1)
    vision_encode=${vision_encode:-0}

    printf '%d,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$run" "$tokens" "$vision_encode" "$ttft_e2e" "$prefill" "$ttft_decode" \
        "$tpot_decode" "$tpot_e2e" "$generation" >> "$CSV"
    printf '  TTFT_e2e=%s ms, prefill=%s ms, TTFT_decode=%s ms, TPOT_decode=%s ms/token, TPOT_e2e=%s ms/token\n' \
        "$ttft_e2e" "$prefill" "$ttft_decode" "$tpot_decode" "$tpot_e2e"
done

median_column() {
    local column=$1
    awk -F, -v column="$column" 'NR > 1 { print $column }' "$CSV" | sort -n | \
        awk '{ values[NR] = $1 } END { if (NR % 2) print values[(NR + 1) / 2]; else printf "%.3f\n", (values[NR / 2] + values[NR / 2 + 1]) / 2 }'
}

printf '\nMedian of %d runs:\n' "$RUNS"
printf '  vision_encode: %s ms\n' "$(median_column 3)"
printf '  TTFT_e2e:      %s ms\n' "$(median_column 4)"
printf '  prefill:       %s ms\n' "$(median_column 5)"
printf '  TTFT_decode:   %s ms\n' "$(median_column 6)"
printf '  TPOT_decode:   %s ms/token\n' "$(median_column 7)"
printf '  TPOT_e2e:      %s ms/token\n' "$(median_column 8)"
printf '  generation:    %s ms\n' "$(median_column 9)"
printf '\nLogs and CSV: %s\n' "$OUTPUT_DIR"
