#!/usr/bin/env bash

set -u

CONFIG="configs/multidoc2dial.yaml"
SIZE=128

echo "============================================================"
echo "SCLC ECIR - MultiDoc2Dial - 128 TOKEN EXPERIMENT"
echo "============================================================"
echo

run_cmd() {
    echo
    echo ">>> $*"
    echo
    "$@"

    if [ $? -ne 0 ]; then
        echo
        echo "============================================================"
        echo "ERROR: command failed"
        echo "============================================================"
        echo
        read -p "Press Enter to close..."
        exit 1
    fi
}

echo "============================================================"
echo "1. BUILD 128-TOKEN RETRIEVAL UNITS"
echo "============================================================"

run_cmd sclc build-units \
    --config "$CONFIG" \
    --retrieval-unit-size "$SIZE"

echo
echo "============================================================"
echo "2. ENCODE GRANITE"
echo "============================================================"

for CONDITION in section_isolated section_constrained global length_matched
do
    run_cmd sclc encode \
        --config "$CONFIG" \
        --condition "$CONDITION" \
        --model granite \
        --retrieval-unit-size "$SIZE" \
        --overwrite
done

echo
echo "============================================================"
echo "3. ENCODE JINA"
echo "============================================================"

for CONDITION in section_isolated section_constrained global length_matched
do
    run_cmd sclc encode \
        --config "$CONFIG" \
        --condition "$CONDITION" \
        --model jina \
        --retrieval-unit-size "$SIZE" \
        --overwrite
done

echo
echo "============================================================"
echo "4. RETRIEVE GRANITE"
echo "============================================================"

for CONDITION in section_isolated section_constrained global length_matched
do
    run_cmd sclc retrieve \
        --config "$CONFIG" \
        --condition "$CONDITION" \
        --model granite \
        --retrieval-unit-size "$SIZE" \
        --overwrite
done

echo
echo "============================================================"
echo "5. RETRIEVE JINA"
echo "============================================================"

for CONDITION in section_isolated section_constrained global length_matched
do
    run_cmd sclc retrieve \
        --config "$CONFIG" \
        --condition "$CONDITION" \
        --model jina \
        --retrieval-unit-size "$SIZE" \
        --overwrite
done

echo
echo "============================================================"
echo "6. EVALUATE GRANITE"
echo "============================================================"

for CONDITION in section_isolated section_constrained global length_matched
do
    run_cmd sclc evaluate \
        --config "$CONFIG" \
        --condition "$CONDITION" \
        --model granite \
        --retrieval-unit-size "$SIZE" \
        --overwrite
done

echo
echo "============================================================"
echo "7. EVALUATE JINA"
echo "============================================================"

for CONDITION in section_isolated section_constrained global length_matched
do
    run_cmd sclc evaluate \
        --config "$CONFIG" \
        --condition "$CONDITION" \
        --model jina \
        --retrieval-unit-size "$SIZE" \
        --overwrite
done

echo
echo "============================================================"
echo "8. COMPARE GRANITE"
echo "============================================================"

run_cmd sclc compare \
    --config "$CONFIG" \
    --model granite \
    --retrieval-unit-size "$SIZE"

echo
echo "============================================================"
echo "9. COMPARE JINA"
echo "============================================================"

run_cmd sclc compare \
    --config "$CONFIG" \
    --model jina \
    --retrieval-unit-size "$SIZE"

echo
echo "============================================================"
echo "128-TOKEN EXPERIMENT COMPLETE"
echo "============================================================"
echo

read -p "Experiment complete. Press Enter to close..."
