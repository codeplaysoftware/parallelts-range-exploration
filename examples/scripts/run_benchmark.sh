#!/bin/bash

BENCHMARK=$1
DEVICE=$2

SIZES="1 2 4 8 16 32 64 128 256"

mkdir -p logs

if [ ! -x ../build/benchmarks/$BENCHMARK ]
then
  echo "No executable for $BENCHMARK"
  exit
fi

for size in $SIZES
do
  COMPUTECPP_TARGET="intel:$DEVICE" ../build/benchmarks/$BENCHMARK $size > "logs/${BENCHMARK}_${DEVICE}_${size}.log"
done
