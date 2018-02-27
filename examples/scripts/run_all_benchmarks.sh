#!/bin/bash

BENCHMARKS="dot_product saxpy word_count sgemv mandelbrot"
DEVICES="gpu cpu"
VERSIONS="slow fast"

for benchmark in $BENCHMARKS
do
  echo $benchmark

  for device in $DEVICES
  do

    echo $device

    for version in $VERSIONS
    do
      echo $version
      ./run_benchmark.zsh "${benchmark}_${version}" $device
    done

  done

done
