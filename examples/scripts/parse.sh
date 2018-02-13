#!/bin/bash

BENCHMARKS="dot_product saxpy word_count sgemv mandelbrot"
DEVICES="gpu cpu"
VERSIONS="slow fast"
SIZES="1 2 4 8 16 32 64 128 256"

FILE=sycl-ranges-results.csv

echo "device,benchmark,size,path,version,time" > $FILE

for benchmark in $BENCHMARKS
do

  for device in $DEVICES
  do

    for version in $VERSIONS
    do
      for size in $SIZES
      do

        LOG_FILE="logs/${benchmark}_${version}_${device}_${size}.log"

        if [ -a $LOG_FILE ]
        then
          run_time=$(awk -F ' ' '/Median/ { print $3; }' $LOG_FILE)

          echo "${device},${benchmark},${size},${version},sycl-ranges,${run_time}" >> $FILE
        fi
      done
    done

  done

done
