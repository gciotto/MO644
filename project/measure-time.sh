#!/bin/bash

APP_NAME=dynamic_tracking
APP_CUDA=cuda_tracking

REPEAT=2000

make all

for TURNS in "10" "100" "1000" "10000"
do
        ./${APP_NAME} ${REPEAT} ${TURNS} > serial_${REPEAT}_${TURNS}.out
        ./${APP_CUDA} ${REPEAT} ${TURNS} > cuda_${REPEAT}_${TURNS}.out

        echo "REPEAT = ${REPEAT} TURNS = ${TURNS}"

        diff serial_${REPEAT}_${TURNS}.out cuda_${REPEAT}_${TURNS}.out
done    
