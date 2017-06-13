#!/bin/bash

N_SET=("100" "1000" "1500" "2000" "2500" "3500" "5000")

APP=code-t12-cloud

for N in ${N_SET[@]}; do

	make N=${N}

	./${APP} > n${N}.out

done
