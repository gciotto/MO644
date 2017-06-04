#!/bin/bash

INPUTS=("alice30" "gmars11" "warw10" "wizoz10")

for INPUT1 in ${INPUTS[@]}; do

	for INPUT2 in ${INPUTS[@]}; do

		if [ "${INPUT1}" != "${INPUT2}" ]; then

			echo "INPUTS ${INPUT1} and ${INPUT2}"

			spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar ${INPUT1}.txt ${INPUT2}.txt > ${INPUT1}_x_${INPUT2}.out

			diff ${INPUT1}_x_${INPUT2}.out ${INPUT1}_x_${INPUT2}.txt

		fi

	done

done
