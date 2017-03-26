#!/bin/bash

PARALLEL=hist_paralelo
SERIAL=hist_s

TMP_IN=temp_in
TMP_OUT_S=temp_out_s
TMP_OUT_P=temp_out_p

CORES=4

# Compile all source files
make build

for INPUT in ${PWD}/*.in; do

	echo $(basename ${INPUT})

	for N_THREAD in 2 4 8 16 ; do

		echo N_THREAD = ${N_THREAD}

		APP_INPUT=$(cat ${INPUT} | sed '1 s/2/'"${N_THREAD}"'/')

		echo $APP_INPUT > ${TMP_IN}

		./${PARALLEL} < ${TMP_IN} > ${TMP_OUT_P}

		T_PARALLEL=$(sed '3q;d' ${TMP_OUT_P})

		./${SERIAL} < ${INPUT} > ${TMP_OUT_S}

		T_SERIAL=$(sed '3q;d' ${TMP_OUT_S})

		diff ${TMP_OUT_P} ${TMP_OUT_S} 

		echo speedup = $( echo "${T_SERIAL} / ${T_PARALLEL}" | bc -l)
		echo efficieny = $( echo "${T_SERIAL} / ( ${CORES} * ${T_PARALLEL})" | bc -l)

		# profiling serial program with gprof

		gprof -b -p ${SERIAL} gmon.out > $(basename ${INPUT})-${N_THREAD}.grof.profile

		# profiling parallel with perf

		perf stat -d -d -d -B -o $(basename ${INPUT})-${N_THREAD}.perf.profile ./${PARALLEL} < ${TMP_IN} > /dev/null

	done 

done

rm -f ${TMP_OUT_S}
rm -f ${TMP_OUT_P}
rm -f ${TMP_IN}
