#!/bin/sh

#set -o xtrace

APP_SERIAL=mvt
APP_PARALLEL=mvt_parallel

PARALLEL_DEFINE=("MEDIUM" "LARGE" "EXTRALARGE")
PARALLEL_FLAGS=("none" "tile" "vectorize")

CFLAGS="-O3 -Wall"
ACLANG_FLAGS="-O3 -rtl-mode=profile"
BUG_ACLANG_FLAGS="-fopenmp -omptargets=opencl-unknown-unknown"

N=5

TIME_PARALLEL_LINE=11

rm -f ${APP_SERIAL} ${APP_PARALLEL}
rm -f *.out *.avg
rm -f *.bc *.cl

# Compile serial source file

for DEFINE in ${PARALLEL_DEFINE[@]}; do

        gcc ${CFLAGS} -D${DEFINE} -o ${APP_SERIAL} ${APP_SERIAL}.c
        ./${APP_SERIAL} > ${APP_SERIAL}_${DEFINE}.out

        TIME_SERIAL=$(sed '2q;d' ${APP_SERIAL}_${DEFINE}.out)

        for FLAG in ${PARALLEL_FLAGS[@]}; do

                aclang ${BUG_ACLANG_FLAGS} ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c &> /dev/null

#                aclang ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c

		# Discards first execution
                ./${APP_PARALLEL} &> ${APP_PARALLEL}_${DEFINE}_${FLAG}_${I}.out

		SUM=0.0
                SUM_READ_TIMES=0.0
                SUM_EXEC_TIMES=0.0
                for I in $(seq 1 ${N}); do

	                ./${APP_PARALLEL} &> ${APP_PARALLEL}_${DEFINE}_${FLAG}_${I}.out

			diff ${APP_PARALLEL}_${DEFINE}_${FLAG}_${I}.out ${APP_SERIAL}_${DEFINE}.out

			TIME_PARALLEL=$(sed "${TIME_PARALLEL_LINE}q;d" ${APP_PARALLEL}_${DEFINE}_${FLAG}_${I}.out)
			SUM=$(echo ${SUM} + ${TIME_PARALLEL} | bc -l)

                        TIMES_READ_BUFFER=$(cat ${APP_PARALLEL}_${DEFINE}_${FLAG}_${I}.out | grep -e _cl_read_buffer -e _cl_offloading | awk '{print $4}')
                        SUM_TIMES=0.0
                        for TIME in ${TIMES_READ_BUFFER}; do

                                SUM_TIMES=$(echo ${TIME} + ${SUM_TIMES} | bc -l)
                        done

			SUM_READ_TIMES=$(echo ${SUM_READ_TIMES} + ${SUM_TIMES} | bc -l)

                        TIMES_EXEC_BUFFER=$(cat ${APP_PARALLEL}_${DEFINE}_${FLAG}_${I}.out | grep _cl_execute_kernel | awk '{print $4}')
                        SUM_TIMES=0.0
                        for TIME in ${TIMES_EXEC_BUFFER}; do

                                SUM_TIMES=$(echo ${TIME} + ${SUM_TIMES} | bc -l)
                        done

			SUM_EXEC_TIMES=$(echo ${SUM_EXEC_TIMES} + ${SUM_TIMES} | bc -l)

		done

		AVG=$(echo ${SUM} / ${N} | bc -l)
		AVG_READ_TIMES=$(echo ${SUM_READ_TIMES} / ${N} / 1000000000 | bc -l)
		AVG_EXEC_TIMES=$(echo ${SUM_EXEC_TIMES} / ${N} / 1000000000 | bc -l)
                RATE=$(echo ${AVG_READ_TIMES} / ${TIME_SERIAL} | bc -l )
                SPEEDUP=$(echo ${TIME_SERIAL} / ${AVG} | bc -l)

		printf "${DEFINE} ${FLAG}\nSerial = ${TIME_SERIAL}; Parallel = ${AVG}; Speedup = ${SPEEDUP}; Rate = ${RATE}\n\n" >> summary_${DEFINE}.res

      done

done
