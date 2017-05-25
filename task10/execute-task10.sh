#!/bin/sh

#set -o xtrace

APP_SERIAL=mvt
APP_PARALLEL=mvt_parallel

PARALLEL_DEFINE=("MEDIUM" "LARGE" "EXTRALARGE")
PARALLEL_FLAGS=("none" "tile" "vectorize")

CFLAGS="-O3 -Wall"
ACLANG_FLAGS="-O3 -rtl-mode=profile"
BUG_ACLANG_FLAGS="-fopenmp -omptargets=opencl-unknown-unknown"

rm -f ${APP_SERIAL} ${APP_PARALLEL}
rm -f *.out

# Compile serial source file

for DEFINE in ${PARALLEL_DEFINE[@]}; do

        gcc ${CFLAGS} -D${DEFINE} -o ${APP_SERIAL} ${APP_SERIAL}.c
        ./${APP_SERIAL} > ${APP_SERIAL}_${DEFINE}.out

       for FLAG in ${PARALLEL_FLAGS[@]}; do

		echo "aclang ${BUG_ACLANG_FLAGS} ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c"
                aclang ${BUG_ACLANG_FLAGS} ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c &> /dev/null

                ./${APP_PARALLEL} &> ${APP_PARALLEL}_${DEFINE}_${FLAG}_bug.out

                echo "aclang ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c"
                aclang ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c

                ./${APP_PARALLEL} &> ${APP_PARALLEL}_${DEFINE}_${FLAG}.out

		diff ${APP_SERIAL}_${DEFINE}.out ${APP_PARALLEL}_${DEFINE}_${FLAG}.out

      done

done

