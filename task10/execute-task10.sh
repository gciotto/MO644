#!/bin/bash

set -o xtrace

APP_SERIAL=mvt
APP_PARALLEL=mvt_parallel

PARALLEL_DEFINE=("MEDIUM" "LARGE" "EXTRALARGE")
PARALLEL_FLAGS=("none" "tile" "vectorize")       

CFLAGS="-O3 -Wall"
ACLANG_FLAGS="-O3 -rtl-mode=profile"

rm -f ${APP_SERIAL} ${APP_PARALLEL}

# Compile serial source file

for DEFINE in ${PARALLEL_DEFINE[@]}; do

        gcc ${CFLAGS} -D${DEFINE} -o ${APP_SERIAL} ${APP_SERIAL}.c
        ./${APP_SERIAL} > ${APP_SERIAL}_${DEFINE}.out

 #       for FLAG in ${PARALLEL_FLAGS[@]}; do

#                aclang ${ACLANG_FLAGS} -opt-poly=${FLAG} -D${DEFINE} -o ${APP_PARALLEL} ${APP_PARALLEL}.c
#                ./${APP_PARALLEL} > ${APP_PARALLEL}_${DEFINE}_${FLAG}.out

  #      done

done

