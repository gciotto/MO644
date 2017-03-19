GPROF				:= -pg
GPROF_FLAGS			:= -b
CFLAGS 				:= -std=c99 -pedantic -Wall -fopenmp -lm -g ${GPROF}
OCFLAG				:= 
SOURCE				:= prod_cons
SOURCE_SERIAL 		:= T2_ser
EXTENSION_SERIAL	:= serial
EXTENSION_PARALLEL	:= parallel
TESTS				:= $(wildcard *.in)

all: test

${SOURCE_SERIAL}:
	gcc ${CFLAGS} ${OCFLAG} -o $@ $@.c

${SOURCE}:
	gcc ${CFLAGS} -o $@ $@.c

test: ${SOURCE} ${SOURCE_SERIAL}
	@$(foreach test_in, ${TESTS}, \
			echo ${test_in}; \
			./${SOURCE} < ${test_in} > $(basename ${test_in}).${EXTENSION_PARALLEL}; \
			gprof ${GPROF_FLAGS} ${SOURCE} gmon.out > ${SOURCE}-$(basename ${test_in}).profile; \
			rm gmon.out; \
			./${SOURCE_SERIAL} < ${test_in} > $(basename ${test_in}).${EXTENSION_SERIAL}; \
			gprof ${GPROF_FLAGS} ${SOURCE_SERIAL} gmon.out > ${SOURCE_SERIAL}-$(basename ${test_in}).profile; \
			rm gmon.out; \
			diff $(basename ${test_in}).${EXTENSION_PARALLEL} $(basename ${test_in}).${EXTENSION_SERIAL};)

clean:
	rm ${SOURCE}
	rm ${SOURCE_SERIAL}
	rm *.${EXTENSION_SERIAL}
	rm *.${EXTENSION_PARALLEL}
	rm *.profile

