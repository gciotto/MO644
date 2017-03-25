GPROF				:= -pg
GPROF_FLAGS			:= -b
CFLAGS 				:= -std=c99 -pedantic -Wall -fopenmp -lm -g
CFLAGS_PARALLEL		:= -lpthread
OCFLAG				:= 
SOURCE				:= hist_paralelo
SOURCE_SERIAL 		:= hist_s
EXTENSION_SERIAL	:= serial
EXTENSION_PARALLEL	:= parallel
TESTS				:= $(wildcard *.in)

all: test

build: clean ${SOURCE} ${SOURCE_SERIAL}

${SOURCE_SERIAL}:
	gcc ${CFLAGS} ${GPROF} ${OCFLAG} -o $@ $@.c

${SOURCE}:
	gcc ${CFLAGS} ${CFLAGS_PARALLEL} -o $@ $@.c

test: build
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
	rm -f ${SOURCE}
	rm -f ${SOURCE_SERIAL}
	rm -f *.${EXTENSION_SERIAL}
	rm -f *.${EXTENSION_PARALLEL}
	rm -f *.profile

