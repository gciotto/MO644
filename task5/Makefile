GPROF			:= 
GPROF_FLAGS		:= 
CFLAGS 			:= -std=c99 -pedantic -Wall
CFLAGS_PARALLEL		:= -lpthread
OCFLAG			:= 
SOURCE			:= senha-paralelo-control-thread
SOURCE_SERIAL 		:= senha-serial
EXTENSION_SERIAL	:= serial
EXTENSION_PARALLEL	:= parallel
TESTS			:= $(wildcard *.in)

all: test

build: clean ${SOURCE} ${SOURCE_SERIAL}

${SOURCE_SERIAL}:
	gcc ${CFLAGS} ${GPROF} ${OCFLAG} -o $@ $@.c

${SOURCE}:
	gcc ${CFLAGS} ${CFLAGS_PARALLEL} -o $@ $@.c

test: build
	@$(foreach test_in, ${TESTS}, \
			echo ${test_in}; \
			./${SOURCE} < ${test_in}> $(basename ${test_in}).${EXTENSION_PARALLEL}; \
			./${SOURCE_SERIAL} < ${test_in} > $(basename ${test_in}).${EXTENSION_SERIAL};)
#			diff $(basename ${test_in}).${EXTENSION_PARALLEL} $(basename ${test_in}).${EXTENSION_SERIAL};)

clean:
	rm -f ${SOURCE}
	rm -f ${SOURCE_SERIAL}
	rm -f *.${EXTENSION_SERIAL}
	rm -f *.${EXTENSION_PARALLEL}
	rm -f *.profile

