CFLAGS 			:= -lm
SOURCE			:= smooth_cuda
SOURCE_SERIAL 	:= smooth_serial
TESTS			:= $(wildcard *.ppm)
EXTENSION_SERIAL	:= serial
EXTENSION_PARALLEL	:= parallel

all: test

build: clean ${SOURCE} ${SOURCE_SERIAL}

${SOURCE_SERIAL}:
	gcc ${CFLAGS} -o $@ $@.c

${SOURCE}:
	nvcc -o $@ $@.cu

test: build
	@$(foreach test_in, ${TESTS}, \
			echo ${test_in}; \
			./${SOURCE} ${test_in}> $(basename ${test_in}).${EXTENSION_PARALLEL}; \
			./${SOURCE_SERIAL} ${test_in} > $(basename ${test_in}).${EXTENSION_SERIAL}; \
			diff $(basename ${test_in}).${EXTENSION_PARALLEL} $(basename ${test_in}).${EXTENSION_SERIAL};)

clean:
	rm -f ${SOURCE}
	rm -f ${SOURCE_SERIAL}
	rm -f *.${EXTENSION_SERIAL}
	rm -f *.${EXTENSION_PARALLEL}
	rm -f *.profile

