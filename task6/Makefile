# Makefile for md5

TARGET=md5
OBJS=md5.o md5_bmark.o
OBJS_PARALLEL=md5.o md5_bmark_parallel.o
CFLAGS=-O2 -g -D_GNU_SOURCE -std=c99 -Wall
GPROF=-pg
LDFLAGS= -fopenmp
CC=gcc
TESTS=$(wildcard *.in)
EXTENSION_SERIAL	:= serial
EXTENSION_PARALLEL	:= parallel

all: $(OBJS) $(OBJS_PARALLEL)
	$(CC) $(CFLAGS) $(GPROF) $(LDFLAGS) $(OBJS) -o $(TARGET) -fopenmp
	$(CC) $(CFLAGS) $(GPROF) $(LDFLAGS) $(OBJS_PARALLEL) -o $(TARGET)_parallel -fopenmp

%.o : %.c
	$(CC) $(CFLAGS) -c $< -fopenmp

test: all
	@$(foreach test_in, ${TESTS}, \
			echo ${test_in}; \
			./${TARGET}_parallel < ${test_in} > $(basename ${test_in}).${EXTENSION_PARALLEL}; \
			./${TARGET} < ${test_in} > $(basename ${test_in}).${EXTENSION_SERIAL}; \
			diff $(basename ${test_in}).${EXTENSION_PARALLEL} $(basename ${test_in}).${EXTENSION_SERIAL};)

clean:
	rm -f *.o $(TARGET) output.txt
