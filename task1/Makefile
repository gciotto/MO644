CFLAGS 	:= -g -Wall -fopenmp
SOURCE 	:= count_sort_paralelo
TESTS	:= $(wildcard *.in)

all:
	gcc ${CFLAGS} -o ${SOURCE} ${SOURCE}.c

test: ${SOURCE}
	$(foreach test_in, ${TESTS}, ./${SOURCE} < ${test_in} > $(basename ${test_in}).out; diff $(basename ${test_in}).out $(basename ${test_in}).res;)

clean:
	rm ${SOURCE}
	rm *.out
