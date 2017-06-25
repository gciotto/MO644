APP_NAME = dynamic_tracking
APP_ORIGINAL = tracking
APP_CUDA = cuda_tracking

CUDA_SOURCES = DynamicSearch.cpp RingElement.cpp CudaDynamicSearch.cu ParallelTracking.cu

CPP = g++
CPPFLAGS += -O3 -Wall --std=c++11 -mfpmath=sse -msse2
GPROF_FLAGS=-pg

OBJS = DynamicSearch.o RingElement.o SerialTracking.o
OBJS_ORIGINAL = tracking_skeleton.o

NVCC = nvcc
NVCCFLAGS = -std=c++11 -DCUDA_FMA

INPUT = $(wildcard input/*.dat)
OUTPUT = output.dat

.PHONY: all clean run

all: clean $(APP_NAME) $(APP_CUDA)

%.o : %.c
	$(CPP) $(CPP_FLAGS) $< -o $@

$(APP_NAME): $(OBJS)
	$(CPP) $(CPPFLAGS) -o $(APP_NAME) $(OBJS)

$(APP_ORIGINAL): $(OBJS_ORIGINAL)
	$(CPP) -o $(APP_ORIGINAL) $(OBJS_ORIGINAL)

$(APP_CUDA):
	$(NVCC) $(NVCCFLAGS) -o $(APP_CUDA) $(CUDA_SOURCES)

clean:
	rm -f *.o
	rm -f $(APP_NAME) $(APP_ORIGINAL) $(APP_CUDA)
	rm -f $(OUTPUT)

run: clean $(APP_CUDA) $(APP_NAME)
	@$(foreach IN, $(INPUT), \
		echo $(IN); \
		printf "\n$(IN) ================================     ================     ================================\n\n" >> $(OUTPUT); \
		echo "Executing serial program (this may take a while)..."; \
		./$(APP_NAME) < $(IN) >  serial.dat; \
		TIME_SERIAL=$$(cat serial.dat | grep "Total elapsed time" | awk '{print $$5}') ; \
		echo "Executing parallel program ..."; \
		./$(APP_CUDA) < $(IN) > parallel.dat; \
		TIME_PARALLEL=$$(cat parallel.dat | grep "Total elapsed time" | awk '{print $$5}') ; \
		cat parallel.dat >> $(OUTPUT); \
		printf "\nSpeedup: " >> $(OUTPUT); \
		echo "$${TIME_SERIAL} / $${TIME_PARALLEL}" | bc -l >> $(OUTPUT); \
		diff parallel.dat serial.dat >> test.dat; )

	@rm -f parallel.dat serial.dat
