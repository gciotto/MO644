APP_NAME = dynamic_tracking
APP_ORIGINAL = tracking
APP_CUDA = cuda_tracking

CUDA_SOURCES = DynamicSearch.cpp RingElement.cpp CudaDynamicSearch.cu ParallelTracking.cu

CPP = g++
CPPFLAGS += -O0 -Wall --std=c++11

OBJS = DynamicSearch.o RingElement.o SerialTracking.o
OBJS_ORIGINAL = tracking_skeleton.o

NVCC = /opt/cuda/bin/nvcc

all: clean $(APP_NAME) $(APP_CUDA)

%.o : %.c
	$(CPP) $(CPP_FLAGS) $< -o $@

$(APP_NAME): $(OBJS)
	$(CPP) -o $(APP_NAME) $(OBJS)

$(APP_ORIGINAL): $(OBJS_ORIGINAL)
	$(CPP) -o $(APP_ORIGINAL) $(OBJS_ORIGINAL)

$(APP_CUDA):
	$(NVCC) -std=c++11 -o $(APP_CUDA) $(CUDA_SOURCES)

clean:
	rm -f *.o
	rm -f $(APP_NAME) $(APP_ORIGINAL) $(APP_CUDA)
