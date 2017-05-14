APP_NAME = dynamic_tracking
APP_ORIGINAL = tracking

CPP = g++
CPPFLAGS += -O0 -Wall --std=c++11

OBJS = DynamicSearch.o RingElement.o SerialTracking.o
OBJS_ORIGINAL = tracking_skeleton.o

all: clean $(APP_NAME) $(APP_ORIGINAL)

%.o : %.c
	$(CPP) $(CPP_FLAGS) $< -o $@
	
$(APP_NAME): $(OBJS)
	$(CPP) -o $(APP_NAME) $(OBJS)
	
$(APP_ORIGINAL): $(OBJS_ORIGINAL)
	$(CPP) -o $(APP_ORIGINAL) $(OBJS_ORIGINAL)
	
clean:
	rm -f *.o
	rm -f $(APP_NAME) $(APP_ORIGINAL)