# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2
OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"

all: hotspot hotspot_offload 


hotspot: hotspot.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot.cpp -o hotspot 

hotspot_offload: hotspot.cpp Makefile
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD hotspot.cpp -o hotspot_offload

clean:
	rm -f hotspot hotspot_offload
