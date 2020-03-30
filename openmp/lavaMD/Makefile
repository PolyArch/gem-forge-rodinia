CC = gcc
CC_FLAGS = -fopenmp -O3
LINK_FLAGS = -lm

all: lavaMD

lavaMD: main.c kernel_cpu.c timer.c num.c
	$(CC) $(CC_FLAGS) -DDUMP_OUTPUT $^ -o $@ $(LINK_FLAGS)

clean:
	rm -f lavaMD *.o output.txt