CC = gcc
CC_FLAGS = -fopenmp -O3
LINK_FLAGS = -lm

all: btree

btree: main.c kernel_cpu.c kernel_cpu_2.c timer.c num.c
	$(CC) $(CC_FLAGS) $^ -o $@ $(LINK_FLAGS)

clean:
	rm btree *.o *.out output.txt