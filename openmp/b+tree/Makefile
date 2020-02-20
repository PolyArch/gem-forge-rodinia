CC = gcc
CC_FLAGS = -fopenmp -O3
LINK_FLAGS = -lm

all: btree

btree: gem5_main.c impl.c kernel_query.c kernel_range.c timer.c num.c
	$(CC) $(CC_FLAGS) -DDUMP_OUTPUT $^ -o $@ $(LINK_FLAGS)

clean:
	rm btree *.o output.txt