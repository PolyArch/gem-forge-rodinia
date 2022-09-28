# C compiler
CC=clang

include ../GemForge.Makefile.include

SOURCES=gem5_main.c impl.c kernel_query.c kernel_range.c timer.c num.c
OBJ_BCS=$(addprefix obj-, ${SOURCES:.c=.bc})

all: raw.bc

obj-%.bc: %.c
	${CC} ${CC_FLAGS} -fno-unroll-loops $^ -emit-llvm -c -o $@

raw.bc: ${OBJ_BCS}
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

native.exe: gem5_main.c impl.c kernel_query.c kernel_range.c timer.c num.c
	${CC} $^ -O3 -fopenmp -lm -o $@

clean:
	rm -f *.exe *.bc *.ll *.o output.txt
