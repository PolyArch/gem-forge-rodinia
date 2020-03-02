# C compiler
CC=clang

# Disable slp vectorizer as it generates v4f64 which can't be handled by gem5.
# CC_FLAGS = -fno-slp-vectorize -fno-vectorize -fno-unroll-loops

include ../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

%.bc: %.c
	$(CC) $(CC_FLAGS) $^ -emit-llvm -c -o $@

raw.bc: particlefilter.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

%.exe: %.bc
	${CC} ${CC_FLAGS} -o $@

raw.riscv.exe: raw.bc
	${CC} ${RISCV_CC_FLAGS} ${CC_FLAGS} $^ -c -o raw.riscv.o
	${RISCV_GCC} raw.riscv.o ${RISCV_LD_FLAGS} -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
