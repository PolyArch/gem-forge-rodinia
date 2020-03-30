# C compiler
CC=clang

include ../GemForge.Makefile.include

all: nn.exe

riscv: raw.riscv.exe

nn.bc: nn.c
	$(CC) $(CC_FLAGS) -fno-vectorize -DIS_BINARY=1 $^ -emit-llvm -c -o $@

raw.bc: nn.bc
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
