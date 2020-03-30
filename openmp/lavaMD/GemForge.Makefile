# C compiler
CC=clang

include ../GemForge.Makefile.include

SOURCES=main.c kernel_cpu.c timer.c num.c
OBJ_BCS=$(addprefix obj-, ${SOURCES:.c=.bc})

all: lavaMD.exe

riscv: raw.riscv.exe

lavaMD.exe: ${SOURCES}
	${CC} ${CC_FLAGS} $^ -o $@

obj-%.bc: %.c
	${CC} ${CC_FLAGS} -fno-unroll-loops $^ -emit-llvm -c -o $@

raw.bc: ${OBJ_BCS}
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
	rm -f *.exe *.bc *.ll *.o output.txt
