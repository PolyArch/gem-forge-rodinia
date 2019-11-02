# C compiler
CC=clang

include ../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

3D.bc: 3D.c
	$(CC) $(CC_FLAGS) $^ -emit-llvm -c -o $@

raw.bc: 3D.bc
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
