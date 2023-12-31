# C compiler
CC=clang++

include ../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

hotspot.bc: hotspot.cpp
	$(CC) $(CC_FLAGS) -DGEM_FORGE_FIX_INPUT -DGEM_FORGE_FIX_INPUT_SIZE=1024 -DRANDOMIZE -mavx512f $^ -emit-llvm -c -o $@

raw.bc: hotspot.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

%.exe: %.bc
	${CC} ${CC_FLAGS} -march=knl -o $@

raw.riscv.exe: raw.bc
	${CC} ${RISCV_CC_FLAGS} ${CC_FLAGS} $^ -c -o raw.riscv.o
	${RISCV_GCC} raw.riscv.o ${RISCV_LD_FLAGS} -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
