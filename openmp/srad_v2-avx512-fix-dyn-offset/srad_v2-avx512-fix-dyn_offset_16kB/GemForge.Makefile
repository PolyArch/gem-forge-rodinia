
CC=clang

include ../../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

%.bc: %.cpp
	$(CC) $(CC_FLAGS) -DGEM_FORGE_FIX_INPUT -DGEM_FORGE_DYN_SCHEDULE=2 -march=knl -fno-unroll-loops $^ -emit-llvm -c -o $@

raw.bc: srad.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
