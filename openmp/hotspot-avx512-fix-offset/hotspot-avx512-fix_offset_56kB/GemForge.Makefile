
CC=clang++

include ../../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

hotspot.bc: hotspot.cpp
	$(CC) $(CC_FLAGS) -DGEM_FORGE_FIX_INPUT -DGEM_FORGE_FIX_INPUT_SIZE=1024 -mavx512f $^ -emit-llvm -c -o $@

raw.bc: hotspot.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
