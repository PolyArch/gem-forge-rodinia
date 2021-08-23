
CC=clang

include ../../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

3D.bc: 3D.c
	$(CC) $(CC_FLAGS) -march=knl -DFIX_ROW=256 -DFIX_COL=1024 -DFIX_Z=8 -DFUSE_OUTER_LOOPS $^ -emit-llvm -c -o $@

raw.bc: 3D.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
