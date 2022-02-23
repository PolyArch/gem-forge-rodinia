
CC=clang

include ../../GemForge.Makefile.include

riscv: raw.riscv.exe

%.bc: %.cpp
	$(CC) $(CC_FLAGS) -DGEM_FORGE_FIX_INPUT -march=knl -fno-unroll-loops $^ -emit-llvm -c -o $@

raw.bc: srad.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
