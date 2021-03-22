# C compiler
CC=clang

# include ../GemForge.Makefile.include
CC_FLAGS=-gline-tables-only -O3 -I${GEM_FORGE_TOP}/gem5/include -DGEM_FORGE -DGEM_FORGE_WARM_CACHE -DGEM_FORGE_MALLOC_ARENA_MAX=2 -mllvm -loop-unswitch-threshold=1 -Rpass-analysis=loop-vectorize
CC_FLAGS+= -DGEM_FORGE_FIX_ONE_CHUNK
CC_FLAGS+= -DGEM_FORGE_FIX_SP_1
CC_FLAGS+= -DGEM_FORGE_FIX_DIM_16
CC_FLAGS+= -DENABLE_THREADS

all: bfs.exe

riscv: raw.riscv.exe

%.bc: %.cpp
	$(CC) $(CC_FLAGS) -mavx512f $^ -emit-llvm -c -o $@ -pthread

raw.bc: streamcluster_gem_forge.bc
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
