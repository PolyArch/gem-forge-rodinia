# C compiler
CC=clang
CC_FLAGS=-gline-tables-only -fopenmp -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -I/home/zhengrong/Documents/llvm-trace-cpu/include -DGEM_FORGE

RISCV_INSTALL=/home/zhengrong/Documents/riscv/install
RISCV_SYSROOT=${RISCV_INSTALL}/sysroot
RISCV_CC_FLAGS=--target=riscv64-unknown-linux-gnu -march=rv64g -mabi=lp64d --sysroot=${RISCV_SYSROOT}
RISCV_GCC=${RISCV_INSTALL}/bin/riscv64-unknown-linux-gnu-g++
RISCV_LD_FLAGS=-lomp -lpthread -static -Wl,--no-as-needed -ldl

all: bfs.exe

riscv: raw.riscv.exe

cluster.bc: cluster.c 
	$(CC) $(CC_FLAGS) $^ -emit-llvm -c -o $@
	
getopt.bc: getopt.c 
	$(CC) $(CC_FLAGS) $^ -emit-llvm -c -o $@
	
kmeans.bc: kmeans.c 
	$(CC) $(CC_FLAGS) $^ -emit-llvm -c -o $@

kmeans_clustering.bc: kmeans_clustering.c
	$(CC) $(CC_FLAGS) $^ -emit-llvm -c -o $@

raw.bc: cluster.bc getopt.bc kmeans.bc kmeans_clustering.bc
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
