import subprocess as sp
import os

offset_bytes = ['0B'] + [
    f'{i}kB' for i in range(4, 68, 4)
]

makefile = """
CC=clang

include ../../GemForge.Makefile.include

riscv: raw.riscv.exe

%.bc: %.cpp
	$(CC) $(CC_FLAGS) -march=knl -fno-unroll-loops $^ -emit-llvm -c -o $@

raw.bc: pathfinder.bc
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
"""

for offset in offset_bytes:
    if offset.endswith('kB'):
        offset_value = int(offset[:-2]) * 1024
    else:
        offset_value = int(offset[:-1])

    dir_name = f'pathfinder-avx512-nounroll_offset_{offset}'
    sp.check_call(['mkdir', '-p', dir_name])
    with open(os.path.join(dir_name, f'pathfinder.cpp'), 'w') as f:
        f.write(f'#define OFFSET_BYTES {offset_value}\n')
        f.write(f'#include "../../pathfinder/pathfinder.cpp"')

    with open(os.path.join(dir_name, f'GemForge.Makefile'), 'w') as f:
        f.write(makefile)