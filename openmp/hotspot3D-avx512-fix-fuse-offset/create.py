import subprocess as sp
import os

offset_bytes = ['0B'] + [
    f'{i}kB' for i in range(4, 68, 4)
]

makefile = """
CC=clang

include ../../GemForge.Makefile.include

all: bfs.exe

riscv: raw.riscv.exe

3D.bc: 3D.c
	$(CC) $(CC_FLAGS) -march=knl -DFIX_ROW=256 -DFIX_COL=1024 -DFUSE_OUTER_LOOPS $^ -emit-llvm -c -o $@

raw.bc: 3D.bc
	llvm-link $^ -o $@
	opt -instnamer $@ -o $@

%.ll: %.bc
	llvm-dis $< -o $@

clean:
	rm -f *.exe *.bc *.ll *.o result.txt
"""

for offset in offset_bytes:
    if offset.endswith('kB'):
        offset_value = int(offset[:-2]) * 1024
    else:
        offset_value = int(offset[:-1])

    dir_name = f'hotspot3D-avx512-fix-fuse_offset_{offset}'
    sp.check_call(['mkdir', '-p', dir_name])
    with open(os.path.join(dir_name, f'3D.c'), 'w') as f:
        f.write(f'#define OFFSET_BYTES {offset_value}\n')
        f.write(f'#include "../../hotspot3D/3D.c"')

    with open(os.path.join(dir_name, f'GemForge.Makefile'), 'w') as f:
        f.write(makefile)
