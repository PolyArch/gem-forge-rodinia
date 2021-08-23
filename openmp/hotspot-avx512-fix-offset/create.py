import subprocess as sp
import os

offset_bytes = ['0B'] + [
    f'{i}kB' for i in range(4, 68, 4)
]

makefile = """
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
"""

for offset in offset_bytes:
    if offset.endswith('kB'):
        offset_value = int(offset[:-2]) * 1024
    else:
        offset_value = int(offset[:-1])

    dir_name = f'hotspot-avx512-fix_offset_{offset}'
    sp.check_call(['mkdir', '-p', dir_name])
    with open(os.path.join(dir_name, f'hotspot.cpp'), 'w') as f:
        f.write(f'#define OFFSET_BYTES {offset_value}\n')
        f.write(f'#include "../../hotspot/hotspot.cpp"')

    with open(os.path.join(dir_name, f'GemForge.Makefile'), 'w') as f:
        f.write(makefile)