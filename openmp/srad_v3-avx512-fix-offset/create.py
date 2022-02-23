import subprocess as sp
import os

offset_bytes = ['0B'] + [
    f'{i}kB' for i in range(28, 32, 4)
]

makefile = """
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
"""

for offset in offset_bytes:
    if offset.endswith('kB'):
        offset_value = int(offset[:-2]) * 1024
    else:
        offset_value = int(offset[:-1])

    dir_name = f'srad_v3-avx512-fix_offset_{offset}'
    sp.check_call(['mkdir', '-p', dir_name])
    with open(os.path.join(dir_name, f'srad.cpp'), 'w') as f:
        f.write(f'#define OFFSET_BYTES {offset_value}\n')
        f.write(f'#include "../../srad_v3-avx512-fix/srad.cpp"')

    with open(os.path.join(dir_name, f'GemForge.Makefile'), 'w') as f:
        f.write(makefile)