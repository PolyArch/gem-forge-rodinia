
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# link objects(binaries) together
heartwall: main.o avilib.o avimod.o
	gcc $^ -lm -fopenmp -o $@

# compile main function file into object (binary)
main.o: main.c define.c kernel.c
	gcc $(OUTPUT) main.c -c -O3 -fopenmp

avilib.o: avilib.c
	gcc -c $^ -o $@

avimod.o: avimod.c
	gcc -c $^ -o $@

# delete all object files
clean:
	rm -f *.o heartwall
