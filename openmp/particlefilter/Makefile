#makefile

openmp: particlefilter.c
	gcc -O3 -ffast-math -fopenmp $^ -o particlefilter.exe -lm 


icc_openmp: particlefilter.c
	icc -O3 -fast -openmp $^ -o particlefilter.exe


clean:
	rm particlefilter.exe
