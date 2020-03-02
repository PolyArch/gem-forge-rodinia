#makefile

openmp: particlefilter.c
	gcc -O3 -ffast-math -fopenmp $^ -o particle_filter -lm 


icc_openmp: particlefilter.c
	icc -O3 -fast -openmp $^ -o particle_filter


clean:
	rm particle_filter
