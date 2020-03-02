LOCAL_CC = gcc -g -O3 -Wall
CC = gcc
CFLAGS = -lm -fopenmp -Wall


all: nn.exe

clean:
	rm -rf *.o nn.exe hurricane_gen.exe

nn.exe: nn.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen.exe: hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

