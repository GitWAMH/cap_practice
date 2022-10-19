CC=icx
CFLAGS=-Wall -Wextra -std=c99 -O2 -xhost -qopt-report -fopenmp

LDFLAGS=-lm

all: heat

heat: heat.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf *.o *opt *yaml heat
