CXX=icpx
CXXFLAGS=-Wall -Wextra -std=c99 -O2 -xhost -qopt-report -fopenmp

LDFLAGS=-lm

all: quicksort.exe

quicksort.exe: quicksort.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf quicksort.exe *.o *opt *yaml heat
