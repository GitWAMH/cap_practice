CXX=icpx
CXXFLAGS=-Wall -Wextra -std=c99 -O2 -xhost -qopt-report -fopenmp

LDFLAGS=-lm

all: quicksort

quicksort: quicksort.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf quicksort *.o *opt *yaml heat
