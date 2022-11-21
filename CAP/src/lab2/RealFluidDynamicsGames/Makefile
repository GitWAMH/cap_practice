CC=icx
CFLAGS=-O2 -qopt-report=1 -qopt-report -std=c99 -Wall -Wextra -Wno-unused-parameter -qopenmp
LDFLAGS=

TARGETS=demo.exe headless.exe
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

all: $(TARGETS)

demo.exe: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless.exe: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o *.opt.yaml


.PHONY: clean all
