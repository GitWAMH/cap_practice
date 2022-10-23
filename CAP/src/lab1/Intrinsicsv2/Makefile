CC   = icx

EXE   = main 

SOURCES    = main.c


OBJS    = stencil.o pngio.o main.o 

CFLAGS     = -O2 -qopt-report=5
NVCFLAGS   = $(CFLAGS)

LIBS =  -lm -lpng

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? -I. $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I.
clean:
	rm -f $(OBJS) $(EXE)
