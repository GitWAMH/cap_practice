# ==============================================================================
NAME=black-scholes

SRC:=                      \
       black-scholes.c     \
       black-scholes_mkl.c \
       data_gen.c          \
       main.c

# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := icc
CFLAGS      += -O2 -g -qopt-report=3
CFLAGS      += -I./

PREC ?= s
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

TARGET_ARCH ?= AVX

ifeq ($(TARGET_ARCH),SSE)
    CFLAGS += -$(QOPT)xSSE4.2
endif
ifeq ($(TARGET_ARCH),AVX)
    CFLAGS += -$(QOPT)xAVX
endif
ifeq ($(TARGET_ARCH),AVX2)
    CFLAGS += -$(QOPT)xCORE_AVX2
endif

ACC ?= ep
ifeq ($(ACC),ha)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)high -D_VML_ACCURACY_HA_
endif
ifeq ($(ACC),la)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)medium -D_VML_ACCURACY_LA_
endif
ifeq ($(ACC),ep)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)low -$(FQOPT)imf-domain-exclusion$(EQCOLON)31 -D_VML_ACCURACY_EP_
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

$(NAME): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -mkl -o $(NAME)


run: $(NAME)
	./black-scholes 10000000

clean:
	rm -rf $(NAME) *.o *.out *.optrpt
