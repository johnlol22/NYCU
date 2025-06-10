CC = gcc
CFLAGS = -fPIC -Wall -shared
LDFLAGS = -ldl -Wl,-soname,libsolver.so

all: libsolver.so

libsolver.so: libsolver.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f libsolver.so

.PHONY: all clean