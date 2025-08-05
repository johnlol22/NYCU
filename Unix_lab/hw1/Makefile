CC = gcc
CFLAGS = -Wall -Wextra -fPIC -g -pipe 
LDFLAGS = -shared -ldl

all: libzpoline.so.1 libzpoline.so.2 libzpoline.so libex3hook.so logger.so

libzpoline.so.1: zpoline1.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

libzpoline.so.2: zpoline2.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Current version
libzpoline.so: zpoline.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lcapstone

# Hook libraries
libex3hook.so: ex3hook.c
	$(CC) $(CFLAGS) -o $@ $< -shared

logger.so: logger.c
	$(CC) -fPIC -shared -ldl -o $@ $< 

clean:
	rm -f libzpoline.so.1 libzpoline.so.2 libzpoline.so libex3hook.so logger.so

.PHONY: all clean 