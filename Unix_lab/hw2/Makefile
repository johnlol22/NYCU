CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99
LDFLAGS = -lcapstone

TARGET = sdb
SOURCES = sdb.c

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/

test: $(TARGET)
	@echo "Testing with a simple program..."
	@echo "int main() { return 42; }" > test.c
	@gcc -o test test.c
	@echo "load ./test" | ./$(TARGET)
	@rm -f test test.c

help:
	@echo "Available targets:"
	@echo "  all       - Build the debugger"
	@echo "  clean     - Remove built files"
	@echo "  install   - Install to /usr/local/bin"
	@echo "  deps      - Install dependencies (Ubuntu/Debian)"
	@echo "  deps-rpm  - Install dependencies (CentOS/RHEL/Fedora)"
	@echo "  test      - Run a simple test"
	@echo "  help      - Show this help message"