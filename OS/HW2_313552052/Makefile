# indicating that target "all" and "clean" are not files
.PHONY: all clean

# set some variables
CC= gcc


SOURCE= sched_demo_313552052.c
OBJS= $(SOURCE:.c=.o)
DEPS= $(SOURCE:.c=.d)
TARGET= sched_demo_313552052

# first command of make
all: $(TARGET)

# import the dependencies of .h .c files from the compiler
-include $(DEPS)

# implicit targets
# %.o: %.c
#        $(CC) $^ -o $@ -c $(CFLAGS)

$(TARGET): $(SOURCE)
	$(CC) -lpthread $(SOURCE) -o $(TARGET)

clean:
	@rm -f $(TARGET) $(OBJS) $(DEPS)
