MODULE_NAME = kfetch_mod_313552052
obj-m := $(MODULE_NAME).o

# Get kernel build directory
KDIR := /lib/modules/$(shell uname -r)/build
# Current directory
PWD := $(shell pwd)

# Default target
all: modules

# Build the module
modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean build files
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f modules.order Module.symvers

# Load the module
load:
	-sudo rmmod $(MODULE_NAME) 2>/dev/null || true
	sudo insmod $(MODULE_NAME).ko

# Unload the module
unload:
	-sudo rmmod $(MODULE_NAME) 2>/dev/null || true

# Install module (optional)
install:
	sudo $(MAKE) -C $(KDIR) M=$(PWD) modules_install
	sudo depmod -a

# Declare the phony targets
.PHONY: all clean load unload install modules