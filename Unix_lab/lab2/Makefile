
obj-m     += cryptomod.o
ccflags-y += -DEXPORT_SYMTAB
all:
	make -C ./dist/modulebuild M=$(PWD) modules

clean:
	make -C ./dist/modulebuild M=$(PWD) clean

install: all
	mkdir -p ./rootfs/modules
	cp fun.jpg.enc test_crypto cryptomod.ko ./rootfs/modules
