#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/cpufreq.h>
#include <linux/sysinfo.h>
#include <linux/utsname.h>
#include <linux/mm.h>
#include <linux/sched/signal.h>

#define DEVICE_NAME "kfetch"
#define CLASS_NAME "kfetch_class"
#define BUFFER_SIZE 4096

#define KFETCH_NUM_INFO 6
#define KFETCH_RELEASE (1 << 0)
#define KFETCH_NUM_CPUS (1 << 1)
#define KFETCH_CPU_MODEL (1 << 2)
#define KFETCH_MEM (1 << 3)
#define KFETCH_UPTIME (1 << 4)
#define KFETCH_NUM_PROCS (1 << 5)
#define KFETCH_FULL_INFO ((1 << KFETCH_NUM_INFO) - 1)

static int major_number;
static struct class *kfetch_class;
static struct cdev kfetch_cdev;
static char output_buffer[BUFFER_SIZE];
static unsigned int mask = KFETCH_FULL_INFO;
static DEFINE_MUTEX(kfetch_mutex);
static DEFINE_MUTEX(buffer_mutex);
static DEFINE_MUTEX(mask_mutex);

static const char *logo[] = {
    "        .-.      ",
    "       (.. |     ",
    "       <>  |     ",
    "      / --- \\    ",
    "     ( |   | )   ",
    "   |\\_)___(_/|   ",
    "  <__)-----(__>  ",
    NULL
};

static int get_cpu_model(char *buf, size_t size) {
    struct cpuinfo_x86 *c = &cpu_data(0);
    return snprintf(buf, size, "%s", c->x86_model_id);
}

static int format_system_info(char *buf, size_t size) {
    int len = 0;
    unsigned long total_mem, free_mem;
    char cpu_model[256];
    struct sysinfo si;
    int proc_count = 0;
    struct task_struct *task;
    char separator[256];
    int hostname_len;
    const int info_offset = 8;  // Logo width
    const int label_width = 8;  // Width for labels
    
    mutex_lock(&mask_mutex);
    unsigned int current_mask = mask;
    mutex_unlock(&mask_mutex);

    si_meminfo(&si);
    total_mem = si.totalram >> (20 - PAGE_SHIFT);
    free_mem = si.freeram >> (20 - PAGE_SHIFT);
    // Print hostname (mandatory) centered
    hostname_len = strlen(utsname()->nodename);
    len += snprintf(buf + len, size - len, "%*s%s\n", info_offset+label_width+1, "", utsname()->nodename);

    // Print the first two lines of the logo without info
    len += snprintf(buf + len, size - len, "%s", logo[0]);

    // Print hostname (mandatory) centered
    //hostname_len = strlen(utsname()->nodename);
    //len += snprintf(buf + len, size - len, "%*s%s\n", 
    //               (info_offset + label_width + hostname_len/2), "", utsname()->nodename);
    //len += snprintf(buf + len, size - len, "%s", logo[1]);
    // Print separator line (mandatory) centered
    memset(separator, '-', hostname_len);
    separator[hostname_len] = '\0';
    len += snprintf(buf + len, size - len, "%*s%s\n", 0,  "", separator);

    // Print the first two lines of the logo without info
    //len += snprintf(buf + len, size - len, "%s\n", logo[0]);
    //len += snprintf(buf + len, size - len, "%s\n", logo[1]);

    // Print remaining lines with info based on mask
    if (current_mask & KFETCH_RELEASE) {
        len += snprintf(buf + len, size - len, "%s%-*s%s\n", 
                       logo[1], label_width, "Kernel:", utsname()->release);
    } else {
        len += snprintf(buf + len, size - len, "%s\n", logo[1]);
    }

    if (current_mask & KFETCH_CPU_MODEL) {
        get_cpu_model(cpu_model, sizeof(cpu_model));
        len += snprintf(buf + len, size - len, "%s%-*s%s\n", 
                       logo[2], label_width, "CPU:", cpu_model);
    } else {
        len += snprintf(buf + len, size - len, "%s\n", logo[2]);
    }

    if (current_mask & KFETCH_NUM_CPUS) {
        len += snprintf(buf + len, size - len, "%s%-*s%d / %d\n", 
                       logo[3], label_width, "CPUs:", 
                       num_online_cpus(), num_present_cpus());
    } else {
        len += snprintf(buf + len, size - len, "%s\n", logo[3]);
    }

    if (current_mask & KFETCH_MEM) {
        len += snprintf(buf + len, size - len, "%s%-*s%lu MB / %lu MB\n", 
                       logo[4], label_width, "Mem:", 
                       free_mem, total_mem);
    } else {
        len += snprintf(buf + len, size - len, "%s\n", logo[4]);
    }

    if (current_mask & KFETCH_NUM_PROCS) {
        rcu_read_lock();
        for_each_process(task) proc_count++;
        rcu_read_unlock();
        len += snprintf(buf + len, size - len, "%s%-*s%d\n", 
                       logo[5], label_width, "Procs:", proc_count);
    } else {
        len += snprintf(buf + len, size - len, "%s\n", logo[5]);
    }

    if (current_mask & KFETCH_UPTIME) {
        len += snprintf(buf + len, size - len, "%*s%-*s%lu mins\n", 
                       info_offset, logo[6], label_width, "Uptime:", si.uptime / 60);
    }else {
        len += snprintf(buf + len, size - len, "%s\n", logo[6]);
    }


    return len;
}

static int kfetch_open(struct inode *inode, struct file *file) {
    if (!mutex_trylock(&kfetch_mutex))
        return -EBUSY;
    return 0;
}

static int kfetch_release(struct inode *inode, struct file *file) {
    mutex_unlock(&kfetch_mutex);
    return 0;
}

static ssize_t kfetch_read(struct file *file, char __user *user_buffer,
                          size_t size, loff_t *offset) {
    static char *buffer_ptr;
    static int buffer_size;
    int bytes_to_copy;
    mutex_lock(&buffer_mutex);
    if (*offset == 0) {
        buffer_size = format_system_info(output_buffer, BUFFER_SIZE);
        buffer_ptr = output_buffer;
    }

    if (*offset >= buffer_size){
	mutex_unlock(&buffer_mutex);
        return 0;
    }

    bytes_to_copy = min(size, (size_t)(buffer_size - *offset));
    
    if (copy_to_user(user_buffer, buffer_ptr + *offset, bytes_to_copy)){
	mutex_unlock(&buffer_mutex);
        return -EFAULT;
    }

    *offset += bytes_to_copy;
    mutex_unlock(&buffer_mutex);
    return bytes_to_copy;
}

static ssize_t kfetch_write(struct file *file, const char __user *user_buffer,
                           size_t size, loff_t *offset) {
    int new_mask;

    if (size != sizeof(int))
        return -EINVAL;
    mutex_lock(&mask_mutex);
    if (copy_from_user(&new_mask, user_buffer, sizeof(int))){
        mutex_unlock(&mask_mutex);
	return -EFAULT;
    }

    if (new_mask < 0 || new_mask >= (1 << KFETCH_NUM_INFO)){
        mutex_unlock(&mask_mutex);
	return -EINVAL;
    }

    mask = new_mask;
    return size;
}

static const struct file_operations kfetch_ops = {
    .owner = THIS_MODULE,
    .read = kfetch_read,
    .write = kfetch_write,
    .open = kfetch_open,
    .release = kfetch_release
};

static int __init kfetch_init(void) {
    dev_t dev;
    int err;

    err = alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME);
    if (err < 0)
        return err;

    major_number = MAJOR(dev);

    cdev_init(&kfetch_cdev, &kfetch_ops);
    err = cdev_add(&kfetch_cdev, dev, 1);
    if (err < 0) {
        unregister_chrdev_region(dev, 1);
        return err;
    }

    kfetch_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(kfetch_class)) {
        cdev_del(&kfetch_cdev);
        unregister_chrdev_region(dev, 1);
        return PTR_ERR(kfetch_class);
    }

    if (IS_ERR(device_create(kfetch_class, NULL, dev, NULL, DEVICE_NAME))) {
        class_destroy(kfetch_class);
        cdev_del(&kfetch_cdev);
        unregister_chrdev_region(dev, 1);
        return PTR_ERR(kfetch_class);
    }

    return 0;
}

static void __exit kfetch_exit(void) {
    device_destroy(kfetch_class, MKDEV(major_number, 0));
    class_destroy(kfetch_class);
    cdev_del(&kfetch_cdev);
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
}

module_init(kfetch_init);
module_exit(kfetch_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Kernel Fetch Information Module");
MODULE_VERSION("1.0");