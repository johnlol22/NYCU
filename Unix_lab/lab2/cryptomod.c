#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/crypto.h>
#include <crypto/skcipher.h>
#include <crypto/aes.h>
#include <linux/scatterlist.h>
#include <linux/string.h>
#include <linux/atomic.h>
#include "cryptomod.h"

#define DEVICE_NAME "cryptodev"     // the name of the device file created in the dev directory
#define CLASS_NAME "crypto"         // the name of the device class in sysfs filesystem
#define PROC_ENTRY "cryptomod"      // the name of the entry created in the proc filesystem

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Kernel Developer");
MODULE_DESCRIPTION("AES ECB Encryption/Decryption Module");
MODULE_VERSION("1.0");

// Function prototypes
static int crypto_open(struct inode *, struct file *);
static int crypto_release(struct inode *, struct file *);
static ssize_t crypto_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t crypto_write(struct file *, const char __user *, size_t, loff_t *);
static long crypto_ioctl(struct file *, unsigned int, unsigned long);
static int crypto_proc_show(struct seq_file *, void *);
static int crypto_proc_open(struct inode *, struct file *);
static void crypto_completion(void *data, int error);

// Device structure for per-file private data
struct crypto_dev {                 // maintains the state of each file descriptor to the crypto device
    struct mutex lock;
    struct crypto_setup {           // encryption/decryption configuration
        char key[CM_KEY_MAX_LEN];
        int key_len;                // 16 24 or 32
        enum IOMode io_mode;        // basic or adv
        enum CryptoMode c_mode;     // enc or dec
        bool is_setup;              // device has been configured?
        bool is_finalized;          // finalization has been applied? (padding added/removed)
    } setup;

    struct crypto_skcipher *tfm;    // transform object (algorithm)
    struct skcipher_request *req;   // request structure for crypto operation
    
    // Buffers for data processing
    unsigned char *in_buffer;
    unsigned char *out_buffer;
    size_t in_buffer_len;
    size_t out_buffer_len;
    size_t in_buffer_pos;
    size_t out_buffer_pos;
};

// Global variables
static int major_number;                        // major device number that identifies the device type to the kernel
static struct class *crypto_class = NULL;
static struct device *crypto_device = NULL;
static struct proc_dir_entry *proc_entry = NULL;
static struct cdev cdev;                        // the character device structure that connects file operation (open, read...) to your device driver functions

// Kernel stats
static atomic_t total_bytes_read = ATOMIC_INIT(0);      // atomic counter tracking the total number of bytes read from the device by user program
static atomic_t total_bytes_written = ATOMIC_INIT(0);   // atomic counter tracking the total number of bytes written from the device by user program
static atomic_t byte_frequency[256] = {};               // Global frequency counter for each byte (0-255) in encrypted data

// Global mutex for protecting shared resources
DEFINE_MUTEX(global_lock);      // global_lock defined in this macro

// File operations
static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = crypto_open,
    .release = crypto_release,
    .read = crypto_read,
    .write = crypto_write,
    .unlocked_ioctl = crypto_ioctl,
};

// Proc file operations
static const struct proc_ops proc_fops = {
    .proc_open = crypto_proc_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

// Callback for crypto operations
// callback for asynchronous cryptographic operations
// error means completion status, data means user data pointer
static void crypto_completion(void *data, int error)
{
    struct crypto_wait *wait = data;
    
    if (error == -EINPROGRESS)      // check if the operation is still in progress
        return;
    
    wait->err = error;
    complete(&wait->completion);    // wake up other waiting threads, like a semaphore
}

// AES encryption/decryption function, u8 means unsigned 8 bits int
static int aes_ecb_crypt(struct crypto_dev *dev, u8 *dst, const u8 *src, 
                        unsigned int nbytes, enum CryptoMode mode)
{
    int ret;
    // scatterlist is a data structure that allow operations on non-contigous memory area as if they were the single buffer
    struct scatterlist sg_in, sg_out;
    DECLARE_CRYPTO_WAIT(wait);      // initialize
    
    // Ensure nbytes is a multiple of block size
    if (nbytes % CM_BLOCK_SIZE != 0) {
        printk(KERN_ERR "aes_ecb_crypt: Invalid data length %u (not a multiple of %d)\n", 
               nbytes, CM_BLOCK_SIZE);
        return -EINVAL;
    }
    
    printk(KERN_DEBUG "Input data: %*ph\n", min(16, (int)nbytes), src);
    
    // Initialize scatterlists
    // initialize a single entry scatterlist
    // sg_in point to the memory at src, nbytes: len of memory area
    sg_init_one(&sg_in, src, nbytes);
    sg_init_one(&sg_out, dst, nbytes);
    
    // Set up the request
    // the function allows setting the callback function that is triggered once the cipher operation completes
    skcipher_request_set_callback(dev->req, CRYPTO_TFM_REQ_MAY_BACKLOG |        // MAY_BACKLOG: the req can be queued if resources are temporarily unavailable
                                 CRYPTO_TFM_REQ_MAY_SLEEP,                      // MAY_SLEEP: the operation is allowed to sleep (e.g. malloc)
                                 crypto_completion, &wait);                     // wait: user data (a pointer to wait structure) (defined in aes_ecb_crypt())
    // setting the source data and destination data scatterlist
    // ENC: source is plaintext, dest is cipher
    // DEC: source is cipher, dest is plaintext
    skcipher_request_set_crypt(dev->req, &sg_in, &sg_out, nbytes, NULL);
    
    // Perform encryption/decryption
    if (mode == ENC)
        ret = crypto_wait_req(crypto_skcipher_encrypt(dev->req), &wait);        // synchronously waits for asynchronous crypto operation to complete
    else // mode == DEC                                                         // it takes the value of crypto operation, like crypto_skcipher_encrypt()
        ret = crypto_wait_req(crypto_skcipher_decrypt(dev->req), &wait);        // takes a wait structure with completion info, return operation status after completion
    
    if (ret == 0) {
        printk(KERN_DEBUG "Output data: %*ph\n", 16, dst);
    }
    
    return ret;
}

// PKCS#7 padding functions
static int add_pkcs7_padding(unsigned char *buf, size_t data_len, size_t buf_size)
{
    size_t pad_len;
    size_t i;
    
    if (buf_size < data_len)
        return -EINVAL;
        
    pad_len = CM_BLOCK_SIZE - (data_len % CM_BLOCK_SIZE);
    
    // If data_len is already a multiple of block size, add a full block
    if (pad_len == 0)
        pad_len = CM_BLOCK_SIZE;
        
    if (data_len + pad_len > buf_size) {
        printk(KERN_ERR "Not enough space for padding: data_len=%zu, pad_len=%zu, buf_size=%zu\n",
              data_len, pad_len, buf_size);
        return -EINVAL;
    }
        
    // Add padding bytes
    for (i = 0; i < pad_len; i++)
        buf[data_len + i] = (unsigned char)pad_len;
        
    return data_len + pad_len;
}

static int remove_pkcs7_padding(unsigned char *buf, size_t data_len)
{
    unsigned char pad_len;
    size_t i;
    // data not aligned
    if (data_len == 0 || data_len % CM_BLOCK_SIZE != 0)
        return -EINVAL;
        
    pad_len = buf[data_len - 1];
    
    if (pad_len > CM_BLOCK_SIZE || pad_len == 0)
        return -EINVAL;
        
    // Verify padding bytes
    for (i = data_len - pad_len; i < data_len; i++) {
        if (buf[i] != pad_len)
            return -EINVAL;
    }
    
    return data_len - pad_len;
}

// Helper function to setup crypto cipher
static int setup_crypto_cipher(struct crypto_dev *dev)
{
    if (!dev->setup.is_setup)
        return -EINVAL;
        
    // Free existing cipher and request if any
    if (dev->req) {
        // free request data structure
        skcipher_request_free(dev->req);
        dev->req = NULL;
    }
    
    if (dev->tfm) {
        // free cipher handle
        crypto_free_skcipher(dev->tfm);
        dev->tfm = NULL;
    }
    
    // Create new cipher
    dev->tfm = crypto_alloc_skcipher("ecb(aes)", 0, 0); // algo name, mask, flags
    if (IS_ERR(dev->tfm)) {
        printk(KERN_ERR "Failed to allocate skcipher\n");
        return PTR_ERR(dev->tfm);
    }
    
    // Set key
    // configure the enc/dec key for cipher, associating the key with the transformation object, performing key scheduling (transform user key into round key)
    if (crypto_skcipher_setkey(dev->tfm, dev->setup.key, dev->setup.key_len)) {
        printk(KERN_ERR "Failed to set key\n");
        crypto_free_skcipher(dev->tfm);
        dev->tfm = NULL;
        return -EINVAL;
    }
    
    // Print key for debugging
    printk(KERN_DEBUG "Key being used (len=%d): %*ph\n", 
           dev->setup.key_len, dev->setup.key_len, dev->setup.key);
    
    // Create request
    // GFP_KERNEL means it can sleep if needed
    dev->req = skcipher_request_alloc(dev->tfm, GFP_KERNEL);
    if (!dev->req) {
        printk(KERN_ERR "Failed to allocate request\n");
        crypto_free_skcipher(dev->tfm);
        dev->tfm = NULL;
        return -ENOMEM;
    }
    
    return 0;
}

// Helper function to clean up buffers
static void cleanup_buffers(struct crypto_dev *dev)
{
    if (dev->in_buffer) {
        memset(dev->in_buffer, 0, dev->in_buffer_len);
        dev->in_buffer_pos = 0;
    }
    
    if (dev->out_buffer) {
        memset(dev->out_buffer, 0, dev->out_buffer_len);
        dev->out_buffer_pos = 0;
    }
}

// Process data for encryption/decryption
static ssize_t process_data(struct crypto_dev *dev)
{
    ssize_t ret = 0;
    size_t bytes_processed = 0;
    size_t processed_block_count = 0;
    
    // Check if device is set up
    if (!dev->setup.is_setup)
        return -EINVAL;
    
    // Different behavior based on I/O mode
    switch (dev->setup.io_mode) {
        case BASIC:
            // In BASIC mode, we only process data when finalized
            if (!dev->setup.is_finalized)
                return 0;  // Return 0 means no data processed, but it's not an error
            
            // Process input data in chunks of up to 1024 bytes
            while (dev->in_buffer_pos > 0) {
                size_t bytes_to_process = min(dev->in_buffer_pos, (size_t)1024);
                
                // AES requires data to be a multiple of the block size
                bytes_to_process = (bytes_to_process / CM_BLOCK_SIZE) * CM_BLOCK_SIZE;
                if (bytes_to_process == 0)
                    break;
                
                // Ensure output buffer has enough space
                if (dev->out_buffer_len < dev->out_buffer_pos + bytes_to_process) {
                    // CM_BLOCK_SIZE provides a safety margin
                    size_t new_size = dev->out_buffer_len + bytes_to_process + CM_BLOCK_SIZE;
                    unsigned char *new_buffer = krealloc(dev->out_buffer, new_size, GFP_KERNEL);
                    if (!new_buffer)
                        return -ENOMEM;
                    
                    dev->out_buffer = new_buffer;
                    dev->out_buffer_len = new_size;
                }
                
                // Perform encryption/decryption
                ret = aes_ecb_crypt(
                    dev,
                    dev->out_buffer + dev->out_buffer_pos,
                    dev->in_buffer,
                    bytes_to_process, 
                    dev->setup.c_mode
                );
                
                if (ret < 0)
                    return ret;
                
                // Update positions and counters
                dev->out_buffer_pos += bytes_to_process;
                
                // Move remaining data to the beginning of the input buffer
                // in_buffer_pos tracks total amount of data in input buffer
                // works at both enc/dec, moves any unprocessed partial data to the begining of input buffer
                if (bytes_to_process < dev->in_buffer_pos) {
                    memmove(dev->in_buffer, 
                            dev->in_buffer + bytes_to_process,
                            dev->in_buffer_pos - bytes_to_process);
                }
                dev->in_buffer_pos -= bytes_to_process;
                bytes_processed += bytes_to_process;
                processed_block_count++;
                
                // Update byte frequency for statistical analysis
                // represent the global frequency of each byte in the data encrypted and read by user programs
                if (dev->setup.c_mode == ENC){
                    mutex_lock(&global_lock);
                    {
                        unsigned int i;
                        for (i = 0; i < bytes_to_process; i++) {
                            unsigned char byte = dev->out_buffer[dev->out_buffer_pos - bytes_to_process + i];
                            atomic_inc(&byte_frequency[byte]);
                        }
                    }
                    mutex_unlock(&global_lock);
                }
            }
            break;
            
        case ADV:
            // In ADV mode, we process complete blocks incrementally
            // key difference with basic mode is that there is a min() limitation in basic mode
            if (dev->setup.c_mode == ENC) {
                // Encryption mode
                // Process complete blocks only (except when finalized)
                size_t block_size = CM_BLOCK_SIZE;
                size_t complete_blocks = dev->in_buffer_pos / block_size;
                
                if (complete_blocks == 0 && !dev->setup.is_finalized)
                    return 0;  // Not enough data to process a block
                
                size_t bytes_to_process = complete_blocks * block_size;
                
                // If finalized, process all remaining bytes (padding was already added in ioctl (in finalize)) 
                if (dev->setup.is_finalized)
                    bytes_to_process = dev->in_buffer_pos;
                
                // Ensure output buffer has enough space
                if (dev->out_buffer_len < dev->out_buffer_pos + bytes_to_process + block_size) {
                    size_t new_size = dev->out_buffer_len + bytes_to_process + block_size;
                    unsigned char *new_buffer = krealloc(dev->out_buffer, new_size, GFP_KERNEL);
                    if (!new_buffer)
                        return -ENOMEM;
                    
                    dev->out_buffer = new_buffer;
                    dev->out_buffer_len = new_size;
                }
                
                // Process data only if we have some to process
                if (bytes_to_process > 0) {
                    // AES requires data to be a multiple of the block size
                    bytes_to_process = (bytes_to_process / CM_BLOCK_SIZE) * CM_BLOCK_SIZE;
                    if (bytes_to_process == 0)
                        return 0;
                    
                    ret = aes_ecb_crypt(
                        dev,
                        dev->out_buffer + dev->out_buffer_pos,
                        dev->in_buffer,
                        bytes_to_process, 
                        dev->setup.c_mode
                    );
                    
                    if (ret < 0)
                        return ret;
                    
                    // Update positions and counters
                    dev->out_buffer_pos += bytes_to_process;
                    
                    // Move remaining data to the beginning of the input buffer
                    if (bytes_to_process < dev->in_buffer_pos) {
                        memmove(dev->in_buffer, 
                                dev->in_buffer + bytes_to_process,
                                dev->in_buffer_pos - bytes_to_process);
                    }
                    dev->in_buffer_pos -= bytes_to_process;
                    bytes_processed += bytes_to_process;
                    processed_block_count += bytes_to_process / block_size;
                    
                    // Update byte frequency
                    // represent the global frequency of each byte in the data encrypted and read by user programs
                    mutex_lock(&global_lock);
                    {
                        unsigned int i;
                        for (i = 0; i < bytes_to_process; i++) {
                            unsigned char byte = dev->out_buffer[dev->out_buffer_pos - bytes_to_process + i];
                            atomic_inc(&byte_frequency[byte]);
                        }
                    }
                    mutex_unlock(&global_lock);
                }
            } else {
                // Decryption mode
                size_t block_size = CM_BLOCK_SIZE;
                
                // In decryption, we always withhold one block until finalization
                size_t complete_blocks = dev->in_buffer_pos / block_size;
                
                if (complete_blocks <= 1 && !dev->setup.is_finalized)   // when finalize is called, the last block can be released
                    return 0;  // Not enough data to process (need at least 2 blocks in non-finalized state)
                
                // Process all blocks except the last one (unless finalized)
                size_t blocks_to_process = dev->setup.is_finalized ? complete_blocks : complete_blocks - 1;
                size_t bytes_to_process = blocks_to_process * block_size;
                
                // Ensure output buffer has enough space
                if (dev->out_buffer_len < dev->out_buffer_pos + bytes_to_process) {
                    size_t new_size = dev->out_buffer_len + bytes_to_process + block_size;
                    unsigned char *new_buffer = krealloc(dev->out_buffer, new_size, GFP_KERNEL);
                    if (!new_buffer)
                        return -ENOMEM;
                    
                    dev->out_buffer = new_buffer;
                    dev->out_buffer_len = new_size;
                }
                
                // Process data only if we have some to process
                if (bytes_to_process > 0) {
                    ret = aes_ecb_crypt(
                        dev,
                        dev->out_buffer + dev->out_buffer_pos,
                        dev->in_buffer,
                        bytes_to_process, 
                        dev->setup.c_mode
                    );
                    
                    if (ret < 0)
                        return ret;
                    
                    // Update positions and counters
                    dev->out_buffer_pos += bytes_to_process;
                    
                    // Move remaining data to the beginning of the input buffer
                    if (bytes_to_process < dev->in_buffer_pos) {
                        memmove(dev->in_buffer, 
                                dev->in_buffer + bytes_to_process,
                                dev->in_buffer_pos - bytes_to_process);
                    }
                    dev->in_buffer_pos -= bytes_to_process;
                    bytes_processed += bytes_to_process;
                    processed_block_count += blocks_to_process;
                    
                }
            }
            break;
            
        default:
            return -EINVAL;
    }
    
    if (processed_block_count > 0) {
        return bytes_processed;
    } else if (dev->setup.io_mode == ADV && 
               ((dev->setup.c_mode == ENC && dev->in_buffer_pos < CM_BLOCK_SIZE) ||        // for basic mode, allowing partial block processing during encryption
                (dev->setup.c_mode == DEC && dev->in_buffer_pos <= CM_BLOCK_SIZE))) {      // for decryption, ensuring at least one full block is required, preventing processing incomplete blocks
        return -EAGAIN;  // Need more data                                                 // guarantees enough data for correct padding removal and decryption
    } else {
        return 0;  // No blocks processed but not an error
    }
}

// Device open, initialize a crypto device
static int crypto_open(struct inode *inode, struct file *file)
{
    if (!inode || !file)
        return -EINVAL;
    struct crypto_dev *dev;
    
    // Allocate private data for this file descriptor
    dev = kzalloc(sizeof(struct crypto_dev), GFP_ATOMIC);
    if (!dev)
        return -ENOMEM;
    
    // Initialize per-file state
    mutex_init(&dev->lock);
    dev->setup.is_setup = false;
    dev->setup.is_finalized = false;
    dev->tfm = NULL;
    dev->req = NULL;
    
    // Allocate initial buffers
    dev->in_buffer = kzalloc(CM_BLOCK_SIZE * 2, GFP_ATOMIC);
    if (!dev->in_buffer) {
        kfree(dev);
        return -ENOMEM;
    }
    dev->in_buffer_len = CM_BLOCK_SIZE * 2;
    
    dev->out_buffer = kzalloc(CM_BLOCK_SIZE * 2, GFP_ATOMIC);
    if (!dev->out_buffer) {
        kfree(dev->in_buffer);
        kfree(dev);
        return -ENOMEM;
    }
    dev->out_buffer_len = CM_BLOCK_SIZE * 2;
    
    file->private_data = dev;
    return 0;
}

// Device release
// cleanup when user closed the device file
static int crypto_release(struct inode *inode, struct file *file)
{
    struct crypto_dev *dev = file->private_data;
    
    if (dev) {
        // Clean up crypto resources
        if (dev->req) {
            skcipher_request_free(dev->req);
        }
        
        if (dev->tfm) {
            crypto_free_skcipher(dev->tfm);
        }
        
        // Free buffers
        kfree(dev->in_buffer);
        kfree(dev->out_buffer);
        
        // Free the device structure
        kfree(dev);
    }
    
    return 0;
}

// Device read
// when user program reads from the device
static ssize_t crypto_read(struct file *file, char __user *buf, size_t count, loff_t *offset)
{
    struct crypto_dev *dev = file->private_data;
    ssize_t ret = 0;
    size_t bytes_to_copy;
    
    if (count == 0)
        return 0;
    
    if (mutex_lock_interruptible(&dev->lock))   // acquire the mutex lock
        return -ERESTARTSYS;                    // similar to mutex_lock(), but if sleeping, can be interrupted by the signal
    
    // Check if the device is set up
    if (!dev->setup.is_setup) {
        mutex_unlock(&dev->lock);
        return -EINVAL;
    }
    
    // Process any pending data
    ret = process_data(dev);                // even the bytes to process is less than block size, which means won't be processed immediately
    if (ret < 0 && ret != -EAGAIN) {        // after receiving finalize or other bytes to compose into a complete block, the data will be processed
        mutex_unlock(&dev->lock);
        return ret;
    }
    
    // Check if there's data to read
    if (dev->out_buffer_pos == 0) {
        if (!dev->setup.is_finalized) {
            mutex_unlock(&dev->lock);
            return -EAGAIN;                 // try again later
        } else {
            // No more data and finalized
            mutex_unlock(&dev->lock);
            return 0;
        }
    }
    
    // Calculate how many bytes to copy
    bytes_to_copy = min(count, dev->out_buffer_pos);            // minimum of requested and available (e.g. out_buffer_pos)
    
    // Copy to user
    if (copy_to_user(buf, dev->out_buffer, bytes_to_copy)) {
        mutex_unlock(&dev->lock);
        return -EFAULT;
    }
    
    // Update statistics
    atomic_add(bytes_to_copy, &total_bytes_read);
    
    // Move remaining data to the beginning of the output buffer
    if (bytes_to_copy < dev->out_buffer_pos) {
        memmove(dev->out_buffer, 
                dev->out_buffer + bytes_to_copy,
                dev->out_buffer_pos - bytes_to_copy);
    }
    dev->out_buffer_pos -= bytes_to_copy;
    
    mutex_unlock(&dev->lock);
    return bytes_to_copy;
}

// Device write
static ssize_t crypto_write(struct file *file, const char __user *buf, size_t count, loff_t *offset)
{
    struct crypto_dev *dev = file->private_data;
    ssize_t ret = 0;
    
    if (count == 0)
        return 0;
    
    if (mutex_lock_interruptible(&dev->lock))
        return -ERESTARTSYS;
    
    // Check if the device is set up
    if (!dev->setup.is_setup) {
        mutex_unlock(&dev->lock);
        return -EINVAL;
    }
    
    // Check if the device is finalized
    if (dev->setup.is_finalized) {
        mutex_unlock(&dev->lock);
        return -EINVAL;
    }
    
    // Ensure input buffer is large enough
    if (dev->in_buffer_len < dev->in_buffer_pos + count) {
        size_t new_size = dev->in_buffer_len + count + CM_BLOCK_SIZE;
        unsigned char *new_buffer = krealloc(dev->in_buffer, new_size, GFP_KERNEL);
        if (!new_buffer) {
            mutex_unlock(&dev->lock);
            return -ENOMEM;
        }
        dev->in_buffer = new_buffer;
        dev->in_buffer_len = new_size;
    }
    
    // Copy data from user
    if (copy_from_user(dev->in_buffer + dev->in_buffer_pos, buf, count)) { 
        mutex_unlock(&dev->lock);
        return -EFAULT;
    }
    
    // Update statistics
    atomic_add(count, &total_bytes_written);
    dev->in_buffer_pos += count;
    
    // Process data (only complete blocks)
    ret = process_data(dev);                // partial block remains in buffer for future processing
    
    mutex_unlock(&dev->lock);
    
    // If no data was processed (ret == 0) and there's an error, return the error
    
    if (ret < 0 && ret != -EAGAIN)
        return ret;
    
    return count;
}

// Device ioctl
static long crypto_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct crypto_dev *dev = file->private_data;
    struct CryptoSetup setup;
    int ret = 0;
    
    if (mutex_lock_interruptible(&dev->lock))
        return -ERESTARTSYS;
    
    switch (cmd) {
        case CM_IOC_SETUP:
            // Copy setup from user (dst, src, count)
            if (copy_from_user(&setup, (struct CryptoSetup __user *)arg, sizeof(setup))) {
                ret = -EINVAL;
                break;
            }

            // Validate key length
            if (setup.key_len != 16 && setup.key_len != 24 && setup.key_len != 32) {
                ret = -EINVAL;
                break;
            }

            // Validate crypto mode enum (ENC or DEC)
            if (setup.c_mode != ENC && setup.c_mode != DEC) {
                printk(KERN_DEBUG "Invalid crypto mode: %d\n", setup.c_mode);
                ret = -EINVAL;
                break;
            }

            // Validate I/O mode enum (BASIC or ADV)
            if (setup.io_mode != BASIC && setup.io_mode != ADV) {
                printk(KERN_DEBUG "Invalid I/O mode: %d\n", setup.io_mode);
                ret = -EINVAL;
                break;
            }

            printk(KERN_DEBUG "Crypto setup: mode=%d, key_len=%d, key=%*ph\n", 
                   setup.c_mode, setup.key_len, min(16, setup.key_len), setup.key);
            
            // Clear existing setup and buffers
            cleanup_buffers(dev);
            
            // Copy new setup (dst, src, len)
            memcpy(&dev->setup.key, setup.key, setup.key_len);
            dev->setup.key_len = setup.key_len;
            dev->setup.io_mode = setup.io_mode;
            dev->setup.c_mode = setup.c_mode;
            dev->setup.is_setup = true;
            dev->setup.is_finalized = false;
            
            // Setup crypto cipher
            ret = setup_crypto_cipher(dev);         // setup transformation obj, key, request
            if (ret < 0)
                dev->setup.is_setup = false;
            
            break;
            
        case CM_IOC_FINALIZE:
            // Add debug logging
            printk(KERN_DEBUG "FINALIZE: mode=%d, input_size=%zu, is_setup=%d, is_finalized=%d\n", 
                   dev->setup.c_mode, dev->in_buffer_pos, dev->setup.is_setup, dev->setup.is_finalized);
            
            // Check if the device is set up
            if (!dev->setup.is_setup) {
                ret = -EINVAL;
                break;
            }
            
            // Check if already finalized
            if (dev->setup.is_finalized) {
                ret = -EINVAL;
                break;
            }
            
            // Check if crypto transform is properly set up
            if (!dev->tfm || !dev->req) {
                printk(KERN_ERR "FINALIZE: crypto transform missing\n");
                ret = -EINVAL;
                break;
            }
            
            if (dev->setup.c_mode == ENC) {
                // Special handling for empty input
                if (dev->in_buffer_pos == 0) {
                    // For zero-length input, just add a full block of padding
                    if (dev->in_buffer_len < CM_BLOCK_SIZE) {
                        // Resize buffer if needed
                        size_t new_size = CM_BLOCK_SIZE;
                        unsigned char *new_buffer = krealloc(dev->in_buffer, new_size, GFP_KERNEL);
                        if (!new_buffer) {
                            ret = -ENOMEM;
                            break;
                        }
                        dev->in_buffer = new_buffer;
                        dev->in_buffer_len = new_size;
                    }
                    
                    // Add a full block of padding (value = block size)
                    memset(dev->in_buffer, CM_BLOCK_SIZE, CM_BLOCK_SIZE);
                    dev->in_buffer_pos = CM_BLOCK_SIZE;
                } else {
                    // Normal padding for non-empty input
                    int padded_len = add_pkcs7_padding(dev->in_buffer, 
                                                     dev->in_buffer_pos, 
                                                     dev->in_buffer_len);
                    if (padded_len < 0) {
                        printk(KERN_ERR "FINALIZE: padding failed with error %d\n", padded_len);
                        ret = padded_len;
                        break;
                    }
                    dev->in_buffer_pos = padded_len;
                }
                
                // Process final data
                // perform encryption
                ret = process_data(dev);
                if (ret < 0) {
                    printk(KERN_ERR "FINALIZE: process_data failed with error %d\n", ret);
                    if (ret != -EAGAIN) {
                        break;
                    }
                }
                ret = 0; // Always set to success if we got here
            } else { // DEC mode
                // Process all data first
                // use basic mode or adv mode
                ret = process_data(dev);
                if (ret < 0 && ret != -EAGAIN) {
                    break;
                }
                
                // Process any remaining data during finalization
                if (dev->in_buffer_pos > 0) {
                    // Ensure data is a multiple of block size
                    if (dev->in_buffer_pos % CM_BLOCK_SIZE != 0) {
                        ret = -EINVAL;
                        break;
                    }
                    
                    // Process remaining data (the last withheld block)
                    ret = aes_ecb_crypt(
                        dev,
                        dev->out_buffer + dev->out_buffer_pos,
                        dev->in_buffer,
                        dev->in_buffer_pos,
                        dev->setup.c_mode
                    );
                    
                    if (ret < 0) {
                        break;
                    }
                    
                    // Update positions
                    dev->out_buffer_pos += dev->in_buffer_pos;
                    dev->in_buffer_pos = 0;
                }
                
                // Remove PKCS#7 padding from output
                int unpadded_len = remove_pkcs7_padding(dev->out_buffer, dev->out_buffer_pos);
                if (unpadded_len < 0) {
                    ret = unpadded_len;
                    break;
                }
                
                dev->out_buffer_pos = unpadded_len;
            }
            
            dev->setup.is_finalized = true;
            break;
            
        case CM_IOC_CLEANUP:
            // Clear buffers and reset finalize state
            cleanup_buffers(dev);
            dev->setup.is_finalized = false;
            break;
            
        case CM_IOC_CNT_RST:
            // Reset counters
            mutex_lock(&global_lock);
            atomic_set(&total_bytes_read, 0);
            atomic_set(&total_bytes_written, 0);
            
            // Reset byte frequency
            {
                int i;
                for (i = 0; i < 256; i++)
                    atomic_set(&byte_frequency[i], 0);
            }
            mutex_unlock(&global_lock);
            break;
            
        default:
            ret = -ENOTTY;
    }
    
    mutex_unlock(&dev->lock);
    return ret;
}

// Proc file show function
static int crypto_proc_show(struct seq_file *m, void *v)
{
    int i, j;
    
    // Print bytes read and written
    seq_printf(m, "%d %d\n", 
              atomic_read(&total_bytes_read), 
              atomic_read(&total_bytes_written));
    
    // Print byte frequency matrix (16x16)
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 16; j++) {
            seq_printf(m, "%d", atomic_read(&byte_frequency[i * 16 + j]));
            if (j < 15)
                seq_printf(m, " ");
        }
        seq_printf(m, "\n");
    }
    
    return 0;
}

// Proc file open function
static int crypto_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, crypto_proc_show, NULL);
}

// Initialize the module
static int __init crypto_init(void)
{
    int ret;
    int i;
    dev_t dev;
    
    // Initialize byte frequency counters
    for (i = 0; i < 256; i++)
        atomic_set(&byte_frequency[i], 0);
    
    // Allocate a major number
    ret = alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ALERT "Failed to register a major number\n");
        return ret;
    }
    major_number = MAJOR(dev);              // stores the allocated major number
    
    // Initialize the character device
    cdev_init(&cdev, &fops);
    cdev.owner = THIS_MODULE;
    ret = cdev_add(&cdev, MKDEV(major_number, 0), 1);       // add initialized character device to kernel's device system
    if (ret < 0) {
        unregister_chrdev_region(MKDEV(major_number, 0), 1);
        printk(KERN_ALERT "Failed to add character device\n");
        return ret;
    }
    
    // Register device class
    crypto_class = class_create(CLASS_NAME);// create device class
    if (IS_ERR(crypto_class)) {
        cdev_del(&cdev);
        unregister_chrdev_region(MKDEV(major_number, 0), 1);
        printk(KERN_ALERT "Failed to register device class\n");
        return PTR_ERR(crypto_class);
    }
    
    // Register the device
    crypto_device = device_create(crypto_class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);   // makes device visible as /dev/cryptodev
    if (IS_ERR(crypto_device)) {
        class_destroy(crypto_class);
        cdev_del(&cdev);
        unregister_chrdev_region(MKDEV(major_number, 0), 1);
        printk(KERN_ALERT "Failed to create the device\n");
        return PTR_ERR(crypto_device);
    }
    
    // Create proc entry
    proc_entry = proc_create(PROC_ENTRY, 0444, NULL, &proc_fops);
    if (!proc_entry) {
        device_destroy(crypto_class, MKDEV(major_number, 0));
        class_destroy(crypto_class);
        cdev_del(&cdev);
        unregister_chrdev_region(MKDEV(major_number, 0), 1);
        printk(KERN_ALERT "Failed to create proc entry\n");
        return -ENOMEM;
    }
    
    printk(KERN_INFO "Crypto module initialized with major number %d\n", major_number);
    return 0;
}

// Clean up the module
static void __exit crypto_exit(void)
{
    // Remove proc entry
    proc_remove(proc_entry);
    
    // Destroy device
    device_destroy(crypto_class, MKDEV(major_number, 0));
    
    // Destroy class
    class_destroy(crypto_class);
    
    // Remove character device
    cdev_del(&cdev);
    
    // Unregister character device region
    unregister_chrdev_region(MKDEV(major_number, 0), 1);
    
    printk(KERN_INFO "Crypto module cleaned up\n");
}

module_init(crypto_init);
module_exit(crypto_exit);