#!/usr/bin/env python3
from pwn import *
from urllib.parse import urlparse
context.log_level = 'error'  # Suppress pwntools messages

def get_ip_info(url):
    """
    Get IP information using pwntools with simplified curl-like output
    """
    # Parse URL
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    path = parsed_url.path or '/'
    
    try:
        # Create connection
        r = remote(host, port)
        
        # Create request as bytes
        request = [
            b"GET /ip HTTP/1.1",
            f"Host: {host}".encode(),
            b"User-Agent: curl/8.5.0",
            b"Accept: */*",
            b"",
            b""
        ]
        
        # Send request
        r.send(b'\r\n'.join(request))
        
        # Initialize empty buffer for complete response
        full_response = b""
        
        # Read all data
        try:
            while True:
                data = r.recv(timeout=1)
                if not data:
                    break
                full_response += data
        except EOFError:
            pass
        
        # Split response into headers and body
        response_parts = full_response.split(b'\r\n\r\n')
        headers = response_parts[0].decode()
        body = response_parts[1].decode() if len(response_parts) > 1 else ""
        
        
        # Print IP address
        if body:
            print(body.strip(), end="")
        
    except Exception as e:
        print(f"* Error: {str(e)}")
    finally:
        if 'r' in locals():
            r.close()

def main():
    url = "http://ipinfo.io/ip"
    get_ip_info(url)

if __name__ == "__main__":
    main()