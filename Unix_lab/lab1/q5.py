#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Lab sample file for the AUP course by Chun-Ying Huang

import sys
from pwn import *
from solpow import solve_pow
import base64
import zlib
from itertools import permutations
import time

def decode_msg(msg):
    try:
        # Extract message between >>> and <<<
        msg = msg.split(b'>>>')[1].split(b'<<<')[0].strip()
        msg = base64.b64decode(msg)
        mlen = int.from_bytes(msg[0:4], 'big')
        if len(msg)-4 != mlen:
            return None
        content = zlib.decompress(msg[4:])
        if content[4] == 65 and content[9] == 66:
            return content
        else:
            return content.decode()
    except Exception as e:
        print(f"Error decoding: {e}")
        return None

def encode_msg(msg):
    # Encode message
    zm = zlib.compress(msg.encode())
    mlen = len(zm)
    encoded = base64.b64encode(mlen.to_bytes(4, 'little') + zm)
    return encoded

def parse_feedback(feedback):
    try:
        # Looking for pattern [0,0,0,x,65,0,0,0,y,66]
        if len(feedback) >= 10 and feedback[4] == 65 and feedback[9] == 66:
            bulls = feedback[3]  # x value
            cows = feedback[8]   # y value
            print(f"Parsed feedback: {bulls} bulls, {cows} cows")
            return bulls, cows
        print(f"Feedback format not recognized: {[b for b in feedback]}")
        return None, None
    except Exception as e:
        print(f"Error parsing feedback: {e}")
        print(f"Raw feedback bytes: {[b for b in feedback]}")
        return None, None

def calculate_bulls_cows(guess, candidate):
    bulls = sum(1 for i in range(4) if guess[i] == candidate[i])
    guess_digits = {}
    candidate_digits = {}
    for i in range(4):
        if guess[i] != candidate[i]:
            guess_digits[guess[i]] = guess_digits.get(guess[i], 0) + 1
            candidate_digits[candidate[i]] = candidate_digits.get(candidate[i], 0) + 1
    cows = sum(min(guess_digits.get(d, 0), candidate_digits.get(d, 0)) 
              for d in guess_digits.keys())
    return bulls, cows

def get_next_guess(previous_guesses, feedback_history):
    if not previous_guesses:
        return '1234'
        
    # Generate remaining candidates
    candidates = [''.join(p) for p in permutations('0123456789', 4)]
    candidates = [c for c in candidates if c not in previous_guesses]
    
    # Filter based on previous feedback
    for guess, (bulls, cows) in feedback_history.items():
        candidates = [c for c in candidates 
                     if calculate_bulls_cows(guess, c) == (bulls, cows)]
        
    if candidates:
        print(f"Found {len(candidates)} possible candidates")
        return candidates[0]
    return None
def main():
    if len(sys.argv) > 1:
        r = remote('up.zoolab.org', 10155)
        solve_pow(r)
    else:
        r = process('./guess_dist.py', shell=False)

    print('*** Starting Bulls and Cows solver...')
    
    previous_guesses = set()
    feedback_history = {}
    attempts = 0
    max_attempts = 10

    # Read and decode initial message
    init_msg = r.recvline().strip()
    print(f"Initial message: {init_msg}")
    decoded_init = decode_msg(init_msg)
    if decoded_init:
        print(f"Decoded initial message: {decoded_init}")

    while attempts < max_attempts:
        # Brief pause to avoid overwhelming the server
        time.sleep(0.1)
        
        try:
            # Make guess
            guess = get_next_guess(previous_guesses, feedback_history)
            if not guess:
                print("No valid guesses remaining!")
                break
            
            previous_guesses.add(guess)
            encoded_guess = encode_msg(guess)
            print(f"Attempt {attempts + 1}: Sending guess {guess}")
            r.sendline(encoded_guess)
            
            # Get initial feedback
            feedback = r.recvline().strip()
            decoded_feedback = decode_msg(feedback)
            print(f'server reply: {decoded_feedback}')
            
            if decoded_feedback:
                bulls, cows = parse_feedback(decoded_feedback)
                if bulls is not None and cows is not None:
                    feedback_history[guess] = (bulls, cows)
            
            # Read any additional output (like the ASCII art)
            while True:
                try:
                    extra = r.recvline(timeout=1).strip()
                    if not extra:
                        break
                    if b'Enter your input' in extra:
                        print("Ready for next guess")
                        break
                    if b'Congratulations' in extra:
                        print("Successfully solved!")
                        return
                    print(decode_msg(extra))
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error reading extra output: {e}")
                    break
            
            if bulls == 4:
                print(f"Found the answer: {guess} in {attempts + 1} attempts!")
                break
            attempts += 1
            
        except EOFError:
            print("Connection closed")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            break

    print("Game finished!")
    r.close()

if __name__ == '__main__':
    main()

# vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4 number cindent fileencoding=utf-8 :