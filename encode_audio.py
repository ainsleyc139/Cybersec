import numpy as np
import wave
import hashlib

def to_bin(data):
    """Convert data into binary format as string"""
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(int(i), "08b") for i in data]
    elif isinstance(data, (int, np.integer)):
        return format(int(data), "08b")
    else:
        raise TypeError(f"Type not supported: {type(data)}")

# for times
def to_float_or_none(x):
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    return float(s)


def hash_to_seed(key_text: str) -> int:
    if not isinstance(key_text, str) or not key_text.strip():
        raise ValueError("Key/passphrase is required.")
    return int.from_bytes(hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little", signed=False)

def get_payload(file_path, is_file):

    if is_file: # Read file as binary
        with open(file_path, "rb") as f:
            byte_data = f.read()

        # End marker 
        end_bin = to_bin("=====")

        # Get the file extension (e.g., '.txt')
        extension = file_path.split('.')[-1]
        header_marker = f"<type:{extension};size:{len(byte_data)}>"
        
        # Convert header and file content to binary
        header_bin = ''.join(to_bin(ord(c)) for c in header_marker)
        content_bin = ''.join(to_bin(byte) for byte in byte_data)
    else: 
        # header with 'nil' filetype for when decoding 
        header_marker = f"<type:nil;size:0>"

        # Convert header and file content to binary
        header_bin = ''.join(to_bin(ord(c)) for c in header_marker)
        content_bin = to_bin(file_path)

    return header_bin + content_bin + end_bin

def encode_audio(file_name, is_file, secret_data_file, output_name, n_bits=1, start_time=None, end_time=None):
    n_bits = int(n_bits)
    with wave.open(file_name, "rb") as audio:
        params = audio.getparams()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))
    
    frame_size = num_channels * sample_width
    total_bytes = len(frames)
    max_bits = total_bytes * n_bits

    # convert user input time
    start_time = to_float_or_none(start_time)
    end_time = to_float_or_none(end_time)

    #convert time into frames, then into bytes
    if start_time is not None and end_time is not None:
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        # calculate start and end byte
        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size
        if end_byte > total_bytes:
            raise ValueError("[!] End time exceeds file length")
        max_bits = (end_byte - start_byte) * n_bits
    else:
        start_byte =0
        end_byte = total_bytes
            
    # read payload file to get payload (binary)
    binary_secret_data = get_payload(secret_data_file, is_file)
    
    while len(binary_secret_data) % n_bits !=0:
        binary_secret_data += '0'

    data_len = len(binary_secret_data)
    print(f"[*] Encoding message of ({data_len} bits)")

    if data_len > max_bits:
        raise ValueError("[!] Insufficient bytes, need bigger image or fewer bits per channel.")

    print("[*] Encoding data...")

    payload_index = 0
    for i in range (start_byte, end_byte):
        if payload_index >= data_len:
            break
        bits_chunk = binary_secret_data[payload_index : payload_index + n_bits]
        if not bits_chunk:
            break
        bits_value = int(bits_chunk,2)

        # Clear the LSBs and insert new bits
        mask = ~((1 << n_bits) - 1) & 0xFF
        frames[i] = (frames[i] & mask) | bits_value
        
        # Increment the payload index
        payload_index += n_bits

    # save modified audio 
    with wave.open(output_name, "wb") as output:
        output.setparams(params)
        output.writeframes(frames)
    print(f"[+] Data encoded successfully into {output_name}")

def encode_audio_with_key(file_name, is_file, secret_data_file, output_name, key, n_bits=1, start_time=None, end_time=None):
    n_bits = int(n_bits)
    with wave.open(file_name, "rb") as audio:
        params = audio.getparams()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))
    
    frame_size = num_channels * sample_width
    total_bytes = len(frames)
    max_bits = total_bytes * n_bits

    # convert user input times to float or none
    start_time = to_float_or_none(start_time)
    end_time = to_float_or_none(end_time)

    # convert time into frames, then into bytes
    if start_time is not None and end_time is not None:
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        # calculate start and end byte
        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size
        if end_byte > total_bytes:
            raise ValueError("[!] End time exceeds file length")
        max_bits = (end_byte - start_byte) * n_bits
    else:
        start_byte =0
        end_byte = total_bytes
            
    # read payload file to get payload (binary)
    binary_secret_data = get_payload(secret_data_file, is_file)
    
    while len(binary_secret_data) % n_bits !=0:
        binary_secret_data += '0'

    data_len = len(binary_secret_data)
    print(f"[*] Encoding message of ({data_len} bits)")

    if data_len > max_bits:
        raise ValueError("[!] Insufficient bytes, need bigger image or fewer bits per channel.")

    print("[*] Encoding data...")

    # Split the secret data into chunks of n_bits length so we can shuffle it with key
    bits_chunks =[]
    for i in range(0, data_len, n_bits):
        bits_chunks.append(binary_secret_data[i:i+n_bits])
    
    # Build shuffled order using PRNG/seed
    order = np.arange(start_byte, end_byte, dtype = np.int64)
    seed = hash_to_seed(key)
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    #pad the length to match the end time so that we can decode it later
    while len(bits_chunks) < len(order):
        bits_chunks.append('0' * n_bits)

    # Use shuffled order to encode
    for i, pos in enumerate(order):
        bits_value = int(bits_chunks[i],2)
        # Clear the LSBs and insert new bits
        mask = ~((1 << n_bits) - 1) & 0xFF
        frames[pos] = (frames[pos] & mask) | bits_value

    # Save modified audio 
    with wave.open(output_name, "wb") as output:
        output.setparams(params)
        output.writeframes(frames)
    print(f"[+] Data encoded successfully into {output_name}")

if __name__ == "__main__":
    print("=== LSB Steganography Encoder ===")
    input_file = input("Enter input WAV filename: ").strip()
    
    is_a_file = input("Is the secret a file? (Y/N): ").strip()
    
    if is_a_file.lower() == 'y': 
        is_file = True
    else: 
        is_file = False

    secret_message_file = input("Enter the secret message to hide: ").strip()

    while True:
        try:
            n_bits = int(input("Enter number of LSBs to use (1–8): ").strip())
            if 1 <= n_bits <= 8:
                break
            else:
                print("⚠️ Please enter a number between 1 and 8.")
        except ValueError:
            print("⚠️ Invalid input. Enter a number between 1 and 8.")
        
    if input_file.lower().endswith(".wav"):
        
        output_file = "encoded_output.wav"
        start = input('Enter start time in seconds:').strip()
        end = input('Enter end time in seconds:').strip()
        usekey = input("Use a secret key? (Y/N) ").strip()
        if usekey.lower() == "y":
            key = input("Enter secret key: ").strip()
            encode_audio_with_key(input_file, is_file, secret_message_file, output_file, key, n_bits, start,end)
        else:
            encode_audio(input_file, is_file, secret_message_file, output_file, n_bits, start,end)
    else:
        print("! Please enter a .wav file")
