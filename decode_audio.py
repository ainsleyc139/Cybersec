import numpy as np
import wave
import hashlib

def to_bin(data):
    """Convert data into binary format as string/list"""
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(int(i), "08b") for i in data]
    elif isinstance(data, (int, np.integer)):
        return format(int(data), "08b")
    else:
        raise TypeError(f"Type not supported: {type(data)}")

def decode_audio_header(binary_data):
    
    header_data = ""

    for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            char = chr(int(byte, 2))
            header_data += char
            if char == '>':
                break
    header_length = len(header_data) * 8
    
    return header_data, header_length

def to_float_or_none(x):
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    return float(s)

def hash_to_seed(key_text: str) -> int:
    if not isinstance(key_text, str) or not key_text.strip():
        raise ValueError("Key/passphrase is required.")
    return int.from_bytes(hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little", signed=False)

def decode_audio(file_name, n_bits=1):
    n_bits = int(n_bits)
    print("[+] Decoding...")
    with wave.open(file_name, "rb") as audio:
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))

    if frames is None:
        raise FileNotFoundError(f"Could not open {file_name}. Check path and extension.")
    
    frame_size = num_channels * sample_width

    binary_data = ""
    decoded_data = ""

    # convert stop marker to binary so we know where to stop decoding
    stop_marker = "====="
    binary_stop_marker = to_bin(stop_marker)
   
    # same for the starting characters of the header '<type:'
    marker = "<type:" 
    binary_header_marker = to_bin(marker)

    bits_buffer = []
    end_marker_len = len(binary_stop_marker)

    for i in range(0,len(frames)):
        byte = frames[i]

        # take the last n_bits from each byte in frames, convert to string
        bits = byte & ((1 << n_bits) - 1)
        bits_buffer.append(f'{bits:0{n_bits}b}')

        #check for stop marker in the last few bits
        if (''.join(bits_buffer[-end_marker_len:]) == binary_stop_marker):
            # Remove the marker bits
            bits_buffer = bits_buffer[:-end_marker_len]
            break

    binary_data = ''.join(bits_buffer)

    # search for the header in the binary data by finding the marker
    header_start = binary_data.find(binary_header_marker)

    # decode the header data '<type: ;size: >'
    header_data, header_length = decode_audio_header(binary_data[header_start:])
    header_parts = header_data.strip("<>").split(';')
    payload_extension = '.' + header_parts[0][5:]
    payload_size = int(header_parts[1][5:])
    
    # If the payload is not a file, extension is put as .nil when encoding
    if payload_extension == ".nil":
        payload_data = binary_data[header_length:]
        while len(payload_data) >= 8:
            byte_data = payload_data[:8]
            payload_data = payload_data[8:]
            decoded_data += chr(int(byte_data, 2))
    
    else:
        # decode payload for files
        header_length = header_start + header_length
        file_bits = binary_data[header_length:header_length + payload_size * 8]
        file_bytes = bytearray(int(file_bits[i:i+8], 2) for i in range(0, len(file_bits), 8))

        with open(f"decoded_file{payload_extension}", "wb") as f:
            f.write(file_bytes)
        print(f"[+] File saved as decoded_file{payload_extension}")
    
def decode_audio_with_key(file_name, key, start_time, end_time, n_bits=1):
    n_bits = int(n_bits)
    print("[+] Decoding...")
    with wave.open(file_name, "rb") as audio:
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))

    if frames is None:
        raise FileNotFoundError(f"Could not open {file_name}. Check path and extension.")
    
    frame_size = num_channels * sample_width
    total_bytes = len(frames)

    
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
        raise ValueError("[!] start and end times must be entered")

    # Build shuffled order using same PRNG/seed as your encoder
    order = np.arange(start_byte, end_byte, dtype = np.int64)
    seed = hash_to_seed(key)
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    bits_buffer =[]
    for pos in order:
        bits = frames[pos] & ((1 << n_bits ) - 1 )
        bits_buffer.append(f'{bits:0{n_bits}b}')
    
    binary_data =''.join(bits_buffer)
    
    # front of header as a market to find it
    marker = "<type:" 
    binary_header_marker = to_bin(marker)

    decoded_data = ""

    # search for the header in the binary data by finding the marker
    header_start = binary_data.find(binary_header_marker)

    # decode the header data '<type: ;size: >'
    header_data, header_length = decode_audio_header(binary_data[header_start:])
    header_parts = header_data.strip("<>").split(';')
    payload_extension = '.' + header_parts[0][5:]
    payload_size = int(header_parts[1][5:])

    
    # If the payload is not a file, extension is put as .nil when encoding
    if payload_extension == ".nil":
        payload_data = binary_data[header_length:]
        while len(payload_data) >= 8:
            byte_data = payload_data[:8]
            payload_data = payload_data[8:]
            decoded_data += chr(int(byte_data, 2))
        decoded_data = decoded_data.removesuffix("=====")
        print('[+] The secret message is: '+ decoded_data)
    
    else:
        # decode payload for files
        header_length = header_start + header_length
        file_bits = binary_data[header_length:header_length + payload_size * 8]
        file_bytes = bytearray(int(file_bits[i:i+8], 2) for i in range(0, len(file_bits), 8))

        with open(f"decoded_file{payload_extension}", "wb") as f:
            f.write(file_bytes)
        print(f"[+] File saved as decoded_file{payload_extension}")

if __name__ == "__main__":
    print("=== LSB Steganography Decoder ===")
    input_file = input("Enter encoded  filename: ").strip()

    while True:
        try:
            n_bits = int(input("Enter number of LSBs used for encoding (1–8): ").strip())
            if 1 <= n_bits <= 8:
                break
            else:
                print("! Please enter a number between 1 and 8.")
        except ValueError:
            print("! Invalid input. Enter a number between 1 and 8.")
    
    if input_file.endswith(".wav"):
        use_key = input("use a secret key? (Y/N) ").strip()
        if use_key.lower() == "y":
            key = input("enter secret key: ").strip()
            start = input("Enter start time in seconds: ").strip()
            end = input("enter end time in seconds: ").strip()
            decode_audio_with_key(input_file, key, start, end, n_bits)
        else: 
            decode_audio(input_file, n_bits)
    else:
        print("Please enter a .wav file")
