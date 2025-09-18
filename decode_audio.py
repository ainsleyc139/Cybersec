import cv2
import numpy as np
import wave

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


def decode(image_name, n_bits=1):
    print("[+] Decoding...")
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_name}. Check path and extension.")

    binary_data = ""
    decoded_data = ""
    stop_marker = "====="

    for row in image:
        for pixel in row:
            # unpack each channel (B, G, R)
            for channel in range(3):
                # take the last n_bits from each channel
                binary_data += to_bin(pixel[channel])[-n_bits:]

                # decode whenever we have at least 8 bits
                while len(binary_data) >= 8:
                    byte = binary_data[:8]
                    binary_data = binary_data[8:]
                    decoded_data += chr(int(byte, 2))

                    # stop when marker is found
                    if decoded_data.endswith(stop_marker):
                        return decoded_data[:-len(stop_marker)]

    print("[!] Warning: stop marker not found")
    return decoded_data


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

def decode_audio(file_name, n_bits=1, start_time=None, end_time=None):
    print("[+] Decoding...")
    with wave.open(file_name, "rb") as audio:
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))

    if frames is None:
        raise FileNotFoundError(f"Could not open {file_name}. Check path and extension.")
    
    frame_size = num_channels * sample_width

    if start_time is not None and end_time is not None:
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        # calculate start and end byte
        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size
        end_byte = min(end_byte,len(frames))
    else:
        start_byte =0
        end_byte = len(frames)

    binary_data = ""
    decoded_data = ""
    stop_marker = "====="

    for i in range(start_byte,end_byte):
        byte = frames[i]
        # take the last n_bits from each byte in frames, convert to string
        binary_data += format(byte & ((1 << n_bits) - 1), f'0{n_bits}b')
        
    header_data, header_length = decode_audio_header(binary_data)
    header_parts = header_data.strip("<>").split(';')
    payload_extension = '.' + header_parts[0][5:]
    payload_size = header_parts[1][5:]

    if payload_extension == ".nil":
        payload_data = binary_data[header_length:]
        while len(payload_data) >= 8:
            byte_data = payload_data[:8]
            payload_data = payload_data[8:]
            decoded_data += chr(int(byte_data, 2))
            
            # stop when marker is found
            if decoded_data.endswith(stop_marker):
                secret_message = decoded_data[:-len(stop_marker)]
                print(f"[+] Hidden message extracted:\n{secret_message}")
        print("[!] Warning: stop marker not found")
        
        
    else:

        file_bits = binary_data[header_length:header_length + int(payload_size) * 8]
        file_bytes = bytearray(int(file_bits[i:i+8], 2) for i in range(0, len(file_bits), 8))

        with open(f"decoded_file{payload_extension}", "wb") as f:
            f.write(file_bytes)
        print(f"[+] File saved as decoded_file{payload_extension}")
    

if __name__ == "__main__":
    print("=== LSB Steganography Decoder ===")
    input_file = input("Enter encoded  filename: ").strip()
    #if not input_file.lower().endswith(".bmp"):
        #input_file += ".bmp"

    while True:
        try:
            n_bits = int(input("Enter number of LSBs used for encoding (1–8): ").strip())
            if 1 <= n_bits <= 8:
                break
            else:
                print("⚠️ Please enter a number between 1 and 8.")
        except ValueError:
            print("⚠️ Invalid input. Enter a number between 1 and 8.")

    if input_file.endswith(".wav"):
        start = input('Enter start time in seconds:')
        end = input('Enter end time in seconds:')
        hidden_message = decode_audio(input_file, n_bits,start,end)
    else:
        hidden_message = decode(input_file, n_bits)
        print("[+] Hidden Message:", hidden_message)
