import cv2
import numpy as np

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


if __name__ == "__main__":
    print("=== LSB Steganography Decoder ===")
    input_file = input("Enter encoded BMP filename: ").strip()
    if not input_file.lower().endswith(".bmp"):
        input_file += ".bmp"

    while True:
        try:
            n_bits = int(input("Enter number of LSBs used for encoding (1–8): ").strip())
            if 1 <= n_bits <= 8:
                break
            else:
                print("⚠️ Please enter a number between 1 and 8.")
        except ValueError:
            print("⚠️ Invalid input. Enter a number between 1 and 8.")

    hidden_message = decode(input_file, n_bits)
    print("[+] Hidden Message:", hidden_message)
