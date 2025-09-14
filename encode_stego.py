import cv2
import numpy as np

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


def encode(image_name, secret_data, output_name, n_bits=1):
    image = cv2.imread(image_name)
    n_bytes = image.shape[0] * image.shape[1] * 3 * n_bits // 8
    print(f"[*] Maximum bytes to encode with {n_bits} LSB(s): {n_bytes}")

    # add stop marker
    secret_data += "====="
    binary_secret_data = to_bin(secret_data)
    data_len = len(binary_secret_data)

    print(f"[*] Encoding message of {len(secret_data)} characters ({data_len} bits)")

    if data_len > image.size * n_bits:
        raise ValueError("[!] Insufficient bytes, need bigger image or fewer bits per channel.")

    print("[*] Encoding data...")
    data_index = 0

    for row in image:
        for pixel in row:
            for channel in range(3):  # B, G, R
                if data_index < data_len:
                    mask = ~((1 << n_bits) - 1) & 0xFF
                    bits = int(binary_secret_data[data_index:data_index+n_bits].ljust(n_bits, '0'), 2)
                    pixel[channel] = np.uint8((int(pixel[channel]) & mask) | bits)
                    data_index += n_bits
                if data_index >= data_len:
                    break
            if data_index >= data_len:
                break
        if data_index >= data_len:
            break

    cv2.imwrite(output_name, image)
    print(f"[+] Data encoded successfully into {output_name}")


if __name__ == "__main__":
    print("=== LSB Steganography Encoder ===")
    input_file = input("Enter input BMP filename: ").strip()
    secret_message = input("Enter the secret message to hide: ").strip()

    while True:
        try:
            n_bits = int(input("Enter number of LSBs to use (1–8): ").strip())
            if 1 <= n_bits <= 8:
                break
            else:
                print("⚠️ Please enter a number between 1 and 8.")
        except ValueError:
            print("⚠️ Invalid input. Enter a number between 1 and 8.")

    output_file = "encoded_output.bmp"
    encode(input_file, secret_message, output_file, n_bits)
