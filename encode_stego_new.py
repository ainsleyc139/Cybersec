import cv2
import numpy as np
import os

def to_bin(data: bytes) -> str:
    """Convert bytes into binary string"""
    return ''.join(format(byte, "08b") for byte in data)


def encode(image_name, payload, output_name, n_bits=1, is_file=False):
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}. Check path and extension.")

    max_bytes = image.shape[0] * image.shape[1] * 3 * n_bits // 8
    print(f"[*] Maximum payload size with {n_bits} LSB(s): {max_bytes} bytes")

    if is_file:
        # Read file as bytes
        with open(payload, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(payload)
        header = f"FILE:{filename};SIZE:{len(file_bytes)};".encode()
        secret_data = header + file_bytes
    else:
        # Treat as plain text
        secret_data = payload.encode()

    # Add stop marker
    secret_data += b"====="

    if len(secret_data) > max_bytes:
        raise ValueError("❌ Insufficient capacity: message/file too large for this image.")

    print(f"[*] Encoding {len(secret_data)} bytes...")
    binary_secret_data = to_bin(secret_data)
    data_len = len(binary_secret_data)

    data_index = 0
    for row in image:
        for pixel in row:
            for channel in range(3):  # B, G, R
                if data_index < data_len:
                    # Clear last n bits and set new bits
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
    if not input_file.lower().endswith(".bmp"):
        input_file += ".bmp"

    mode = input("Hide Text or File? (T/F): ").strip().upper()
    if mode == "F":
        payload = input("Enter file path to hide: ").strip()
        if not os.path.exists(payload):
            raise FileNotFoundError(f"❌ File not found: {payload}")
        is_file = True
    else:
        payload = input("Enter secret message: ").strip()
        if not payload:
            raise ValueError("❌ Secret message cannot be empty.")
        is_file = False

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
    encode(input_file, payload, output_file, n_bits, is_file)
