import cv2
import numpy as np
import os


def to_bin(data: bytes) -> str:
    """Convert bytes into binary string"""
    return ''.join(format(byte, "08b") for byte in data)


def encode(image_name, payload, output_name, n_bits=1, is_file=False, region=None):
    """
    Encode payload into a rectangular region of the cover image.

    region: (x1, y1, x2, y2) inclusive coordinates
            If None → use entire image.
    """
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}. Check path and extension.")

    h, w, _ = image.shape

    # Default to full image if no region specified
    if region is None:
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    else:
        x1, y1, x2, y2 = region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

    region_pixels = (x2 - x1 + 1) * (y2 - y1 + 1)
    max_bytes = region_pixels * 3 * n_bits // 8
    print(f"[*] Maximum payload size in region with {n_bits} LSB(s): {max_bytes} bytes")

    # Build payload with metadata
    if is_file:
        with open(payload, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(payload)
        header = f"REGION:{x1},{y1},{x2},{y2};LSB:{n_bits};FILE:{filename};SIZE:{len(file_bytes)};".encode()
        secret_data = header + file_bytes
    else:
        header = f"REGION:{x1},{y1},{x2},{y2};LSB:{n_bits};TEXT;SIZE:{len(payload)};".encode()
        secret_data = header + payload.encode()

    secret_data += b"====="  # stop marker

    if len(secret_data) > max_bytes:
        raise ValueError("❌ Insufficient capacity: payload too large for selected region.")

    print(f"[*] Encoding {len(secret_data)} bytes...")
    binary_secret_data = to_bin(secret_data)
    data_len = len(binary_secret_data)
    data_index = 0

    # Encode bits into the selected region
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            pixel = image[y, x]
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

    region_input = input("Enter region coordinates as x1 y1 x2 y2 (leave blank for full image): ").strip()
    if region_input:
        try:
            x1, y1, x2, y2 = map(int, region_input.split())
            region = (x1, y1, x2, y2)
        except ValueError:
            raise ValueError("❌ Invalid region format. Use 4 integers: x1 y1 x2 y2")
    else:
        region = None

    output_file = "encoded_output.bmp"
    encode(input_file, payload, output_file, n_bits, is_file, region)
