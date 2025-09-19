import cv2
import numpy as np
import os

START_MARKER = "<<<START>>>"
STOP_MARKER = "====="


def to_bin(data: bytes) -> str:
    return ''.join(format(byte, "08b") for byte in data)


def hide_bits(image, data_bits, n_bits, start_xy=(0, 0)):
    """Write bits sequentially into image starting at start_xy"""
    h, w, _ = image.shape
    data_index = 0
    x0, y0 = start_xy

    for y in range(y0, h):
        for x in range(x0, w):
            pixel = image[y, x]
            for channel in range(3):
                if data_index < len(data_bits):
                    mask = ~((1 << n_bits) - 1) & 0xFF
                    bits = int(data_bits[data_index:data_index + n_bits].ljust(n_bits, '0'), 2)
                    pixel[channel] = np.uint8((int(pixel[channel]) & mask) | bits)
                    data_index += n_bits
            if data_index >= len(data_bits):
                return
    return


def encode(image_name, payload, output_name, n_bits=1, is_file=False, region=None):
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}")

    h, w, _ = image.shape

    # Default region = whole image
    if region is None or len(region) != 4:
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    else:
        x1, y1, x2, y2 = region

    # Prepare payload
    if is_file:
        with open(payload, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(payload)
        payload_bytes = file_bytes
        header = f"{START_MARKER}REGION:{x1},{y1},{x2},{y2};LSB:{n_bits};FILE:{filename};SIZE:{len(payload_bytes)};{STOP_MARKER}"
    else:
        payload_bytes = payload.encode()
        header = f"{START_MARKER}REGION:{x1},{y1},{x2},{y2};LSB:{n_bits};TEXT;SIZE:{len(payload_bytes)};{STOP_MARKER}"

    header_bits = to_bin(header.encode())
    payload_bits = to_bin(payload_bytes)

    # Capacity check
    region_capacity = (x2 - x1 + 1) * (y2 - y1 + 1) * 3 * n_bits // 8
    if len(payload_bytes) > region_capacity:
        raise ValueError("❌ Payload too large for selected region.")

    print(f"[*] Region capacity: {region_capacity} bytes")
    print(f"[*] Encoding payload of {len(payload_bytes)} bytes into region ({x1},{y1})–({x2},{y2})")
    print(f"[*] Metadata header length: {len(header)} bytes")

    # Step 1: hide header at (0,0) using 1 LSB only
    hide_bits(image, header_bits, 1, (0, 0))

    # Step 2: hide payload in chosen region using user n_bits
    data_index = 0
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            pixel = image[y, x]
            for channel in range(3):
                if data_index < len(payload_bits):
                    mask = ~((1 << n_bits) - 1) & 0xFF
                    bits = int(payload_bits[data_index:data_index + n_bits].ljust(n_bits, '0'), 2)
                    pixel[channel] = np.uint8((int(pixel[channel]) & mask) | bits)
                    data_index += n_bits
            if data_index >= len(payload_bits):
                break
        if data_index >= len(payload_bits):
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
    region = tuple(map(int, region_input.split())) if region_input else None

    output_file = "encoded_output.bmp"
    encode(input_file, payload, output_file, n_bits, is_file, region)
