import cv2

def decode(image_name, n_bits, x1, y1, x2, y2):
    """Decode payload from stego image, using user-specified n_bits and region"""
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}")

    # Step 1: extract header inside given region
    header_str = extract_header(image, n_bits, x1, y1, x2, y2)
    if not header_str:
        raise ValueError("❌ No header found (check n_bits/region).")

    print(f"[*] Raw header: {header_str}")

    # Step 2: parse header
    parts = header_str.split(";")
    meta = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            meta[k] = v

    if "SIZE" not in meta:
        raise ValueError("❌ Invalid metadata header.")

    size = int(meta["SIZE"])
    if "FILE" in meta:
        filename = meta["FILE"]
        payload_type = "FILE"
    else:
        filename = None
        payload_type = "TEXT"

    # Step 3: extract payload
    return extract_payload(image, x1, y1, x2, y2, n_bits, size, payload_type, filename)


def extract_header(image, n_bits, x1, y1, x2, y2):
    """Extract header from the specified region"""
    binary_data = ""
    decoded_data = ""

    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            pixel = image[y, x]
            for channel in range(3):
                binary_data += format(pixel[channel], "08b")[-n_bits:]
                while len(binary_data) >= 8:
                    byte = binary_data[:8]
                    binary_data = binary_data[8:]
                    decoded_data += chr(int(byte, 2))
                    if decoded_data.endswith("====="):
                        return decoded_data[:-5]  # strip marker
    return None


def extract_payload(image, x1, y1, x2, y2, n_bits, size, payload_type, filename):
    """Extract payload from given region"""
    binary_data = ""
    payload_bytes = bytearray()
    data_index = 0

    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            pixel = image[y, x]
            for channel in range(3):
                binary_data += format(pixel[channel], "08b")[-n_bits:]
                while len(binary_data) >= 8 and data_index < size:
                    byte = binary_data[:8]
                    binary_data = binary_data[8:]
                    payload_bytes.append(int(byte, 2))
                    data_index += 1
                    if data_index >= size:
                        break
            if data_index >= size:
                break
        if data_index >= size:
            break

    if payload_type == "FILE":
        with open(f"decoded_{filename}", "wb") as f:
            f.write(payload_bytes)
        print(f"[+] File extracted and saved as decoded_{filename}")
    else:
        message = payload_bytes.decode(errors="ignore")
        print(f"[+] Hidden message extracted:\n{message}")

    return payload_bytes


if __name__ == "__main__":
    print("=== LSB Steganography Decoder (Manual Region) ===")
    input_file = input("Enter stego BMP filename: ").strip()
    if not input_file.lower().endswith(".bmp"):
        input_file += ".bmp"
    n_bits = int(input("Enter number of LSBs used (1–8): ").strip())
    x1, y1, x2, y2 = map(int, input("Enter region coordinates x1 y1 x2 y2: ").split())
    decode(input_file, n_bits, x1, y1, x2, y2)
