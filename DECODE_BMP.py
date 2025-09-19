import cv2
import os

START_MARKER = "<<<START>>>"
STOP_MARKER = "====="


def decode(image_name):
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}")

    h, w, _ = image.shape

    # Step 1: always decode header using 1 LSB
    binary_data, decoded_data = "", ""
    for y in range(h):
        for x in range(w):
            pixel = image[y, x]
            for channel in range(3):
                binary_data += format(pixel[channel], "08b")[-1:]  # always 1 bit
                while len(binary_data) >= 8:
                    byte = binary_data[:8]
                    binary_data = binary_data[8:]
                    decoded_data += chr(int(byte, 2))

                    if decoded_data.endswith(STOP_MARKER):
                        decoded_data = decoded_data[:-len(STOP_MARKER)]
                        if START_MARKER in decoded_data:
                            header_str = decoded_data.split(START_MARKER, 1)[1]
                            return parse_and_extract(header_str, image)
    raise ValueError("❌ No hidden data found.")


def parse_and_extract(header_str, image):
    parts = header_str.split(";")
    meta = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            meta[k] = v

    if "REGION" not in meta or "SIZE" not in meta or "LSB" not in meta:
        raise ValueError("❌ Invalid metadata header.")

    x1, y1, x2, y2 = map(int, meta["REGION"].split(","))
    size = int(meta["SIZE"])
    n_bits = int(meta["LSB"])

    if "FILE" in meta:
        filename = meta["FILE"]
        payload_type = "FILE"
    else:
        filename = None
        payload_type = "TEXT"

    print(f"[*] Found header: region=({x1},{y1},{x2},{y2}), lsb={n_bits}, size={size}")

    return extract_payload(image, x1, y1, x2, y2, n_bits, size, payload_type, filename)


def extract_payload(image, x1, y1, x2, y2, n_bits, size, payload_type, filename):
    binary_data, payload_bytes, data_index = "", bytearray(), 0

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
                        if payload_type == "FILE":
                            with open(f"decoded_{filename}", "wb") as f:
                                f.write(payload_bytes)
                            print(f"[+] File extracted and saved as decoded_{filename}")
                        else:
                            message = payload_bytes.decode(errors="ignore")
                            print(f"[+] Hidden message extracted:\n{message}")
                        return payload_bytes
            if data_index >= size:
                break
        if data_index >= size:
            break

    raise ValueError(f"❌ Could not fully extract payload (decoded {data_index}/{size} bytes)")


if __name__ == "__main__":
    print("=== LSB Steganography Decoder ===")
    input_file = input("Enter stego BMP filename: ").strip()
    if not input_file.lower().endswith(".bmp"):
        input_file += ".bmp"
    decode(input_file)
