import cv2
import os
import hashlib
import numpy as np   # ✅ needed to reproduce the same shuffle as encode()



START_MARKER = "<<<START>>>"
STOP_MARKER = "====="

def hash_to_seed(key_text: str) -> int:
    if not isinstance(key_text, str) or not key_text.strip():
        raise ValueError("Key/passphrase is required.")
    # 64-bit seed from SHA-256 (works with NumPy PCG64 etc.)
    return int.from_bytes(hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little", signed=False)

def decode(image_name,key_text:str, user_n_bits):
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}")
    if not isinstance(key_text, str) or key_text.strip() == "":
        raise ValueError("Key/passphrase is required for decoding (image).")

    h, w, _ = image.shape

    # Step 1: always decode header using 1 LSB
    binary_data, decoded_data = "", ""
    for y in range(h):
        for x in range(w):
            pixel = image[y, x]
            for channel in range(3):
                binary_data += format(pixel[channel], "08b")[-1:]  # always 1 bit for header
                while len(binary_data) >= 8:
                    byte = binary_data[:8]
                    binary_data = binary_data[8:]
                    decoded_data += chr(int(byte, 2))

                    if decoded_data.endswith(STOP_MARKER):
                        decoded_data = decoded_data[:-len(STOP_MARKER)]
                        if START_MARKER in decoded_data:
                            header_str = decoded_data.split(START_MARKER, 1)[1]
                            # inside decode(), when header is found:
                            return parse_and_extract(header_str, image, user_n_bits, key_text)

    raise ValueError("❌ No hidden data found.")


def parse_and_extract(header_str, image, user_n_bits, key_text):
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
    true_n_bits = int(meta["LSB"])

    # ✅ Only show generic error, don’t reveal correct LSB
    if user_n_bits != true_n_bits:
        raise ValueError("❌ Wrong LSB used. Decoding failed.")

    if "FILE" in meta:
        filename = meta["FILE"]
        payload_type = "FILE"
    else:
        filename = None
        payload_type = "TEXT"

    print(f"[*] Found header: region=({x1},{y1},{x2},{y2}), lsb={true_n_bits}, size={size}")

    # ✅ derive the same seed as encoder
    seed = hash_to_seed(key_text)

    return extract_payload(image, x1, y1, x2, y2, true_n_bits, size, payload_type, filename, seed)



def extract_payload(image, x1, y1, x2, y2, n_bits, size, payload_type, filename, seed):
    binary_data, payload_bytes, data_index = "", bytearray(), 0

    # ✅ rebuild the same (y,x,ch) order as encoder
    order = []
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            order.append((y, x, 0))  # B
            order.append((y, x, 1))  # G
            order.append((y, x, 2))  # R

    # ✅ shuffle deterministically using the same PRNG/seed
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    # ✅ read bits following the shuffled order
    for (yy, xx, ch) in order:
        pixel = image[yy, xx]
        binary_data += format(pixel[ch], "08b")[-n_bits:]
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

    raise ValueError(f"❌ Could not fully extract payload (decoded {data_index}/{size} bytes)")



if __name__ == "__main__":
    print("=== LSB Steganography Decoder ===")
    input_file = input("Enter stego BMP filename: ").strip()
    if not input_file.lower().endswith(".bmp"):
        input_file += ".bmp"

    key_text = input("Enter key/passphrase: ").strip()
    if not key_text:
        raise ValueError("❌ Key/passphrase cannot be empty.")

    while True:
        try:
            user_n_bits = int(input("Enter number of LSBs used (1–8): ").strip())
            if 1 <= user_n_bits <= 8:
                break
            else:
                print("⚠️ Please enter a number between 1 and 8.")
        except ValueError:
            print("⚠️ Invalid input. Enter a number between 1 and 8.")

    decode(input_file, key_text, user_n_bits)

