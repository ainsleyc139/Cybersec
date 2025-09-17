import cv2
import numpy as np

def to_bin(data):
    """Convert int or ndarray to binary string(s)."""
    if isinstance(data, (int, np.integer)):
        return format(int(data), "08b")
    elif isinstance(data, np.ndarray):
        return [format(int(i), "08b") for i in data]
    else:
        raise TypeError(f"Unsupported type: {type(data)}")


def decode(image_name, n_bits=1):
    print("[+] Decoding...")
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"❌ Could not open {image_name}. Check path and extension.")

    binary_data = ""
    decoded_bytes = bytearray()
    stop_marker = b"====="

    for row in image:
        for pixel in row:
            for channel in range(3):
                binary_data += to_bin(pixel[channel])[-n_bits:]

                while len(binary_data) >= 8:
                    byte = binary_data[:8]
                    binary_data = binary_data[8:]
                    decoded_bytes.append(int(byte, 2))

                    if decoded_bytes.endswith(stop_marker):
                        decoded_bytes = decoded_bytes[:-len(stop_marker)]
                        return decoded_bytes

    raise ValueError("❌ Stop marker not found. Possibly wrong LSB count or corrupted data.")


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

    data = decode(input_file, n_bits)

    # 🔹 Distinguish between text and file payloads
    if data.startswith(b"FILE:"):
        try:
            # Parse filename
            header_end = data.find(b";SIZE:")
            if header_end == -1:
                raise ValueError("❌ Malformed header: missing SIZE field.")

            filename = data[5:header_end].decode(errors="ignore")

            # Parse size
            size_start = header_end + len(";SIZE:")
            size_end = data.find(b";", size_start)
            size = int(data[size_start:size_end].decode())

            # Extract raw payload
            payload = data[size_end+1:size_end+1+size]

            out_name = f"decoded_{filename}"
            with open(out_name, "wb") as f:
                f.write(payload)

            print(f"[+] File extracted and saved as {out_name}")
        except Exception as e:
            print(f"❌ Error extracting file: {e}")
    else:
        try:
            message = data.decode()
            print("[+] Hidden message:", message)
        except UnicodeDecodeError:
            print("[+] Hidden raw bytes (non-text payload)")
