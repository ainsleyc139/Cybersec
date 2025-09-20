import cv2
import os

START_MARKER = b"<<<START>>>"
STOP_MARKER  = b"====="

def to_bin(data: bytes) -> str:
    return ''.join(format(b, "08b") for b in data)

def write_lossless_avi(path, fps, size):
    # Force uncompressed BI_RGB on Windows
    fourcc = cv2.VideoWriter_fourcc(*"DIB ")
    return cv2.VideoWriter(path, fourcc, fps, size)

def encode(video_path, payload_str, output_path, n_bits=1, start_sec=0.0):
    if not (1 <= n_bits <= 8):
        raise ValueError("n_bits must be 1..8")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(start_sec * fps)))

    # Build payload: header + message (TEXT) + STOP
    msg_bytes = payload_str.encode()
    header = START_MARKER + f"TEXT;SIZE:{len(msg_bytes)};LSB:{n_bits};".encode() + STOP_MARKER
    secret = header + msg_bytes
    bits = to_bin(secret)
    need_bits = len(bits)
    capacity_bits = total * width * height * 3 * n_bits
    if need_bits > capacity_bits:
        cap.release()
        raise ValueError(f"❌ Payload too large. Need {need_bits//8} bytes, capacity {(capacity_bits//8)} bytes.")

    out = write_lossless_avi(output_path, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("❌ Could not open VideoWriter with lossless codec. Try a different path/permission.")

    data_idx = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= start_frame and data_idx < need_bits:
            # embed into this frame
            for y in range(height):
                row = frame[y]
                for x in range(width):
                    px = row[x]
                    for c in range(3):  # B,G,R
                        if data_idx < need_bits:
                            # clear n LSBs, set new
                            mask = ~((1 << n_bits) - 1) & 0xFF
                            chunk = bits[data_idx:data_idx+n_bits].ljust(n_bits, '0')
                            val = (int(px[c]) & mask) | int(chunk, 2)
                            px[c] = val
                            data_idx += n_bits
                        else:
                            break
                    if data_idx >= need_bits:
                        break
                if data_idx >= need_bits:
                    break

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[+] Data encoded successfully into {output_path}")

    # ------- Self-check: ensure header can be found in the written file -------
    try:
        self_check_header(output_path, n_bits)
        print("[✓] Self-check passed: header found in output (LSBs preserved).")
    except Exception as e:
        raise RuntimeError(
            "❌ Self-check failed: could not find header in written video. "
            "This means the codec altered the LSBs. "
            "Make sure you're using the provided script (DIB / uncompressed AVI) and not re-encoding."
        ) from e

def self_check_header(stego_path, n_bits):
    cap = cv2.VideoCapture(stego_path)
    if not cap.isOpened():
        raise FileNotFoundError("cannot open written video for self-check")

    marker = START_MARKER + b"."  # just presence of START is enough, we scan for full START..STOP below
    # We'll scan up to a sane limit of bytes to find START..STOP in header.
    MAX_SCAN_BYTES = 8192

    bitbuf = []
    byte_stream = bytearray()

    def try_pop_byte():
        nonlocal bitbuf
        if len(bitbuf) >= 8:
            v = int(''.join(bitbuf[:8]), 2)
            del bitbuf[:8]
            return v
        return None

    found = False
    scanned_bytes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        for y in range(h):
            for x in range(w):
                px = frame[y, x]
                for c in range(3):
                    bits = format(px[c], "08b")[-n_bits:]
                    bitbuf.extend(bits)
                    while True:
                        b = try_pop_byte()
                        if b is None:
                            break
                        byte_stream.append(b)
                        scanned_bytes += 1
                        if scanned_bytes > MAX_SCAN_BYTES:
                            cap.release()
                            raise RuntimeError("header not found within scan limit")
                        # check for full START...STOP
                        if START_MARKER in byte_stream:
                            # try slicing after START and looking for STOP
                            start_idx = byte_stream.find(START_MARKER)
                            tail = byte_stream[start_idx+len(START_MARKER):]
                            if STOP_MARKER in tail:
                                found = True
                                break
                if found:
                    break
            if found:
                break
        if found:
            break

    cap.release()
    if not found:
        raise RuntimeError("header not found")

if __name__ == "__main__":
    print("=== Video Steganography Encoder (Lossless AVI / OpenCV) ===")
    mode = input("Hide Text or File? (T/F): ").strip().lower()
    if mode != "t":
        print("For this minimal fix, use Text mode. (File mode is easy to add after verify.)")
    in_video = input("Enter input video filename: ").strip()
    if not os.path.exists(in_video):
        raise FileNotFoundError(f"❌ Not found: {in_video}")
    msg = input("Enter secret message: ").strip()
    n_bits = int(input("Enter number of LSBs to use (1–8): ").strip())
    start = float(input("Enter start timestamp in seconds: ").strip())
    encode(in_video, msg, "stego_output.avi", n_bits, start)
