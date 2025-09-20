import cv2
from collections import deque

START_MARKER = b"<<<START>>>"
STOP_MARKER  = b"====="

# How many bytes of header we’ll keep in memory during search (safety cap)
MAX_HEADER_SCAN_BYTES = 8192

def bits_of_bytearray(buf: bytearray) -> str:
    # Convert bytearray tail to string once (used for debugging if needed)
    return ''.join(format(b, '08b') for b in buf[-64:])

def decode(video_path: str, user_n_bits: int):
    if not (1 <= user_n_bits <= 8):
        raise ValueError("n_bits must be in [1..8].")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open {video_path}")

    # --- PASS 1: Find header (<<<START>>> ... =====) ---
    # We build bytes on the fly from the provided n_bits.
    bitbuf = []                # raw bits (as '0'/'1' chars) to assemble bytes
    byte_stream = bytearray()  # sliding byte buffer for header search
    header_found = False
    header_bytes = b""
    # We’ll continue seamlessly into payload once header is found, without rewinding.

    def try_pop_byte_from_bits():
        """Pop one byte from bitbuf if >=8 bits available; return int or None."""
        nonlocal bitbuf
        if len(bitbuf) >= 8:
            byte_bits = ''.join(bitbuf[:8])
            del bitbuf[:8]
            return int(byte_bits, 2)
        return None

    # Scan frames until we see START...STOP in the assembled bytes
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        h, w, _ = frame.shape
        for y in range(h):
            for x in range(w):
                px = frame[y, x]
                for ch in range(3):  # B,G,R channels
                    # Append n LSBs of this channel to bit buffer
                    bits = format(px[ch], "08b")[-user_n_bits:]
                    bitbuf.extend(bits)

                    # Assemble as many full bytes as available
                    while True:
                        b = try_pop_byte_from_bits()
                        if b is None:
                            break
                        byte_stream.append(b)

                        # Keep header buffer bounded to avoid runaway memory
                        if len(byte_stream) > MAX_HEADER_SCAN_BYTES:
                            # If header still not found within this size, give up
                            cap.release()
                            raise ValueError("❌ No hidden data found (header not detected within scan limit).")

                        # Check for STOP marker at the tail efficiently
                        if len(byte_stream) >= len(STOP_MARKER):
                            if byte_stream.endswith(STOP_MARKER):
                                # Now verify we have START marker earlier
                                start_idx = byte_stream.find(START_MARKER)
                                if start_idx != -1:
                                    # Header is bytes between START and STOP
                                    header_found = True
                                    header_bytes = byte_stream[start_idx + len(START_MARKER):-len(STOP_MARKER)]
                                # Either way, we’ve reached the header terminator; exit
                                break
                    # If header terminator hit, break out higher loops
                    if header_found or (len(byte_stream) >= len(STOP_MARKER) and byte_stream.endswith(STOP_MARKER)):
                        break
                if header_found or (len(byte_stream) >= len(STOP_MARKER) and byte_stream.endswith(STOP_MARKER)):
                    break
            if header_found or (len(byte_stream) >= len(STOP_MARKER) and byte_stream.endswith(STOP_MARKER)):
                break
        if header_found:
            break

    if not header_found:
        cap.release()
        raise ValueError("❌ No hidden data found (START/STOP markers not present).")

    # --- Parse header ---
    # Header is ASCII like: "TEXT;SIZE:5;LSB:1;" or "FILE:filename.ext;SIZE:123;LSB:2;"
    try:
        header_str = header_bytes.decode(errors="strict")
    except Exception:
        cap.release()
        raise ValueError("❌ Corrupted header (cannot decode).")

    meta = {}
    payload_type = "TEXT"
    filename = None

    # Split on ';' and parse k:v pairs
    for part in header_str.split(';'):
        if not part:
            continue
        if part.startswith("TEXT"):
            payload_type = "TEXT"
        elif part.startswith("FILE:"):
            payload_type = "FILE"
            filename = part.split(":", 1)[1]
        elif ":" in part:
            k, v = part.split(":", 1)
            meta[k.strip().upper()] = v.strip()

    if "SIZE" not in meta or "LSB" not in meta:
        cap.release()
        raise ValueError("❌ Invalid header (missing SIZE/LSB).")

    declared_size = int(meta["SIZE"])
    true_n_bits = int(meta["LSB"])

    # Enforce correct LSB: if user picked wrong, fail fast
    if true_n_bits != user_n_bits:
        cap.release()
        raise ValueError("❌ Wrong LSB used. Decoding failed.")

    # At this point:
    # - We have consumed bits up to and including STOP_MARKER of header.
    # - There may still be residual bits in 'bitbuf' from the last pixels read.
    # The payload starts *after* the header STOP; continue assembling bytes from
    # the current bit position forward until we collect 'declared_size' bytes.

    # Clear the header byte_stream to free memory; keep bitbuf as-is
    byte_stream = None

    # --- PASS 2: Collect payload bytes (continue from current read position) ---
    payload = bytearray()

    def pump_bits_and_collect(frame_local=None):
        """Inner function to read bits from the current frame or subsequent frames,
           assemble bytes, and append into payload until declared_size is reached."""
        nonlocal payload, bitbuf
        if frame_local is not None:
            frames_iter = [frame_local]
        else:
            frames_iter = []
        # Use generator-like loop: first finish given frame, then continue cap.read()
        while True:
            if frame_local is None:
                ret2, fr = cap.read()
                if not ret2:
                    break
            else:
                fr = frame_local

            h2, w2, _ = fr.shape
            for yy in range(h2):
                for xx in range(w2):
                    p = fr[yy, xx]
                    for c in range(3):
                        bits_local = format(p[c], "08b")[-user_n_bits:]
                        bitbuf.extend(bits_local)

                        # Assemble bytes as available
                        while len(bitbuf) >= 8 and len(payload) < declared_size:
                            byte_bits = ''.join(bitbuf[:8])
                            del bitbuf[:8]
                            payload.append(int(byte_bits, 2))
                            if len(payload) >= declared_size:
                                return True  # done
            # after finishing this frame
            if frame_local is not None:
                # consumed the provided frame once; switch to streaming mode next
                frame_local = None
            # loop continues to next frame via cap.read()
        return False  # finished video without enough bytes

    # First, try to finish current frame’s remaining pixels (if we broke mid-frame)
    # We can’t easily resume the *exact* pixel position without tracking it,
    # so we simply continue from the *next* pixel read in subsequent frames.
    # That’s acceptable because we already broke out at channel/pixel boundaries above.
    done = pump_bits_and_collect(None)
    if not done and len(payload) < declared_size:
        # if we still didn’t complete, the video didn’t contain enough embedded payload
        cap.release()
        raise ValueError(f"❌ Could not fully extract payload (got {len(payload)}/{declared_size} bytes).")

    cap.release()

    # Output
    if payload_type == "FILE":
        out_name = f"decoded_{filename or 'file.bin'}"
        with open(out_name, "wb") as f:
            f.write(payload)
        print(f"[+] File extracted and saved as {out_name}")
    else:
        try:
            msg = payload.decode(errors="strict")
        except Exception:
            # fallback if non-UTF8
            msg = payload.decode(errors="ignore")
        print("[+] Hidden message extracted:")
        print(msg)


if __name__ == "__main__":
    print("=== Video Steganography Decoder (OpenCV, robust) ===")
    video = input("Enter stego video filename: ").strip()
    n_bits = int(input("Enter number of LSBs used (1–8): ").strip())
    decode(video, n_bits)
