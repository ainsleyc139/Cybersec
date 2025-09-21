import wave
import numpy as np
import hashlib

STOP_MARKER = "====="

def hash_to_seed(key_text: str) -> int:
    if not isinstance(key_text, str) or not key_text.strip():
        raise ValueError("Key/passphrase is required.")
    return int.from_bytes(hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little", signed=False)

def _normalize_time_boundaries(audio, start_time, end_time):
    sample_rate  = audio.getframerate()
    sample_width = audio.getsampwidth()
    num_channels = audio.getnchannels()
    num_frames   = audio.getnframes()

    # Parse times
    def _to_float_or_none(x):
        if x is None: return None
        s = str(x).strip()
        if s == "": return None
        return float(s)

    start_sec = _to_float_or_none(start_time)
    end_sec   = _to_float_or_none(end_time)

    if start_sec is not None and end_sec is not None:
        start_frame = max(0, min(int(start_sec * sample_rate), num_frames))
        end_frame   = max(0, min(int(end_sec   * sample_rate), num_frames))
        if end_frame <= start_frame:
            raise ValueError("End time must be greater than start time.")
    else:
        start_frame, end_frame = 0, num_frames

    frame_size = num_channels * sample_width  # bytes per frame
    start_byte = start_frame * frame_size
    end_byte   = end_frame   * frame_size

    return {
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "num_channels": num_channels,
        "num_frames": num_frames,
        "frame_size": frame_size,
        "start_byte": start_byte,
        "end_byte": end_byte,
    }

def _read_bits_from_indices(frames: bytearray, indices, n_bits: int) -> str:
    """Read n_bits LSBs from each selected byte index and return a big '0'/'1' string."""
    mask = (1 << n_bits) - 1
    out_bits = []
    # Pre-size for speed
    out_bits_append = out_bits.append
    for i in indices:
        v = frames[i] & mask
        out_bits_append(format(v, f"0{n_bits}b"))
    return "".join(out_bits)

def _parse_header_from_bits(binary_data: str, max_header_bytes: int = 512):
    """
    Parse header like <type:ext;size:N> from the bitstream.
    Returns (header_str, header_bits_len, payload_type, payload_size).
    payload_type is the string after 'type:' (e.g., 'nil', 'txt', 'png'...).
    """
    header_chars = []
    for i in range(0, min(len(binary_data), max_header_bytes * 8), 8):
        byte = binary_data[i:i+8]
        if len(byte) < 8:
            break
        ch = chr(int(byte, 2))
        header_chars.append(ch)
        if ch == '>':
            break
    header_str = "".join(header_chars)
    if not (header_str.startswith("<") and header_str.endswith(">")):
        raise ValueError("❌ Invalid or missing header. Expected format like <type:ext;size:N>.")

    header_bits_len = len(header_str) * 8
    # Example: "<type:txt;size:123>"
    inner = header_str[1:-1]  # drop < >
    parts = inner.split(";")
    kv = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            kv[k.strip().lower()] = v.strip()

    if "type" not in kv or "size" not in kv:
        raise ValueError("❌ Invalid header fields (need type and size).")

    payload_type = kv["type"]  # 'nil' means text with STOP_MARKER
    try:
        payload_size = int(kv["size"])
    except:
        raise ValueError("❌ Invalid size in header.")

    return header_str, header_bits_len, payload_type, payload_size

def _bits_to_bytes(bits: str) -> bytearray:
    """Convert string of '0'/'1' bits to bytearray (length must be multiple of 8)."""
    if len(bits) % 8 != 0:
        raise ValueError("Bit length not multiple of 8.")
    return bytearray(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))

def decode_audio(file_name, n_bits=1, start_time=None, end_time=None):
    """
    Sequential decoder: reads bytes in order from start_byte..end_byte.
    Header: <type:EXT;size:N>
      - type:nil => text, terminated by STOP_MARKER
      - otherwise => binary file of N bytes saved as decoded_file.EXT
    """
    print("[+] Decoding (sequential)...")
    with wave.open(file_name, "rb") as audio:
        params = _normalize_time_boundaries(audio, start_time, end_time)
        frames = bytearray(audio.readframes(params["num_frames"]))

    start_byte, end_byte = params["start_byte"], params["end_byte"]
    indices = range(start_byte, end_byte)

    # Read all LSB bits in the selected range
    binary_data = _read_bits_from_indices(frames, indices, n_bits)

    # Parse header first
    header_str, header_bits_len, payload_type, payload_size = _parse_header_from_bits(binary_data)

    if payload_type.lower() == "nil":
        # Text mode: decode until STOP_MARKER
        decoded = []
        buf = ""
        for i in range(header_bits_len, len(binary_data), 8):
            byte_bits = binary_data[i:i+8]
            if len(byte_bits) < 8:
                break
            ch = chr(int(byte_bits, 2))
            buf += ch
            if buf.endswith(STOP_MARKER):
                buf = buf[:-len(STOP_MARKER)]
                print(f"[+] Hidden message extracted:\n{buf}")
                return buf
        print("[!] Warning: stop marker not found; returning best-effort text.")
        print(f"[+] Partial message:\n{buf}")
        return buf
    else:
        # Binary mode: read exactly payload_size bytes
        needed_bits = payload_size * 8
        start = header_bits_len
        end = start + needed_bits
        if end > len(binary_data):
            raise ValueError(f"❌ Not enough bits for declared payload size ({payload_size} bytes).")
        file_bits = binary_data[start:end]
        file_bytes = _bits_to_bytes(file_bits)
        ext = payload_type  # e.g., 'png', 'txt'
        outname = f"decoded_file.{ext}"
        with open(outname, "wb") as f:
            f.write(file_bytes)
        print(f"[+] File saved as {outname}")
        return outname

def decode_audio_with_key(file_name, key_text: str, n_bits=1, start_time=None, end_time=None):
    """
    Keyed decoder: reconstructs same byte order shuffled by key (PCG64),
    then reads bits in that order. Header and payload handling same as sequential.
    """
    print("[+] Decoding (key-shuffled)...")
    with wave.open(file_name, "rb") as audio:
        params = _normalize_time_boundaries(audio, start_time, end_time)
        frames = bytearray(audio.readframes(params["num_frames"]))

    start_byte, end_byte = params["start_byte"], params["end_byte"]

    # Build shuffled order using same PRNG/seed as your encoder
    order = np.arange(start_byte, end_byte, dtype=np.int64)
    seed = hash_to_seed(key_text)
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    binary_data = _read_bits_from_indices(frames, order, n_bits)

    # Parse header
    header_str, header_bits_len, payload_type, payload_size = _parse_header_from_bits(binary_data)

    if payload_type.lower() == "nil":
        # Text mode
        decoded = []
        buf = ""
        for i in range(header_bits_len, len(binary_data), 8):
            byte_bits = binary_data[i:i+8]
            if len(byte_bits) < 8:
                break
            ch = chr(int(byte_bits, 2))
            buf += ch
            if buf.endswith(STOP_MARKER):
                buf = buf[:-len(STOP_MARKER)]
                print(f"[+] Hidden message extracted:\n{buf}")
                return buf
        print("[!] Warning: stop marker not found; returning best-effort text.")
        print(f"[+] Partial message:\n{buf}")
        return buf
    else:
        # Binary mode
        needed_bits = payload_size * 8
        start = header_bits_len
        end = start + needed_bits
        if end > len(binary_data):
            raise ValueError(f"❌ Not enough bits for declared payload size ({payload_size} bytes).")
        file_bits = binary_data[start:end]
        file_bytes = _bits_to_bytes(file_bits)
        ext = payload_type
        outname = f"decoded_file.{ext}"
        with open(outname, "wb") as f:
            f.write(file_bytes)
        print(f"[+] File saved as {outname}")
        return outname

# ---------- Example CLI ----------
if __name__ == "__main__":
    print("=== LSB Audio Decoder ===")
    input_file = input("Enter encoded WAV filename: ").strip()
    if not input_file.lower().endswith(".wav"):
        raise ValueError("Please provide a .wav file.")

    while True:
        try:
            n_bits = int(input("Enter number of LSBs used for encoding (1–8): ").strip())
            if 1 <= n_bits <= 8:
                break
            else:
                print("⚠️ Please enter a number between 1 and 8.")
        except ValueError:
            print("⚠️ Invalid input. Enter a number between 1 and 8.")

    start = input("Start time (sec, blank for start): ").strip() or None
    end   = input("End time (sec, blank for end): ").strip() or None

    use_key = input("Was a key used to shuffle? (Y/N): ").strip().lower() == 'y'
    if use_key:
        key_text = input("Enter key/passphrase: ").strip()
        if not key_text:
            raise ValueError("❌ Key/passphrase cannot be empty.")
        decode_audio_with_key(input_file, key_text, n_bits, start, end)
    else:
        decode_audio(input_file, n_bits, start, end)
