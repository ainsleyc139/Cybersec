import cv2
import numpy as np
import wave
import hashlib

def hash_to_seed(key_text: str) -> int:
    if not isinstance(key_text, str) or not key_text.strip():
        raise ValueError("Key/passphrase is required.")
    return int.from_bytes(hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little", signed=False)

def encode_audio_with_key(file_name, is_file, secret_data_file, output_name,
                          n_bits=1, start_time=None, end_time=None, key_text: str = ""):
    with wave.open(file_name, "rb") as audio:
        params = audio.getparams()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        num_frames = audio.getnframes()
        frames = bytearray(audio.readframes(num_frames))

    frame_size = num_channels * sample_width
    total_bytes = len(frames)

    # Normalize start/end times
    start_sec = float(start_time) if (start_time is not None and str(start_time).strip() != "") else None
    end_sec   = float(end_time)   if (end_time   is not None and str(end_time).strip()   != "") else None

    if start_sec is not None and end_sec is not None:
        start_frame = max(0, min(int(start_sec * sample_rate), num_frames))
        end_frame   = max(0, min(int(end_sec   * sample_rate), num_frames))
        if end_frame <= start_frame:
            raise ValueError("End time must be greater than start time.")
        start_byte = start_frame * frame_size
        end_byte   = end_frame   * frame_size
    else:
        start_byte = 0
        end_byte   = total_bytes

    binary_secret_data = get_payload(secret_data_file, is_file)

    # Optional: include LSB in your header for verification during decode
    # (If you do this, add ;lsb:{n_bits} to your header inside get_payload)

    # Pad to multiple of n_bits
    rem = len(binary_secret_data) % n_bits
    if rem != 0:
        binary_secret_data += '0' * (n_bits - rem)

    data_len_bits = len(binary_secret_data)
    num_bytes_range = end_byte - start_byte
    max_bits = num_bytes_range * n_bits

    print(f"[*] Encoding payload of {data_len_bits} bits into bytes [{start_byte}:{end_byte}) using {n_bits} LSB(s) with key")

    if data_len_bits > max_bits:
        raise ValueError("[!] Insufficient capacity in selected audio range. Use fewer bits, a longer range, or smaller payload.")

    # Build byte index order and shuffle with key
    order = np.arange(start_byte, end_byte, dtype=np.int64)
    seed = hash_to_seed(key_text)
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    payload_index = 0
    mask = ~((1 << n_bits) - 1) & 0xFF

    for i in order:
        if payload_index >= data_len_bits:
            break
        bits_chunk = binary_secret_data[payload_index: payload_index + n_bits]
        bits_value = int(bits_chunk, 2)
        frames[i] = (frames[i] & mask) | bits_value
        payload_index += n_bits

    with wave.open(output_name, "wb") as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(frames)

    print(f"[+] Data encoded successfully into {output_name}")
