import sys, os, base64, wave, importlib.util, tempfile
import cv2, numpy as np
import pygame, tempfile, time
import hashlib
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from encode_audio import encode_audio
from decode_audio import decode_audio
pygame.mixer.init()
from PySide6.QtCore import Qt, QRect, QSize, QPoint, Signal, QUrl, QTimer
from PySide6.QtGui import QColor, QPalette, QPainter, QBrush, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QFormLayout, QTabWidget, QSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QStackedWidget, QSizePolicy, QRadioButton,
    QRubberBand, QSplitter, QSlider, QProgressBar, QDialog
)
import wave

# Try to import pydub for MP3 support
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("Warning: pydub not installed. MP3 support disabled. Install with: pip install pydub")

# ==========================
# Load Image Backends (ENCODE_BMP / DECODE_BMP)
# ==========================
def _load_module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_image_backends():
    # Try local folder first
    try:
        from ENCODE_BMP import encode as bmp_encode
        from DECODE_BMP import decode as bmp_decode
        return bmp_encode, bmp_decode
    except Exception:
        pass
    # Try /mnt/data (for this environment)
    try:
        enc_mod = _load_module_from("/mnt/data/ENCODE_BMP.py", "ENCODE_BMP")
        dec_mod = _load_module_from("/mnt/data/DECODE_BMP.py", "DECODE_BMP")
        return enc_mod.encode, dec_mod.decode
    except Exception as e:
        raise ImportError(
            "Could not import ENCODE_BMP.py / DECODE_BMP.py from local folder or /mnt/data.\n"
            f"Details: {e}"
        )

bmp_encode, bmp_decode = _load_image_backends()

# ==========================
# Audio conversion helpers
# ==========================
def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to WAV and return the temporary WAV path."""
    if not HAS_PYDUB:
        raise ImportError("pydub is required for MP3 support. Install with: pip install pydub")
    
    try:
        # Load MP3 and convert to WAV
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav.close()
        
        # Export as WAV
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        raise Exception(f"Failed to convert MP3 to WAV: {e}")

def get_audio_duration(audio_path):
    """Get duration of audio file (supports both WAV and MP3)."""
    if audio_path.lower().endswith('.wav'):
        with wave.open(audio_path, "rb") as wf:
            frame_rate = wf.getframerate()
            total_frames = wf.getnframes()
            return total_frames / frame_rate
    elif audio_path.lower().endswith('.mp3') and HAS_PYDUB:
        audio = AudioSegment.from_mp3(audio_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    else:
        raise ValueError("Unsupported audio format or missing dependencies")

class AudioPlayerWidget(QWidget):
    """Audio player widget with play/pause/stop controls using pygame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.duration = 0
        self.position = 0
        self.is_playing = False
        self.is_paused = False
        self.volume = 0.7

        self.init_ui()

        # Timer for updating progress bar
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_position)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(6)

        self.file_label = QLabel("No audio loaded")
        self.file_label.setStyleSheet("color:#aaa; font-size:12px;")
        layout.addWidget(self.file_label)

        # Progress + time
        time_layout = QHBoxLayout()
        self.time_current = QLabel("0:00")
        self.time_total = QLabel("0:00")
        self.time_current.setStyleSheet("color:#999; font-size:10px;")
        self.time_total.setStyleSheet("color:#999; font-size:10px;")
        time_layout.addWidget(self.time_current)
        time_layout.addStretch()
        time_layout.addWidget(self.time_total)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)

        layout.addLayout(time_layout)
        layout.addWidget(self.progress_bar)

        # Controls
        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play_pause)
        self.play_btn.setEnabled(False)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)

        vol_label = QLabel("Vol:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)

        controls.addWidget(self.play_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(vol_label)
        controls.addWidget(self.volume_slider)
        controls.addStretch()

        layout.addLayout(controls)
        self.setLayout(layout)

    def load_audio(self, file_path):
        self.current_file = file_path
        self.position = 0
        self.is_playing = False
        self.is_paused = False

        try:
            if file_path.lower().endswith(".mp3") and HAS_PYDUB:
                seg = AudioSegment.from_mp3(file_path)
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                seg.export(temp_wav.name, format="wav")
                self.current_file = temp_wav.name
            self.duration = get_audio_duration(file_path)
            mins, secs = divmod(int(self.duration), 60)
            self.file_label.setText(f"{os.path.basename(file_path)} ({mins}:{secs:02d})")
            self.time_total.setText(f"{mins}:{secs:02d}")
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            self.file_label.setText(f"Error: {e}")

    def play_pause(self):
        if not self.current_file:
            return
        if not self.is_playing:
            pygame.mixer.music.load(self.current_file)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            self.play_btn.setText("⏸ Pause")
            self.update_timer.start(500)
        elif self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.play_btn.setText("⏸ Pause")
        else:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.play_btn.setText("▶ Resume")

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.progress_bar.setValue(0)
        self.time_current.setText("0:00")
        self.play_btn.setText("▶ Play")
        self.update_timer.stop()

    def set_volume(self, value):
        self.volume = value / 100.0
        pygame.mixer.music.set_volume(self.volume)

    def update_position(self):
        if self.is_playing and not self.is_paused and self.duration > 0:
            pos = pygame.mixer.music.get_pos() / 1000  # ms → sec
            percentage = (pos / self.duration) * 100
            self.progress_bar.setValue(int(min(100, percentage)))
            mins, secs = divmod(int(pos), 60)
            self.time_current.setText(f"{mins}:{secs:02d}")

# ==========================
# Audio backend (integrated from encode_audio.py and decode_audio.py)
# ==========================

def to_bin(data):
    """Convert data into binary format as string"""
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(int(i), "08b") for i in data]
    elif isinstance(data, (int, np.integer)):
        return format(int(data), "08b")
    else:
        raise TypeError(f"Type not supported: {type(data)}")

def to_float_or_none(x):
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    return float(s)

def hash_to_seed(key_text: str) -> int:
    """Convert key text to seed for PRNG"""
    if not isinstance(key_text, str) or not key_text.strip():
        raise ValueError("Key/passphrase is required.")
    return int.from_bytes(hashlib.sha256(key_text.encode("utf-8")).digest()[:8], "little", signed=False) 

def get_payload(file_path, is_file):
    """Prepare payload with header information"""
    if is_file:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"❌ File '{file_path}' does not exist. Select a valid file.")
        with open(file_path, "rb") as f:
            byte_data = f.read()
        # Get the file extension (e.g., '.txt')
        extension = os.path.splitext(file_path)[1]  # Better than split('.')
        if not extension:
            extension = ".bin"  # default
        header_marker = f"<type:{extension[1:]};size:{len(byte_data)}>"  # Remove the dot from extension
        header_bin = ''.join(to_bin(c) for c in header_marker)
        content_bin = ''.join(to_bin(byte) for byte in byte_data)
        end_bin = to_bin("=====")
    else: 
        # Text payload: header with 'nil' filetype
        header_marker = f"<type:nil;size:{len(file_path)}>"
        header_bin = ''.join(to_bin(c) for c in header_marker)
        content_bin = to_bin(file_path)
        end_bin = to_bin("=====")
    return header_bin + content_bin + end_bin

def decode_audio_header(binary_data):
    """Decode header from binary data"""
    header_data = ""
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i+8]
        char = chr(int(byte, 2))
        header_data += char
        if char == '>':
            break
    header_length = len(header_data) * 8
    return header_data, header_length

def encode_audio_with_key(file_name, is_file, secret_data_file, output_name, key, n_bits=1, start_time=None, end_time=None):
    """Encode audio with key-based shuffling"""
    n_bits = int(n_bits)
    with wave.open(file_name, "rb") as audio:
        params = audio.getparams()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))
    
    frame_size = num_channels * sample_width
    total_bytes = len(frames)
    max_bits = total_bytes * n_bits

    # convert user input times to float or none
    start_time = to_float_or_none(start_time)
    end_time = to_float_or_none(end_time)

    # convert time into frames, then into bytes
    if start_time is not None and end_time is not None:
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        # calculate start and end byte
        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size
        if end_byte > total_bytes:
            raise ValueError("[!] End time exceeds file length")
        max_bits = (end_byte - start_byte) * n_bits
    else:
        start_byte = 0
        end_byte = total_bytes
            
    # read payload file to get payload (binary)
    binary_secret_data = get_payload(secret_data_file, is_file)
    
    while len(binary_secret_data) % n_bits != 0:
        binary_secret_data += '0'

    data_len = len(binary_secret_data)

    if data_len > max_bits:
        raise ValueError("[!] Insufficient bytes, need bigger audio or fewer bits per channel.")

    # Split the secret data into chunks of n_bits length so we can shuffle it with key
    bits_chunks = []
    for i in range(0, data_len, n_bits):
        bits_chunks.append(binary_secret_data[i:i+n_bits])
    
    # Build shuffled order using PRNG/seed
    order = np.arange(start_byte, end_byte, dtype=np.int64)
    seed = hash_to_seed(key)
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    # pad the length to match the end time so that we can decode it later
    while len(bits_chunks) < len(order):
        bits_chunks.append('0' * n_bits)

    # Use shuffled order to encode
    for i, pos in enumerate(order):
        bits_value = int(bits_chunks[i], 2)
        # Clear the LSBs and insert new bits
        mask = ~((1 << n_bits) - 1) & 0xFF
        frames[pos] = (frames[pos] & mask) | bits_value

    # Save modified audio 
    with wave.open(output_name, "wb") as output:
        output.setparams(params)
        output.writeframes(frames)

def decode_audio_with_key(file_name, key, start_time=None, end_time=None, n_bits=1):
    """Decode audio with key-based shuffling.
    If start_time or end_time are not given, decode the entire file automatically.
    """
    import wave, base64, numpy as np
    n_bits = int(n_bits)
    with wave.open(file_name, "rb") as audio:
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        num_channels = audio.getnchannels()
        frames = bytearray(audio.readframes(audio.getnframes()))
    if frames is None:
        raise FileNotFoundError(f"❌ Could not open {file_name}. Check path and extension.")
    frame_size = num_channels * sample_width
    total_bytes = len(frames)

    # --- Handle time range ---
    if start_time is None or end_time is None:
        # Default: use full audio
        start_byte = 0
        end_byte = total_bytes
        max_bits = (end_byte - start_byte) * n_bits
    else:
        start_frame = int(float(start_time) * sample_rate)
        end_frame = int(float(end_time) * sample_rate)
        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size
        if end_byte > total_bytes:
            raise ValueError("[!] End time exceeds file length")
        max_bits = (end_byte - start_byte) * n_bits

    # --- Build shuffled order with same seed ---
    # Now start_byte and end_byte are guaranteed to be defined
    order = np.arange(start_byte, end_byte, dtype=np.int64)
    seed = hash_to_seed(key)  # Use the global function
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(order)

    # --- Collect bits ---
    bits_buffer = []
    for pos in order:
        bits = frames[pos] & ((1 << n_bits) - 1)
        bits_buffer.append(f'{bits:0{n_bits}b}')
    binary_data = ''.join(bits_buffer)

    # --- Look for header ---
    marker = "<type:"
    binary_marker = ''.join([format(ord(c), "08b") for c in marker])
    header_start = binary_data.find(binary_marker)
    if header_start == -1:
        raise ValueError("❌ Header marker not found in decoded data")

    # Decode header string until ">"
    header_bits = binary_data[header_start:]
    header_data = ""
    for i in range(0, len(header_bits), 8):
        byte = header_bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(byte, 2))
        header_data += char
        if char == ">":
            break

    header_parts = header_data.strip("<>").split(';')
    payload_extension = '.' + header_parts[0][5:]
    payload_size = int(header_parts[1][5:])

    # --- Extract payload ---
    start_idx = header_start + len(header_data) * 8
    if payload_extension == ".nil":
        # Text payload
        payload_bits = binary_data[start_idx:]
        decoded_text = ""
        for i in range(0, len(payload_bits), 8):
            byte = payload_bits[i:i+8]
            if len(byte) < 8:
                break
            decoded_text += chr(int(byte, 2))
        
        # Remove the ===== delimiter and everything after it
        if "=====" in decoded_text:
            decoded_text = decoded_text.split("=====")[0]
        
        return decoded_text
    else:
        # File payload
        file_bits = binary_data[start_idx:start_idx + payload_size * 8]
        file_bytes = bytearray(int(file_bits[i:i+8], 2) for i in range(0, len(file_bits), 8))
        return base64.b64encode(file_bytes).decode("utf-8")

# Legacy function names for compatibility
def encode_audio(wav_path, secret_data_text, output_path, n_bits=1, key=None):
    """Legacy wrapper for GUI compatibility - encodes text payload"""
    if key is None:
        raise ValueError("Key is required for encoding (audio).")
    # Create a temporary file with the text content
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(secret_data_text)
        temp_file_path = temp_file.name
    try:
        encode_audio_with_key(wav_path, False, secret_data_text, output_path, str(key), n_bits)
    finally:
        try:
            os.unlink(temp_file_path)
        except:
            pass

def decode_audio(wav_path, n_bits=1, key=None):
    """Legacy wrapper for GUI compatibility - simplified decode for text"""
    if key is None:
        raise ValueError("Key is required for decoding (audio).")
    # For GUI integration, we'll assume full audio decode without time segments for now
    # This is a simplified version - the GUI will use the full function with segments
    with wave.open(wav_path, "rb") as audio:
        sample_rate = audio.getframerate()
        total_frames = audio.getnframes()
        total_duration = total_frames / sample_rate
    
    # Use the full audio duration
    return decode_audio_with_key(wav_path, str(key), 0, total_duration, n_bits)

# ==========================
# Small helpers
# ==========================
def load_binary_as_text(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_text_as_binary(text, output_path):
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(text.encode("utf-8")))

def style_line_edit(le: QLineEdit):
    le.setStyleSheet("QLineEdit { color:#fff; background:#222; border:1px solid #444; padding:6px; }")

def style_text_edit(te: QTextEdit):
    te.setStyleSheet("QTextEdit { color:#fff; background:#222; border:1px solid #444; }")
def _mask_key_display(k: str, keep_tail: int = 4) -> str:
    k = (k or "").strip()
    if not k: return "(empty)"
    if len(k) <= keep_tail:
        return "•" * (len(k) - 1) + k[-1]
    return "•" * (len(k) - keep_tail) + k[-keep_tail:]

def _fmt_bytes(n: int) -> str:
    n = int(n)
    units = ["B","KB","MB","GB","TB"]
    i = 0
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}" if i else f"{int(n)} {units[i]}"


# ==========================
# UI widgets
# ==========================
class ToggleSwitch(QWidget):
    """Ball left = Encode (False). Ball right = Decode (True)."""
    toggled = Signal(bool)
    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        self._checked = bool(checked)
        self.setFixedSize(64, 32)
        self.setCursor(Qt.PointingHandCursor)
    def isChecked(self): return self._checked
    def setChecked(self, v: bool):
        v = bool(v)
        if v != self._checked:
            self._checked = v
            self.toggled.emit(self._checked)
            self.update()
    def mousePressEvent(self, _): self.setChecked(not self._checked)
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        track = QColor("#4CAF50") if self._checked else QColor("#777")
        p.setBrush(track); p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 6, self.width(), 20, 10, 10)
        thumb_d = 24
        x = self.width() - thumb_d - 4 if self._checked else 4
        p.setBrush(QBrush(QColor("#fff")))
        p.drawEllipse(x, 4, thumb_d, thumb_d)

class DraggableLabel(QLabel):
    fileDropped = Signal(str)
    def __init__(self, text="Drag & Drop File Here\nor click to browse", parent=None, min_height=90):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setText(text)
        self.setStyleSheet("""
            border: 2px dashed #aaa;
            border-radius: 10px;
            background-color: #1e1e1e;
            color: #ccc;
            font-size: 13px;
            padding: 10px;
        """)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())

class ImagePreviewSelector(QLabel):
    """Large image preview that allows region selection with a rubber band."""
    regionSelected = Signal(int, int, int, int)  # x1, y1, x2, y2 (image coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color:#111; border:1px solid #333;")
        self.setMinimumHeight(360)
        self._pix = None
        self._img_w = 0
        self._img_h = 0
        self._scaled_rect = QRect()  # where the image is drawn inside the label
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = QPoint()
        self._selecting = False

    def set_image(self, path_or_pixmap):
        if isinstance(path_or_pixmap, QPixmap):
            pm = path_or_pixmap
        else:
            pm = QPixmap(path_or_pixmap)
        if pm.isNull():
            self._pix = None
            self.setText("Failed to load preview")
            return
        self._pix = pm
        self._img_w = pm.width()
        self._img_h = pm.height()
        self._update_scaled_rect()
        self.setText("")  # clear any placeholder
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_scaled_rect()

    def _update_scaled_rect(self):
        if not self._pix:
            self._scaled_rect = QRect()
            return
        # Fit pixmap inside label preserving aspect ratio
        avail_w = self.width() - 12
        avail_h = self.height() - 12
        scaled = self._pix.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._scaled_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def mousePressEvent(self, ev):
        if not self._pix or not self._scaled_rect.contains(ev.pos()):
            return
        self._selecting = True
        self._origin = ev.pos()
        self._rubber.setGeometry(QRect(self._origin, QSize()))
        self._rubber.show()

    def mouseMoveEvent(self, ev):
        if self._selecting:
            rect = QRect(self._origin, ev.pos()).normalized()
            rect = rect.intersected(self._scaled_rect)  # constrain to image
            self._rubber.setGeometry(rect)

    def mouseReleaseEvent(self, ev):
        if not self._selecting:
            return
        self._selecting = False
        sel = self._rubber.geometry().intersected(self._scaled_rect)
        self._rubber.hide()
        if sel.width() < 2 or sel.height() < 2:
            return  # ignore tiny
        # Map from view coords to image coords
        x_ratio = self._img_w / self._scaled_rect.width()
        y_ratio = self._img_h / self._scaled_rect.height()
        x1 = int((sel.left() - self._scaled_rect.left()) * x_ratio)
        y1 = int((sel.top() - self._scaled_rect.top()) * y_ratio)
        x2 = int((sel.right() - self._scaled_rect.left()) * x_ratio)
        y2 = int((sel.bottom() - self._scaled_rect.top()) * y_ratio)
        self.regionSelected.emit(x1, y1, x2, y2)


# ==========================
# Waveform Comparison Widget
# ==========================
class WaveformComparisonWidget(QWidget):
    """A widget to display original and stego audio waveforms side by side."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_data = None
        self.stego_data = None
        self.original_path = None
        self.stego_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def clear(self):
        """Clear both waveforms."""
        self.original_data = None
        self.stego_data = None
        self.original_path = None
        self.stego_path = None
        self.figure.clear()
        self.canvas.draw()

    def load_original(self, wav_path):
        """Load and plot original audio waveform."""
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()

            # Convert raw bytes to numpy array
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sample_width, np.int16)
            audio_data = np.frombuffer(frames, dtype=dtype)

            # Handle stereo
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                audio_data = audio_data.mean(axis=1)  # Average channels

            # Normalize to [-1, 1]
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() != audio_data.min():
                audio_data = (audio_data - audio_data.min()) / (audio_data.max() - audio_data.min())
                audio_data = audio_data * 2 - 1  # Scale to [-1, 1]

            self.original_data = audio_data
            self.original_path = wav_path
            self._update_plot()

        except Exception as e:
            print(f"Error loading original audio: {e}")
            self.original_data = None
            self.original_path = None
            self._update_plot()

    def load_stego(self, wav_path):
        """Load and plot stego audio waveform."""
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sample_width, np.int16)
            audio_data = np.frombuffer(frames, dtype=dtype)

            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                audio_data = audio_data.mean(axis=1)

            audio_data = audio_data.astype(np.float32)
            if audio_data.max() != audio_data.min():
                audio_data = (audio_data - audio_data.min()) / (audio_data.max() - audio_data.min())
                audio_data = audio_data * 2 - 1

            self.stego_data = audio_data
            self.stego_path = wav_path
            self._update_plot()

        except Exception as e:
            print(f"Error loading stego audio: {e}")
            self.stego_data = None
            self.stego_path = None
            self._update_plot()

    def _update_plot(self):
        """Update the waveform plot with both signals."""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        # Plot original
        if self.original_data is not None:
            time_axis = np.linspace(0, len(self.original_data) / 44100, num=len(self.original_data))  # Assume 44.1kHz
            ax.plot(time_axis, self.original_data, color='lightblue', linewidth=0.8)
            ax.set_title("Original Audio", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No original audio\nloaded", horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title("Original Audio", fontsize=10)

        # Plot stego
        if self.stego_data is not None:
            time_axis = np.linspace(0, len(self.stego_data) / 44100, num=len(self.stego_data))
            ax2.plot(time_axis, self.stego_data, color='orange', linewidth=0.8)
            ax2.set_title("Stego Audio", fontsize=10)
            ax2.set_xlabel("Time (s)", fontsize=8)
            ax2.set_ylabel("Amplitude", fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No stego audio\nloaded", horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes, fontsize=10, color='gray')
            ax2.set_title("Stego Audio", fontsize=10)

        self.figure.tight_layout()
        self.canvas.draw()
        
# ==========================
# Main Window
# ==========================
class StegoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔒 LSB Steganography & Steganalysis Tool (AY25-ACW1)")
        self.resize(1200, 850)

        # Per-media, per-mode widget state
        self.state = {
            'image': {'encode': {}, 'decode': {}, 'last_stego': None, 'last_extracted_bytes': None, 'last_extracted_text': None},
            'audio': {'encode': {}, 'decode': {}, 'last_stego': None, 'last_extracted_bytes': None, 'last_extracted_text': None},
        }
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout()
        tabs = QTabWidget()

        img_tab = QWidget(); self.setup_media_tab(img_tab, "image")
        aud_tab = QWidget(); self.setup_media_tab(aud_tab, "audio")

        tabs.addTab(img_tab, "🖼️ Image / GIF")
        tabs.addTab(aud_tab, "🔊 Audio")

        root.addWidget(tabs)
        w = QWidget(); w.setLayout(root)
        self.setCentralWidget(w)

    # --- Build a media tab with header (Encode | switch | Decode) and distinct pages ---
    def setup_media_tab(self, parent_widget, media_type):
        v = QVBoxLayout()

        header = QHBoxLayout()
        header.addStretch(1)
        lbl_enc = QLabel("Encode"); lbl_enc.setStyleSheet("color:#ddd; font-weight:600;")
        toggle = ToggleSwitch(checked=False)
        lbl_dec = QLabel("Decode"); lbl_dec.setStyleSheet("color:#ddd; font-weight:600;")
        header.addWidget(lbl_enc); header.addSpacing(12); header.addWidget(toggle); header.addSpacing(12); header.addWidget(lbl_dec)
        header.addStretch(1)
        v.addLayout(header)

        stack = QStackedWidget()
        enc_page = QWidget(); dec_page = QWidget()
        self.build_encode_page(enc_page, media_type)
        self.build_decode_page(dec_page, media_type)
        stack.addWidget(enc_page)   # 0 = Encode
        stack.addWidget(dec_page)   # 1 = Decode
        toggle.toggled.connect(lambda checked: stack.setCurrentIndex(1 if checked else 0))

        v.addWidget(stack)
        parent_widget.setLayout(v)

    def _show_encode_summary(self, *, media: str, cover: str, output: str,
                         payload_label: str, payload_size: int,
                         n_bits: int, key_text: str,
                         extra_lines: list[str] | None = None):
        lines = [
            f"<b>Media:</b> {media}",
            f"<b>Cover:</b> {os.path.basename(cover)}",
            f"<b>Output:</b> {os.path.basename(output)}",
            f"<b>Payload:</b> {payload_label}",
            f"<b>Payload Size:</b> {_fmt_bytes(payload_size)}",
            f"<b>LSBs Used:</b> {n_bits}",
            f"<b>Key (masked):</b> {_mask_key_display(key_text)}",
        ]
        if extra_lines:
            lines += extra_lines

        box = QMessageBox(self)
        box.setWindowTitle("Encoding Summary")
        box.setIcon(QMessageBox.Information)
        box.setText("✅ Encoding completed")
        box.setInformativeText("<br>".join(lines))
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()


    # --- Encode page ---
    def build_encode_page(self, page: QWidget, media: str):
        s = self.state[media]['encode']

        # Use a splitter to give big space to preview
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)

        # LEFT PANE (compact inputs)
        left = QWidget()
        lv = QVBoxLayout()

        # COVER
        cover_grp = QGroupBox("📁 Cover")
        cv = QVBoxLayout()
        s['cover_label'] = DraggableLabel(min_height=80)
        s['cover_label'].fileDropped.connect(lambda p, m=media: self._set_file(m, 'encode', 'cover', p))
        cv.addWidget(s['cover_label'])
        b1 = QPushButton("Choose Cover"); b1.clicked.connect(lambda checked=False, m=media: self._browse_file(m, 'encode', 'cover'))
        cv.addWidget(b1)
        cover_grp.setLayout(cv)

        # PAYLOAD
        payload_grp = QGroupBox("📥 Payload")
        pv = QVBoxLayout()
        s['payload_label'] = DraggableLabel("Drop Payload Here\nor click to browse", min_height=80)
        s['payload_label'].fileDropped.connect(lambda p, m=media: self._set_file(m, 'encode', 'payload', p))
        pv.addWidget(s['payload_label'])
        s['payload_choose_btn'] = QPushButton("Choose Payload")
        s['payload_choose_btn'].clicked.connect(lambda checked=False, m=media: self._browse_file(m, 'encode', 'payload'))
        pv.addWidget(s['payload_choose_btn'])
        payload_grp.setLayout(pv)

        # SETTINGS
        settings = QGroupBox("⚙️ Settings")
        form = QFormLayout()
        s['lsb'] = QSpinBox(); s['lsb'].setRange(1, 8); s['lsb'].setValue(1)
        form.addRow("LSBs:", s['lsb'])

        #key
        s['key'] = QLineEdit(); s['key'].setEchoMode(QLineEdit.Password); s['key'].setPlaceholderText("(Required) Integer key for scrambling")
        style_line_edit(s['key'])
        form.addRow("Key:", s['key'])

        # ADD THIS FOR AUDIO:
        if media == "audio":
            s['mode_text'] = QRadioButton("Text")
            s['mode_file'] = QRadioButton("File")
            s['mode_text'].setChecked(True)  # Default to Text for audio
            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel("Payload Type:"))
            mode_row.addWidget(s['mode_text'])
            mode_row.addWidget(s['mode_file'])
            form.addRow(mode_row)

            s['text_input'] = QTextEdit()
            s['text_input'].setMaximumHeight(80)
            s['text_input'].setPlaceholderText("Enter secret message (Text mode)")
            style_text_edit(s['text_input'])
            form.addRow("Text:", s['text_input'])

            # Connect toggle handlers
            s['mode_text'].toggled.connect(lambda _c, m=media: self._update_payload_ui_audio(m))
            s['mode_file'].toggled.connect(lambda _c, m=media: self._update_payload_ui_audio(m))

        settings.setLayout(form)
        if media == "image":
            s['mode_text'] = QRadioButton("Text")
            s['mode_file'] = QRadioButton("File")
            s['mode_file'].setChecked(True)
            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel("Payload Type:"))
            mode_row.addWidget(s['mode_text'])
            mode_row.addWidget(s['mode_file'])
            form.addRow(mode_row)

            s['text_input'] = QTextEdit()
            s['text_input'].setMaximumHeight(80)

            style_text_edit(s['text_input'])  # Use your existing style_text_edit function
            s['text_input'].setPlaceholderText("Enter secret message (Text mode)")
            style_line_edit(s['text_input'])
            form.addRow("Text:", s['text_input'])

            # Region fields (auto-filled from preview selection)
            s['x1'] = QSpinBox(); s['x1'].setRange(0, 10000)
            s['y1'] = QSpinBox(); s['y1'].setRange(0, 10000)
            s['x2'] = QSpinBox(); s['x2'].setRange(0, 10000)
            s['y2'] = QSpinBox(); s['y2'].setRange(0, 10000)
            row = QHBoxLayout()
            row.addWidget(QLabel("x1:")); row.addWidget(s['x1'])
            row.addWidget(QLabel("y1:")); row.addWidget(s['y1'])
            row.addWidget(QLabel("x2:")); row.addWidget(s['x2'])
            row.addWidget(QLabel("y2:")); row.addWidget(s['y2'])
            form.addRow("Region:", row)

            # Toggle visibility based on payload type
            s['mode_text'].toggled.connect(lambda _c, m=media: self._update_payload_ui(m))
            s['mode_file'].toggled.connect(lambda _c, m=media: self._update_payload_ui(m))
        # --- AUDIO SEGMENT SELECTOR (only for audio) ---
        elif media == "audio":
            segment_grp = QGroupBox("⏱️ Select Audio Segment")
            seg_layout = QVBoxLayout()
            seg_layout.setSpacing(8)

            # Duration label (updated when WAV is loaded)
            s['duration_label'] = QLabel("Duration: — sec")
            s['duration_label'].setStyleSheet("color:#aaa; font-size:12px;")
            seg_layout.addWidget(s['duration_label'])

            # Start slider
            start_hbox = QHBoxLayout()
            start_hbox.addWidget(QLabel("Start:"))
            s['start_slider'] = QSlider(Qt.Horizontal)  # Fixed: Use QSlider
            s['start_slider'].setMinimum(0)
            s['start_slider'].setMaximum(60)  # default max 60 sec
            s['start_slider'].setValue(0)
            s['start_slider'].setTickPosition(QSlider.TicksBelow)  # Fixed: Use QSlider.TicksBelow
            s['start_slider'].setTickInterval(5)
            s['start_time_label'] = QLabel("0.0 sec")
            start_hbox.addWidget(s['start_slider'])
            start_hbox.addWidget(s['start_time_label'])
            seg_layout.addLayout(start_hbox)

            # End slider
            end_hbox = QHBoxLayout()
            end_hbox.addWidget(QLabel("End:"))
            s['end_slider'] = QSlider(Qt.Horizontal)  # Fixed: Use QSlider
            s['end_slider'].setMinimum(1)
            s['end_slider'].setMaximum(60)
            s['end_slider'].setValue(10)
            s['end_slider'].setTickPosition(QSlider.TicksBelow)  # Fixed: Use QSlider.TicksBelow
            s['end_slider'].setTickInterval(5)
            s['end_time_label'] = QLabel("10.0 sec")
            end_hbox.addWidget(s['end_slider'])
            end_hbox.addWidget(s['end_time_label'])
            seg_layout.addLayout(end_hbox)

            # Progress bar visualization
            s['segment_progress'] = QProgressBar()
            s['segment_progress'].setRange(0, 100)
            s['segment_progress'].setValue(0)
            s['segment_progress'].setTextVisible(False)
            s['segment_progress'].setStyleSheet("""
                QProgressBar { 
                    border: 1px solid #333; 
                    border-radius: 4px; 
                    background: #222; 
                    height: 8px;
                }
                QProgressBar::chunk {
                    background: #4CAF50;
                    border-radius: 3px;
                }
            """)
            seg_layout.addWidget(s['segment_progress'])

            # Segment info label
            s['segment_info'] = QLabel("Selected: 0.0–10.0s")
            s['segment_info'].setStyleSheet("color:#4CAF50; font-weight:bold;")
            seg_layout.addWidget(s['segment_info'])

            segment_grp.setLayout(seg_layout)
            lv.addWidget(segment_grp)

        # ACTIONS
        # ACTIONS
        act = QHBoxLayout()

        if media == "audio":
            # Audio: Calculate = capacity popup, Encode = actually embed
            enc_btn = QPushButton("🚀 Calculate Size")
            enc_btn.clicked.connect(lambda checked=False: self._run_encoding_audio(preview_only=True))

            save_btn = QPushButton("💾 Encode and Save…")
            save_btn.clicked.connect(lambda checked=False: self._run_encoding_audio(preview_only=False))
        else:
            # Image keeps the existing logic (calculate + save separately)
            enc_btn = QPushButton("🚀 Calculate Size")
            enc_btn.clicked.connect(lambda checked=False, m=media: self.run_encoding(m))

            save_btn = QPushButton("💾 Encode and Save…")
            save_btn.clicked.connect(lambda checked=False, m=media: self._save_last_stego(m))

        act.addWidget(enc_btn)
        act.addWidget(save_btn)
        act.addStretch(1)

        # act = QHBoxLayout()
        # enc_btn = QPushButton("🚀 Calculate Size"); enc_btn.clicked.connect(lambda checked=False, m=media: self.run_encoding(m))
        # save_btn = QPushButton("💾 Encode and Save…"); save_btn.clicked.connect(lambda checked=False, m=media: self._save_last_stego(m))
        # # enc_btn = QPushButton("🚀 Calculate Size")
        # enc_btn.clicked.connect(lambda checked=False, m=media: self._run_encoding_audio(preview_only=True))

        # save_btn = QPushButton("💾 Encode and Save…")
        # save_btn.clicked.connect(lambda checked=False, m=media: self._run_encoding_audio(preview_only=False))

        # act.addWidget(enc_btn); act.addWidget(save_btn); act.addStretch(1)

        s['status'] = QLabel(); s['status'].setWordWrap(True); s['status'].setStyleSheet("color:#9ad;")

        lv.addWidget(cover_grp)
        lv.addWidget(payload_grp)
        lv.addWidget(settings)
        lv.addLayout(act)
        lv.addWidget(s['status'])
        left.setLayout(lv)

        # RIGHT PANE (preview or placeholder)
        right = QWidget()
        rv = QVBoxLayout()
        if media == "image":
            s['preview'] = ImagePreviewSelector()
            s['preview'].regionSelected.connect(lambda x1,y1,x2,y2, m=media: self._apply_region_from_preview(m, x1,y1,x2,y2))
            rv.addWidget(s['preview'])
            reset_btn = QPushButton("Reset Region"); reset_btn.clicked.connect(lambda checked=False, m=media: self._reset_region(m))
            rv.addWidget(reset_btn, alignment=Qt.AlignRight)
        else:  # audio
            # Create a container for the audio preview and controls
            s['waveform_widget'] = WaveformComparisonWidget()
            rv.addWidget(s['waveform_widget'])

            audio_container = QWidget()
            audio_layout = QVBoxLayout()
            
            # Preview title
            preview_title = QLabel("🎵 Audio Preview")
            preview_title.setAlignment(Qt.AlignCenter)
            preview_title.setStyleSheet("color:#4CAF50; font-weight:bold; font-size:14px; margin-bottom:10px;")
            audio_layout.addWidget(preview_title)
            
            # Audio player for original file
            s['audio_preview'] = AudioPlayerWidget()
            audio_layout.addWidget(s['audio_preview'])
            
            # Separator
            separator = QLabel()
            separator.setFixedHeight(1)
            separator.setStyleSheet("background-color:#333; margin:10px 0;")
            audio_layout.addWidget(separator)
            
            # Stego preview (will be populated after encoding)
            stego_preview_section = QWidget()
            stego_layout = QVBoxLayout()
            
            stego_preview_title = QLabel("🔐 Stego Audio Preview")
            stego_preview_title.setAlignment(Qt.AlignCenter)
            stego_preview_title.setStyleSheet("color:#4CAF50; font-weight:bold; font-size:14px;")
            stego_layout.addWidget(stego_preview_title)
            
            s['stego_preview'] = AudioPlayerWidget()
            stego_layout.addWidget(s['stego_preview'])
            
            stego_preview_section.setLayout(stego_layout)
            stego_preview_section.setVisible(False)  # Initially hidden until encoding happens
            s['stego_preview_section'] = stego_preview_section
            
            audio_layout.addWidget(stego_preview_section)
            
            # Placeholder for additional info
            info_label = QLabel("Load an audio file to see preview and controls.\nSupports WAV and MP3 formats.")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color:#aaa; font-size:12px;")
            audio_layout.addWidget(info_label)
            s['info_label'] = info_label
            
            audio_layout.addStretch(1)
            audio_container.setLayout(audio_layout)
            rv.addWidget(audio_container)
        right.setLayout(rv)

        # Splitter arrangement
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)  # left (compact)
        splitter.setStretchFactor(1, 3)  # right (big preview)

        page_layout = QVBoxLayout()
        page_layout.addWidget(splitter)
        page.setLayout(page_layout)

        # Connect slider updates (only for audio)
        if media == "audio":
            s['start_slider'].valueChanged.connect(lambda v, m=media: self._update_segment_ui(m))
            s['end_slider'].valueChanged.connect(lambda v, m=media: self._update_segment_ui(m))
            self._update_segment_ui(media)  # Initialize UI

        # Initialize payload UI visibility for image
        if media == "image":
            self._update_payload_ui(media)
        elif media == "audio":
            self._update_payload_ui_audio(media)

    # --- Decode page ---
    def build_decode_page(self, page: QWidget, media: str):
        s = self.state[media]['decode']
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        # LEFT PANE
        left = QWidget()
        lv = QVBoxLayout()
        stego_grp = QGroupBox("🔍 Stego")
        sv = QVBoxLayout()
        s['stego_label'] = DraggableLabel(min_height=80)
        s['stego_label'].fileDropped.connect(lambda p, m=media: self._set_file(m, 'decode', 'stego', p))
        sv.addWidget(s['stego_label'])
        b = QPushButton("Choose Stego"); b.clicked.connect(lambda checked=False, m=media: self._browse_file(m, 'decode', 'stego'))
        sv.addWidget(b)
        stego_grp.setLayout(sv)
        settings = QGroupBox("⚙️ Decode Settings")
        form = QFormLayout()
        s['lsb'] = QSpinBox(); s['lsb'].setRange(1, 8); s['lsb'].setValue(1)
        form.addRow("LSBs:", s['lsb'])  # audio uses it; image backend reads header LSB internally
        s['key'] = QLineEdit(); s['key'].setEchoMode(QLineEdit.Password); s['key']
        style_line_edit(s['key'])
        form.addRow("Key:", s['key'])

        # --- Payload Type Toggle (Text vs File) ---
        s['mode_text'] = QRadioButton("Text Payload")
        s['mode_file'] = QRadioButton("File Payload")
        s['mode_text'].setChecked(True)  # Default to Text (most common for audio)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Payload Type:"))
        mode_row.addWidget(s['mode_text'])
        mode_row.addWidget(s['mode_file'])
        form.addRow(mode_row)
        # Add audio segment selector to decode page (only for audio)
        if media == "audio":
            settings.setLayout(form)  # Close settings first
            # Audio segment selector for decode (MANUAL ONLY)
            segment_grp = QGroupBox("⏱️ Select Audio Segment to Decode")
            seg_layout = QVBoxLayout()
            seg_layout.setSpacing(8)
            # Duration label
            s['duration_label'] = QLabel("Duration: — sec")
            s['duration_label'].setStyleSheet("color:#aaa; font-size:12px;")
            seg_layout.addWidget(s['duration_label'])
            # Start slider
            start_hbox = QHBoxLayout()
            start_hbox.addWidget(QLabel("Start:"))
            s['start_slider'] = QSlider(Qt.Horizontal)
            s['start_slider'].setMinimum(0)
            s['start_slider'].setMaximum(60)
            s['start_slider'].setValue(0)
            s['start_slider'].setTickPosition(QSlider.TicksBelow)
            s['start_slider'].setTickInterval(5)
            s['start_time_label'] = QLabel("0.0 sec")
            start_hbox.addWidget(s['start_slider'])
            start_hbox.addWidget(s['start_time_label'])
            seg_layout.addLayout(start_hbox)
            # End slider
            end_hbox = QHBoxLayout()
            end_hbox.addWidget(QLabel("End:"))
            s['end_slider'] = QSlider(Qt.Horizontal)
            s['end_slider'].setMinimum(1)
            s['end_slider'].setMaximum(60)
            s['end_slider'].setValue(10)
            s['end_slider'].setTickPosition(QSlider.TicksBelow)
            s['end_slider'].setTickInterval(5)
            s['end_time_label'] = QLabel("10.0 sec")
            end_hbox.addWidget(s['end_slider'])
            end_hbox.addWidget(s['end_time_label'])
            seg_layout.addLayout(end_hbox)
            # Progress bar
            s['segment_progress'] = QProgressBar()
            s['segment_progress'].setRange(0, 100)
            s['segment_progress'].setValue(0)
            s['segment_progress'].setTextVisible(False)
            s['segment_progress'].setStyleSheet("""
                QProgressBar { 
                    border: 1px solid #333; 
                    border-radius: 4px; 
                    background: #222; 
                    height: 8px;
                }
                QProgressBar::chunk {
                    background: #FF9800;
                    border-radius: 3px;
                }
            """)
            seg_layout.addWidget(s['segment_progress'])
            # Segment info
            s['segment_info'] = QLabel("Mode: Manual Segment Selection")
            s['segment_info'].setStyleSheet("color:#FF9800; font-weight:bold;")
            seg_layout.addWidget(s['segment_info'])
            segment_grp.setLayout(seg_layout)
            lv.addWidget(segment_grp)
            # Ensure sliders are ALWAYS enabled
            s['start_slider'].setEnabled(True)
            s['end_slider'].setEnabled(True)
        else:
            settings.setLayout(form)  # For image mode
        act = QHBoxLayout()
        dec_btn = QPushButton("🔍 Decode"); dec_btn.clicked.connect(lambda checked=False, m=media: self.run_decoding(m))
        save_btn = QPushButton("💾 Save Extracted…"); save_btn.clicked.connect(lambda checked=False, m=media: self._save_extracted(m))
        act.addWidget(dec_btn); act.addWidget(save_btn); act.addStretch(1)
        s['status'] = QLabel(); s['status'].setWordWrap(True); s['status'].setStyleSheet("color:#9ad;")
        lv.addWidget(stego_grp)
        lv.addWidget(settings)
        lv.addLayout(act)
        lv.addWidget(s['status'])
        left.setLayout(lv)
        # RIGHT PANE (big preview or text)
        right = QWidget()
        rv = QVBoxLayout()
        if media == "image":
            s['decode_preview'] = ImagePreviewSelector()
            rv.addWidget(s['decode_preview'])
            note = QLabel("Tip: Region is auto-detected from header.\n(Selection disabled in decode.)")
            note.setStyleSheet("color:#aaa;")
            rv.addWidget(note, alignment=Qt.AlignRight)
        else:
            # Audio preview and controls
            audio_container = QWidget()
            audio_layout = QVBoxLayout()
            # Preview title
            preview_title = QLabel("🎵 Stego Audio Preview")
            preview_title.setAlignment(Qt.AlignCenter)
            preview_title.setStyleSheet("color:#4CAF50; font-weight:bold; font-size:14px; margin-bottom:10px;")
            audio_layout.addWidget(preview_title)
            # Audio player for stego file
            s['decode_audio_preview'] = AudioPlayerWidget()
            audio_layout.addWidget(s['decode_audio_preview'])
            # Separator
            separator = QLabel()
            separator.setFixedHeight(1)
            separator.setStyleSheet("background-color:#333; margin:10px 0;")
            audio_layout.addWidget(separator)
            # Extracted payload preview
            payload_title = QLabel("📄 Extracted Payload")
            payload_title.setAlignment(Qt.AlignCenter)
            payload_title.setStyleSheet("color:#4CAF50; font-weight:bold; font-size:14px; margin:10px 0;")
            audio_layout.addWidget(payload_title)
            s['preview'] = QTextEdit()
            s['preview'].setReadOnly(True)
            s['preview'].setPlaceholderText("Extracted payload...")
            s['preview'].setMinimumHeight(120)
            style_text_edit(s['preview'])
            audio_layout.addWidget(s['preview'])
            # Placeholder info label
            info_label = QLabel("Load a stego audio file and decode to see extracted content")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color:#aaa; font-size:12px;")
            audio_layout.addWidget(info_label)
            s['info_label'] = info_label
            audio_container.setLayout(audio_layout)
            rv.addWidget(audio_container)
        right.setLayout(rv)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        page_layout = QVBoxLayout()
        page_layout.addWidget(splitter)
        page.setLayout(page_layout)
        # Connect slider updates for audio decode
        if media == "audio":
            s['start_slider'].valueChanged.connect(lambda v, m=media: self._update_decode_segment_ui(m))
            s['end_slider'].valueChanged.connect(lambda v, m=media: self._update_decode_segment_ui(m))
            self._update_decode_segment_ui(media)  # Initialize UI

    # ==========================
    # File helpers
    # ==========================
    def _browse_file(self, media, mode, which):
        if media == "image":
            filt = "Image Files (*.bmp *.png *.jpg *.jpeg *.gif);;All Files (*)" if which in ("cover","stego") else "All Files (*)"
        else:
            filt = "Audio Files (*.wav *.mp3);;WAV Audio (*.wav);;MP3 Audio (*.mp3);;All Files (*)" if which in ("cover","stego") else "All Files (*)"
        fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", filt)
        if fname:
            self._set_file(media, mode, which, fname)
    def generate_difference_map(self, cover_path, stego_path):
        try:
            img_cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
            img_stego = cv2.imread(stego_path, cv2.IMREAD_COLOR)
            if img_cover is None or img_stego is None:
                raise ValueError("Failed to load one or both images.")

            # Ensure same size
            if img_cover.shape != img_stego.shape:
                raise ValueError("Cover and stego images must have the same dimensions.")

            # XOR to highlight differences
            diff = cv2.bitwise_xor(img_cover, img_stego)

            # Amplify for visibility (optional, scale difference to full 0–255)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Convert back to 3-channel for display
            diff_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return diff_colored

        except Exception as e:
            print(f"Error generating difference map: {e}")
            return None
    def _update_payload_ui_audio(self, media: str):
        """Toggle between text input and file selector for audio encoding"""
        s = self.state[media]['encode']
        is_text = s.get('mode_text') and s['mode_text'].isChecked()
        
        if 'text_input' in s: 
            s['text_input'].setVisible(is_text)
        if 'payload_label' in s: 
            s['payload_label'].setVisible(not is_text)
        if 'payload_choose_btn' in s: 
            s['payload_choose_btn'].setVisible(not is_text)
    def _show_image_comparison_popup(self, original_path, stego_path, payload_size, n_bits):
        """Show a popup with before/after images, difference map, and capacity details."""
        if not (original_path and stego_path):
            return

        try:
            # --- Compute details ---
            img = cv2.imread(original_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to load cover image.")
            h, w, ch = img.shape
            cover_pixels = h * w
            cover_capacity_bits = cover_pixels * ch * n_bits
            cover_capacity_bytes = cover_capacity_bits // 8

            sufficiency = "✅ Sufficient capacity" if payload_size <= cover_capacity_bytes else "❌ Insufficient capacity"

            # --- Dialog UI ---
            dialog = QDialog(self)
            dialog.setWindowTitle("Before & After Comparison")
            dialog.resize(1400, 650)

            main_layout = QVBoxLayout()

            # Image comparison row
            img_layout = QHBoxLayout()

            # Cover image
            orig_label = QLabel()
            orig_pix = QPixmap(original_path)
            orig_label.setPixmap(orig_pix.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            orig_label.setAlignment(Qt.AlignCenter)
            orig_label.setToolTip("Original Cover Image")
            img_layout.addWidget(orig_label)

            # Stego image
            stego_label = QLabel()
            stego_pix = QPixmap(stego_path)
            stego_label.setPixmap(stego_pix.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            stego_label.setAlignment(Qt.AlignCenter)
            stego_label.setToolTip("Stego Image (with hidden payload)")
            img_layout.addWidget(stego_label)

            # Difference map (highlight LSB changes)
            diff_img = self.generate_difference_map(original_path, stego_path)
            if diff_img is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                cv2.imwrite(temp_file.name, diff_img)
                diff_label = QLabel()
                diff_pix = QPixmap(temp_file.name)
                diff_label.setPixmap(diff_pix.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                diff_label.setAlignment(Qt.AlignCenter)
                diff_label.setToolTip("Difference Map (white = changed pixels)")
                img_layout.addWidget(diff_label)

            main_layout.addLayout(img_layout)

            # Details row
            details = QTextEdit()
            details.setReadOnly(True)
            details.setMinimumHeight(140)
            details.setStyleSheet("color:#ddd; background:#222; border:1px solid #444; font-family:monospace;")
            details.setText(
                f"📊 Cover Details:\n"
                f" - Resolution: {w} x {h}\n"
                f" - Channels: {ch}\n"
                f" - Total Pixels: {cover_pixels}\n"
                f" - LSBs Used: {n_bits}\n"
                f" - Capacity: {cover_capacity_bytes} bytes\n\n"
                f"📦 Payload:\n"
                f" - Size: {payload_size} bytes\n\n"
                f"⚖️ Comparison:\n"
                f" - {sufficiency}\n"
                f" - Stego File: {os.path.basename(stego_path)}"
            )
            main_layout.addWidget(details)

            dialog.setLayout(main_layout)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show comparison:\n{e}")



    def _set_file(self, media, mode, which, path):
        s = self.state[media][mode]
        s[f'{which}_path'] = path
        lbl_key = f'{which}_label'
        if lbl_key in s and isinstance(s[lbl_key], QLabel):
            s[lbl_key].setText(os.path.basename(path))
        # Image previews
        if media == "image":
            if mode == "encode" and which == "cover":
                prev = self.state[media][mode].get('preview')
                if isinstance(prev, ImagePreviewSelector) and os.path.isfile(path):
                    prev.set_image(path)
            if mode == "decode" and which == "stego":
                prev = self.state[media][mode].get('decode_preview')
                if isinstance(prev, ImagePreviewSelector) and os.path.isfile(path):
                    prev.set_image(path)
        # Audio: auto-set duration and handle MP3 conversion

        # Add after the encode cover file handling
        if mode == "decode" and which == "stego":
            if path.lower().endswith(('.wav', '.mp3')):
                try:
                    total_duration = get_audio_duration(path)
                    max_val = int(total_duration)
                    
                    # Set up sliders for decode
                    if 'start_slider' in s:
                        s['start_slider'].setMaximum(max_val)
                        s['end_slider'].setMaximum(max_val)
                        s['start_slider'].setValue(0)
                        s['end_slider'].setValue(min(max_val, 10))
                    
                    if 'duration_label' in s:
                        s['duration_label'].setText(f"Duration: {total_duration:.1f} sec")
                
                except Exception as e:
                    if 'duration_label' in s:
                        s['duration_label'].setText(f"Duration: Unknown")

        if media == "audio":
            if mode == "encode" and which == "cover":
                if path.lower().endswith(('.wav', '.mp3')):
                    try:
                        # Handle MP3 conversion
                        working_path = path
                        if path.lower().endswith('.mp3'):
                            if not HAS_PYDUB:
                                QMessageBox.warning(None, "MP3 Support Missing", 
                                    "MP3 support requires pydub. Install with: pip install pydub\n"
                                    "For now, please use WAV files.")
                                return
                            # Convert MP3 to WAV
                            working_path = convert_mp3_to_wav(path)
                            s['temp_wav_path'] = working_path  # Store temp path for cleanup
                        
                        # Get audio duration and properties
                        total_duration = get_audio_duration(path)
                        
                        # For WAV processing, we need the WAV file
                        with wave.open(working_path, "rb") as wf:
                            frame_rate = wf.getframerate()
                            total_frames = wf.getnframes()
                        
                        # Set slider maximums to the actual audio duration
                        max_val = int(total_duration)
                        s['start_slider'].setMaximum(max_val)
                        s['end_slider'].setMaximum(max_val)
                        
                        # Set reasonable default values
                        s['start_slider'].setValue(0)
                        # Set end to either 10 seconds or the full duration, whichever is smaller
                        default_end = min(max_val, 10) if max_val > 0 else max_val
                        s['end_slider'].setValue(default_end)
                        
                        # Update tick intervals based on duration
                        if total_duration <= 30:
                            tick_interval = 5
                        elif total_duration <= 120:
                            tick_interval = 10
                        else:
                            tick_interval = max(int(total_duration / 10), 10)
                        
                        s['start_slider'].setTickInterval(tick_interval)
                        s['end_slider'].setTickInterval(tick_interval)
                        
                        file_type = "MP3" if path.lower().endswith('.mp3') else "WAV"
                        s['duration_label'].setText(f"Duration: {total_duration:.1f} sec ({file_type})")
                        
                        # Load audio into preview player and hide info label
                        if 'audio_preview' in s:
                            s['audio_preview'].load_audio(path)
                        if 'info_label' in s:
                            s['info_label'].setVisible(False)

                        if 'waveform_widget' in s:
                            s['waveform_widget'].clear()  # Clear any previous data
                            s['waveform_widget'].load_original(path)
                            
                        # Hide stego preview section until encoding happens
                        if 'stego_preview_section' in s:
                            s['stego_preview_section'].setVisible(False)
                        
                        # Update the segment UI to reflect the new values
                        self._update_segment_ui(media)
                        
                    except Exception as e:
                        s['duration_label'].setText(f"Duration: Unknown (error: {e})")
                        # Reset to default values on error
                        s['start_slider'].setMaximum(60)
                        s['end_slider'].setMaximum(60)
                        s['start_slider'].setValue(0)
                        s['end_slider'].setValue(10)
                        if 'temp_wav_path' in s:
                            try:
                                os.unlink(s['temp_wav_path'])
                            except:
                                pass
                            del s['temp_wav_path']
            
            # For decode mode, load the stego audio into the preview player
        if mode == "decode" and which == "stego":
            if path.lower().endswith(('.wav', '.mp3')):
                try:
                    total_duration = get_audio_duration(path)
                    max_val = int(total_duration)
                    
                    if 'start_slider' in s:
                        s['start_slider'].setMaximum(max_val)
                        s['end_slider'].setMaximum(max_val)
                        s['start_slider'].setValue(0)
                        s['end_slider'].setValue(min(max_val, 10))
                    
                    if 'duration_label' in s:
                        s['duration_label'].setText(f"Duration: {total_duration:.1f} sec")
                    
                    # Load audio into preview
                    if 'decode_audio_preview' in s:
                        s['decode_audio_preview'].load_audio(path)
                    if 'info_label' in s:
                        s['info_label'].setVisible(False)
                        
                except Exception as e:
                    if 'duration_label' in s:
                        s['duration_label'].setText(f"Duration: Unknown")

    # ==========================
    # Helpers: payload UI toggle & region
    # ==========================
    def _update_payload_ui(self, media: str):
        """Show Text field when Text mode; show File dropzone + button when File mode (IMAGE encode page)."""
        s = self.state[media]['encode']
        is_text = s.get('mode_text') and s['mode_text'].isChecked()
        if 'text_input' in s: s['text_input'].setVisible(is_text)
        if 'payload_label' in s: s['payload_label'].setVisible(not is_text)
        if 'payload_choose_btn' in s: s['payload_choose_btn'].setVisible(not is_text)

    def _apply_region_from_preview(self, media, x1, y1, x2, y2):
        s = self.state[media]['encode']
        for k, v in zip(('x1','y1','x2','y2'), (x1,y1,x2,y2)):
            if k in s: s[k].setValue(max(0, v))

    def _reset_region(self, media):
        s = self.state[media]['encode']
        for k in ('x1','y1','x2','y2'):
            if k in s: s[k].setValue(0)
    def _show_audio_capacity_popup(self, cover_path, is_file, payload, n_bits, key_text, start_sec, end_sec):
        """Show audio cover capacity vs payload details without saving."""
        with wave.open(cover_path, "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_channels = wf.getnchannels()
            total_frames = wf.getnframes()
            duration = total_frames / sample_rate
            total_bytes = total_frames * num_channels * sample_width

        # Restrict to selected segment
        if end_sec is None or end_sec > duration:
            end_sec = duration
        if start_sec is None:
            start_sec = 0
        seg_frames = int((end_sec - start_sec) * sample_rate)
        seg_bytes = seg_frames * num_channels * sample_width
        capacity_bits = seg_bytes * n_bits
        capacity_bytes = capacity_bits // 8

        # Payload size
        if is_file:
            payload_size = os.path.getsize(payload)
            payload_label = os.path.basename(payload)
        else:
            payload_size = len(payload.encode("utf-8"))
            payload_label = f"(Text) {payload[:40] + ('…' if len(payload) > 40 else '')}"

        sufficiency = "✅ Sufficient capacity" if payload_size <= capacity_bytes else "❌ Insufficient capacity"

        # Popup dialog
        box = QMessageBox(self)
        box.setWindowTitle("Audio Capacity Check")
        box.setIcon(QMessageBox.Information)
        box.setText("📊 Audio Cover Details")
        box.setInformativeText(
            f"<b>Cover:</b> {os.path.basename(cover_path)}<br>"
            f"<b>Duration:</b> {duration:.2f} sec<br>"
            f"<b>Channels:</b> {num_channels}, Sample Rate: {sample_rate} Hz<br>"
            f"<b>Segment:</b> {start_sec:.1f}–{end_sec:.1f}s<br>"
            f"<b>Capacity:</b> {capacity_bytes} bytes (with {n_bits} LSBs)<br><br>"
            f"<b>Payload:</b> {payload_label}<br>"
            f"<b>Payload Size:</b> {payload_size} bytes<br><br>"
            f"<b>Result:</b> {sufficiency}"
        )
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _update_segment_ui(self, media):
        """Update time labels, progress bar, and constraints based on sliders."""
        s = self.state[media]['encode']
        if 'start_slider' not in s or 'end_slider' not in s:
            return

        start_sec = s['start_slider'].value()
        end_sec = s['end_slider'].value()

        # Enforce: end >= start + 1
        if end_sec <= start_sec:
            s['end_slider'].setValue(start_sec + 1)
            end_sec = start_sec + 1

        # Update labels
        s['start_time_label'].setText(f"{start_sec}.0s")
        s['end_time_label'].setText(f"{end_sec}.0s")
        s['segment_info'].setText(f"Selected: {start_sec}.0—{end_sec}.0s")

        # Update progress bar
        total_range = s['end_slider'].maximum() - s['start_slider'].minimum()
        if total_range > 0:
            selected_range = end_sec - start_sec
            percent = int((selected_range / total_range) * 100)
            s['segment_progress'].setValue(percent)
        else:
            s['segment_progress'].setValue(0)

    def _update_decode_segment_ui(self, media):
        """Update decode segment UI labels and progress"""
        s = self.state[media]['decode']
        
        start_sec = s['start_slider'].value()
        end_sec = s['end_slider'].value()
        
        # Enforce constraint
        if end_sec <= start_sec:
            s['end_slider'].setValue(start_sec + 1)
            end_sec = start_sec + 1
        
        # Update labels
        s['start_time_label'].setText(f"{start_sec}.0s")
        s['end_time_label'].setText(f"{end_sec}.0s")
        s['segment_info'].setText(f"Selected: {start_sec}.0—{end_sec}.0s")
        s['segment_info'].setStyleSheet("color:#FF9800; font-weight:bold;")

        # Update progress bar
        total_range = s['end_slider'].maximum() - s['start_slider'].minimum()
        if total_range > 0:
            selected_range = end_sec - start_sec
            percent = int((selected_range / total_range) * 100)
            s['segment_progress'].setValue(percent)
        else:
            s['segment_progress'].setValue(0)

    def _preview_bytes_as_image(self, payload_bytes, target_preview_widget):
        """
        Try to preview payload bytes as an image in the given ImagePreviewSelector (or QLabel).
        Returns True if it looked like an image and was previewed, else False.
        """
        try:
            arr = np.frombuffer(payload_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return False
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            qimage = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimage)
            if isinstance(target_preview_widget, ImagePreviewSelector):
                target_preview_widget.set_image(pix)
            elif isinstance(target_preview_widget, QLabel):
                target_preview_widget.setPixmap(pix.scaled(
                    target_preview_widget.width(),
                    target_preview_widget.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            return True
        except Exception:
            return False


    # ==========================
    # Encode / Decode actions
    # ==========================
    def run_encoding(self, media):
        if media == "image":
            self._run_encoding_image()
        else:
            self._run_encoding_audio()

    def run_decoding(self, media):
        if media == "image":
            self._run_decoding_image()
        else:
            self._run_decoding_audio()

    # ---- Image (uses ENCODE_BMP / DECODE_BMP) ----
    def _run_encoding_image(self):
        s = self.state['image']['encode']
        cover = s.get('cover_path')
        payload_path = s.get('payload_path')
        n_bits = s['lsb'].value()
        key_text = s['key'].text().strip()
        if not key_text:
            QMessageBox.warning(self, "Missing key", "Key/passphrase is required for encoding.")
            return

        if not cover:
            QMessageBox.warning(self, "Error", "Select a cover image (BMP/PNG/JPG/GIF).")
            return

        # Payload mode
        if 'mode_text' in s and s['mode_text'].isChecked():
            is_file = False
            payload_arg = s['text_input'].toPlainText()
            if not payload_arg:
                QMessageBox.warning(self, "Error", "Enter the secret message (Text mode).")
                return
        else:
            is_file = True
            if not payload_path:
                QMessageBox.warning(self, "Error", "Select a payload file.")
                return
            payload_arg = payload_path

        # Region (0,0,0,0 => full image)
        region = None
        if all(k in s for k in ('x1','y1','x2','y2')):
            x1, y1, x2, y2 = s['x1'].value(), s['y1'].value(), s['x2'].value(), s['y2'].value()
            if (x1, y1, x2, y2) != (0, 0, 0, 0):
                region = (x1, y1, x2, y2)

        output = "stego_output.bmp"
        try:
            bmp_encode(
                image_name=cover,
                payload=payload_arg,
                output_name=output,
                n_bits=n_bits,
                is_file=is_file,
                region=region,
                key_text=key_text
            )
            self.state['image']['last_stego'] = output
            s['status'].setText(f"✅ Stego created: {output}")
            prev = s.get('preview')
            if isinstance(prev, ImagePreviewSelector) and os.path.exists(output):
                prev.set_image(output)
            # after setting preview/status
# Build payload label/size for summary
            if is_file:
                payload_size = os.path.getsize(payload_arg)
                payload_label = os.path.basename(payload_arg)
            else:
                payload_bytes = payload_arg.encode("utf-8")
                payload_size = len(payload_bytes)
                payload_label = f"(Text) {payload_arg[:40] + ('…' if len(payload_arg) > 40 else '')}"

            # Optional: show estimated region capacity (display only)
            extra = []
            try:
                img0 = cv2.imread(cover, cv2.IMREAD_COLOR)
                if img0 is not None:
                    h, w, _ = img0.shape
                    if region is None:
                        x1=y1=0; x2=w-1; y2=h-1
                    else:
                        x1,y1,x2,y2 = region
                    pixels = (x2-x1+1)*(y2-y1+1)
                    est_capacity_bytes = (pixels*3*n_bits)//8
                    extra.append(f"<b>Region:</b> ({x1},{y1})–({x2},{y2})")
                    extra.append(f"<b>Est. Capacity:</b> {_fmt_bytes(est_capacity_bytes)}")
            except Exception:
                pass

            self._show_encode_summary(
                media="image",
                cover=cover,
                output=output,
                payload_label=payload_label,
                payload_size=payload_size,
                n_bits=n_bits,
                key_text=key_text,
                extra_lines=extra
            )

        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))

    def _run_decoding_image(self):
        s = self.state['image']['decode']
        stego = s.get('stego_path')
        if not stego:
            QMessageBox.warning(self, "Error", "Select a stego image (BMP).")
            return
        key_text = s['key'].text().strip()
        if not key_text:
            QMessageBox.warning(self, "Missing key", "Key/passphrase is required for decoding.")
            return
        n_bits = s['lsb'].value()
        try:
            payload_bytes = bmp_decode(stego, key_text, n_bits)
            self.state['image']['last_extracted_bytes'] = payload_bytes

            # 1) Try to preview bytes as an image (PNG/JPG/GIF/BMP)
            preview_widget = s.get('decode_preview')
            previewed = self._preview_bytes_as_image(payload_bytes, preview_widget) if preview_widget else False

            # 2) Try to locate an auto-saved file named decoded_<...>
            auto_saved_msg = ""
            try:
                candidate_dirs = [os.getcwd(), os.path.dirname(stego)]
                found_path = None
                newest_mtime = -1
                for d in candidate_dirs:
                    if not d: continue
                    for name in os.listdir(d):
                        if name.lower().startswith("decoded_"):
                            full = os.path.join(d, name)
                            try:
                                m = os.path.getmtime(full)
                                if m > newest_mtime:
                                    newest_mtime, found_path = m, full
                            except Exception:
                                pass
                if found_path:
                    auto_saved_msg = f" Auto-saved: {os.path.basename(found_path)}"
            except Exception:
                pass

            # 3) Decide message: text vs file
            try:
                text = payload_bytes.decode("utf-8")
                self.state['image']['last_extracted_text'] = text
                s['status'].setText("✅ Payload extracted (TEXT). Use 'Save Extracted…' to save bytes." + auto_saved_msg)
            except UnicodeDecodeError:
                if previewed:
                    s['status'].setText("✅ Payload extracted (FILE – looks like an image)." + auto_saved_msg)
                else:
                    s['status'].setText("✅ Payload extracted (FILE)." + auto_saved_msg)

        except Exception as e:
            QMessageBox.critical(self, "Decoding Failed", str(e))

    # ---- Audio (key-shuffled, text payload demo) ----
    def _run_encoding_audio(self, preview_only=False):
        s = self.state['audio']['encode']
        cover = s.get('cover_path')
        n_bits = s['lsb'].value()
        key_text = s['key'].text().strip()

        if not cover or not key_text:
            QMessageBox.warning(self, "Error", "Select a cover WAV and enter a key.")
            return

        # Payload mode
        if s.get('mode_text') and s['mode_text'].isChecked():
            payload = s['text_input'].toPlainText().strip()
            if not payload:
                QMessageBox.warning(self, "Error", "Enter the secret message (Text mode).")
                return
            is_file = False
        else:
            payload = s.get('payload_path')
            if not payload:
                QMessageBox.warning(self, "Error", "Select a payload file.")
                return
            is_file = True

        start_sec = s['start_slider'].value() if 'start_slider' in s else 0
        end_sec = s['end_slider'].value() if 'end_slider' in s else None

        if preview_only:
            # Show capacity popup, no file written
            self._show_audio_capacity_popup(cover, is_file, payload, n_bits, key_text, start_sec, end_sec)
            return

        # Else, actually encode + save
        try:
            # === Save As dialog (instead of hardcoding path) ===
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Encoded Audio",
                "stego_output.wav",
                "WAV files (*.wav)"
            )
            if not out_path:
                return  # user cancelled

            encode_audio_with_key(cover, is_file, payload, out_path, key_text, n_bits, start_sec, end_sec)
            QMessageBox.information(self, "Success", f"Encoded audio saved to {out_path}")
            self.state['audio']['last_stego'] = out_path

            if 'waveform_widget' in s:
                s['waveform_widget'].load_stego(out_path)
            if 'stego_preview' in s:
                s['stego_preview'].load_audio(out_path)
            if 'stego_preview_section' in s:
                s['stego_preview_section'].setVisible(True)

        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))


    def _run_decoding_audio(self):
        s = self.state['audio']['decode']
        stego = s.get('stego_path')
        n_bits = s['lsb'].value()
        key_text = s['key'].text().strip()
        
        if not stego or not key_text:
            QMessageBox.warning(self, "Error", "Select a stego WAV and enter a key.")
            return
        
        try:
            start_sec = s['start_slider'].value()
            end_sec = s['end_slider'].value()
            
            result = decode_audio_with_key(stego, key_text, start_sec, end_sec, n_bits)
            
            # Check if result is Base64-encoded file or plain text
            is_base64 = False
            try:
                # Try to decode as Base64 to check if it's a file
                decoded_bytes = base64.b64decode(result)
                is_base64 = True
            except:
                is_base64 = False
            
            if is_base64:
                # It's a file - try to preview it
                self.state['audio']['last_extracted_bytes'] = decoded_bytes
                
                # Try to decode as text first
                try:
                    text_content = decoded_bytes.decode('utf-8')
                    s['preview'].setPlainText(text_content)
                    s['status'].setText(f"✅ File extracted ({len(decoded_bytes)} bytes) - displaying as text.")
                except UnicodeDecodeError:
                    # Binary file - show hex preview
                    hex_preview = decoded_bytes[:500].hex()
                    preview_text = f"Binary file extracted ({len(decoded_bytes)} bytes)\n\n"
                    preview_text += f"Hex preview (first 500 bytes):\n{hex_preview}"
                    s['preview'].setPlainText(preview_text)
                    s['status'].setText(f"✅ Binary file extracted ({len(decoded_bytes)} bytes). Use 'Save Extracted...'")
            else:
                # Plain text payload
                s['preview'].setPlainText(result if result else "(no payload found)")
                s['status'].setText("✅ Text payload extracted.")
            
            self.state['audio']['last_extracted_text'] = result
            
        except Exception as e:
            QMessageBox.critical(self, "Decoding Failed", str(e))

    def _save_last_stego(self, media):
        path = self.state[media].get('last_stego')
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Please calculate size before encoding and saving.")
            return
        
        ext = ".wav" if media == "audio" else ".bmp"
        fname, _ = QFileDialog.getSaveFileName(self, "Save Stego As", "", f"*{ext}")
        if fname:
            if not fname.lower().endswith(ext):
                fname += ext
            try:
                os.replace(path, fname)
                self.state[media]['last_stego'] = fname
                QMessageBox.information(self, "Saved", f"Stego saved: {fname}")

                # === NEW: show popup for image ===
                if media == "image":
                    s = self.state['image']['encode']
                    orig = s.get('cover_path')
                    n_bits = s['lsb'].value()

                    # payload size: if text mode, use string length, else file size
                    if s.get('mode_text') and s['mode_text'].isChecked():
                        payload_size = len(s['text_input'].toPlainText().encode("utf-8"))
                    else:
                        payload_size = os.path.getsize(s.get('payload_path')) if s.get('payload_path') else 0

                    self._show_image_comparison_popup(orig, fname, payload_size, n_bits)


            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))



    def _save_extracted(self, media):
        if media == "image":
            data = self.state['image'].get('last_extracted_bytes')
            if not data:
                QMessageBox.warning(self, "Error", "Nothing extracted yet.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Save Extracted Payload", "", "All Files (*)")
            if fname:
                try:
                    with open(fname, "wb") as f:
                        f.write(data)
                    QMessageBox.information(self, "Saved", f"Saved: {fname}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Failed", str(e))
        else:  # audio
            # Check if we have binary data (decoded file)
            binary_data = self.state['audio'].get('last_extracted_bytes')
            text_data = self.state['audio'].get('last_extracted_text')
            
            if binary_data:
                # Save binary file
                fname, _ = QFileDialog.getSaveFileName(self, "Save Extracted File", "", "All Files (*)")
                if fname:
                    try:
                        with open(fname, "wb") as f:
                            f.write(binary_data)
                        QMessageBox.information(self, "Saved", f"Saved: {fname}")
                    except Exception as e:
                        QMessageBox.critical(self, "Save Failed", str(e))
            elif text_data:
                # Save text
                fname, _ = QFileDialog.getSaveFileName(self, "Save Extracted Text", "", "Text Files (*.txt);;All Files (*)")
                if fname:
                    try:
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write(text_data)
                        QMessageBox.information(self, "Saved", f"Saved: {fname}")
                    except Exception as e:
                        QMessageBox.critical(self, "Save Failed", str(e))
            else:
                QMessageBox.warning(self, "Error", "Nothing extracted yet.")

# ==========================
# Theme
# ==========================
def apply_dark_theme(app):
    app.setStyle("Fusion")
    dark = QPalette()
    dark.setColor(QPalette.Window, QColor(35, 35, 35))
    dark.setColor(QPalette.WindowText, Qt.white)
    dark.setColor(QPalette.Base, QColor(25, 25, 25))
    dark.setColor(QPalette.Text, Qt.white)
    dark.setColor(QPalette.Button, QColor(45, 45, 45))
    dark.setColor(QPalette.ButtonText, Qt.white)
    dark.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    dark.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = StegoMainWindow()
    win.show()
    sys.exit(app.exec())