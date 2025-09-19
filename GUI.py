import sys, os, base64, wave, importlib.util, tempfile
import cv2, numpy as np
import pygame, tempfile, time
pygame.mixer.init()
from PySide6.QtCore import Qt, QRect, QSize, QPoint, Signal, QUrl, QTimer
from PySide6.QtGui import QColor, QPalette, QPainter, QBrush, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QFormLayout, QTabWidget, QSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QStackedWidget, QSizePolicy, QRadioButton,
    QRubberBand, QSplitter, QSlider, QProgressBar
)

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
# Audio backend (key-shuffled LSB)
# ==========================
STOP_MARKER = b"====="

def generate_coords(length, key: int):
    coords = list(range(length))
    rng = np.random.RandomState(int(key))
    rng.shuffle(coords)
    return coords

def encode_audio(wav_path, secret_data_text, output_path, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for encoding (audio).")
    with wave.open(wav_path, "rb") as audio:
        params = audio.getparams()
        frames = bytearray(audio.readframes(audio.getnframes()))
    secret_bytes = secret_data_text.encode("utf-8") + STOP_MARKER
    data_bits = ''.join(format(b, '08b') for b in secret_bytes)
    max_bits = len(frames) * n_bits
    if len(data_bits) > max_bits:
        raise ValueError(f"Payload too large! Capacity: {max_bits} bits, Required: {len(data_bits)} bits.")
    coords = generate_coords(len(frames), key)
    mask = ~((1 << n_bits) - 1) & 0xFF
    bit_index = 0
    for idx in coords:
        if bit_index >= len(data_bits):
            break
        chunk = data_bits[bit_index:bit_index+n_bits]
        bit_index += n_bits
        val = int(chunk.ljust(n_bits, '0'), 2)
        frames[idx] = (frames[idx] & mask) | val
    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)

def decode_audio(wav_path, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for decoding (audio).")
    with wave.open(wav_path, "rb") as audio:
        frames = bytearray(audio.readframes(audio.getnframes()))
    coords = generate_coords(len(frames), key)
    bits = ""
    out = bytearray()
    for idx in coords:
        bits += format(frames[idx] & ((1 << n_bits) - 1), f'0{n_bits}b')
        while len(bits) >= 8:
            byte = int(bits[:8], 2)
            bits = bits[8:]
            out.append(byte)
            if out.endswith(STOP_MARKER):
                return out[:-len(STOP_MARKER)].decode("utf-8", errors="replace")
    raise ValueError("Stop marker not found in audio payload.")

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
            'audio': {'encode': {}, 'decode': {}, 'last_stego': None, 'last_extracted_text': None},
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

        # Key
        s['key'] = QLineEdit(); s['key'].setEchoMode(QLineEdit.Password); s['key'].setPlaceholderText("(Required) Integer key for scrambling")
        style_line_edit(s['key'])
        form.addRow("Key:", s['key'])

        settings.setLayout(form)

        # --- AUDIO SEGMENT SELECTOR (only for audio) ---
        if media == "audio":
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
        act = QHBoxLayout()
        enc_btn = QPushButton("🚀 Encode"); enc_btn.clicked.connect(lambda checked=False, m=media: self.run_encoding(m))
        save_btn = QPushButton("💾 Save Stego As…"); save_btn.clicked.connect(lambda checked=False, m=media: self._save_last_stego(m))
        act.addWidget(enc_btn); act.addWidget(save_btn); act.addStretch(1)

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

        s['key'] = QLineEdit(); s['key'].setEchoMode(QLineEdit.Password); s['key'].setPlaceholderText("(Optional) Not used for image backend")
        style_line_edit(s['key'])
        form.addRow("Key:", s['key'])
        settings.setLayout(form)

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
            s['preview'].setPlaceholderText("Extracted payload (BASE64 text preview)...")
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
                    if 'decode_audio_preview' in s:
                        s['decode_audio_preview'].load_audio(path)
                    if 'info_label' in s:
                        s['info_label'].setVisible(False)

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

    def _update_segment_ui(self, media):
        """Update time labels, progress bar, and constraints based on sliders."""
        s = self.state[media]['encode']
        if 'start_slider' not in s or 'end_slider' not in s:
            return

        start_sec = s['start_slider'].value()
        end_sec = s['end_slider'].value()

        # Enforce: end >= start + 1 (at least 1 second)
        if end_sec <= start_sec:
            s['end_slider'].setValue(start_sec + 1)
            end_sec = start_sec + 1

        # Update labels
        s['start_time_label'].setText(f"{start_sec}.0s")
        s['end_time_label'].setText(f"{end_sec}.0s")
        s['segment_info'].setText(f"Selected: {start_sec}.0–{end_sec}.0s")

        # Update progress bar (as % of total range)
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

        if not cover:
            QMessageBox.warning(self, "Error", "Select a cover image (BMP/PNG/JPG/GIF).")
            return

        # Payload mode
        if 'mode_text' in s and s['mode_text'].isChecked():
            is_file = False
            payload_arg = s['text_input'].text()
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
                region=region
            )
            self.state['image']['last_stego'] = output
            s['status'].setText(f"✅ Stego created: {output}")
            prev = s.get('preview')
            if isinstance(prev, ImagePreviewSelector) and os.path.exists(output):
                prev.set_image(output)
        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))

    def _run_decoding_image(self):
        s = self.state['image']['decode']
        stego = s.get('stego_path')
        if not stego:
            QMessageBox.warning(self, "Error", "Select a stego image (BMP).")
            return
        try:
            payload_bytes = bmp_decode(stego)  # auto-detects region/lsb & may save decoded_<filename>
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
    def _run_encoding_audio(self):
        s = self.state['audio']['encode']
        cover = s.get('cover_path')
        payload_path = s.get('payload_path')
        n_bits = s['lsb'].value()
        key_text = s['key'].text().strip()

        if not cover:
            QMessageBox.warning(self, "Error", "Select an audio cover (WAV or MP3).")
            return
        if not payload_path:
            QMessageBox.warning(self, "Error", "Select a payload file.")
            return
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Key must be an integer.")
            return

        # Handle MP3 conversion if needed
        working_cover = cover
        temp_files = []
        
        try:
            if cover.lower().endswith('.mp3'):
                if not HAS_PYDUB:
                    QMessageBox.warning(self, "Error", "MP3 support requires pydub. Please install it or use WAV files.")
                    return
                working_cover = convert_mp3_to_wav(cover)
                temp_files.append(working_cover)

            # Load WAV to get sample rate and duration
            with wave.open(working_cover, "rb") as wf:
                frame_rate = wf.getframerate()
                total_frames = wf.getnframes()
                total_duration = total_frames / frame_rate

            # Get selected segment from sliders
            start_sec = s['start_slider'].value()
            end_sec = s['end_slider'].value()

            if end_sec > total_duration:
                QMessageBox.warning(self, "Warning", f"End time ({end_sec}s) exceeds audio length ({total_duration:.1f}s). Truncating.")
                end_sec = int(total_duration)
                s['end_slider'].setValue(end_sec)

            if start_sec >= end_sec:
                QMessageBox.warning(self, "Error", "Start time must be before end time.")
                return

            # Convert to frame indices
            start_frame = int(start_sec * frame_rate)
            end_frame = int(end_sec * frame_rate)

            # Ensure we don't exceed audio length
            end_frame = min(end_frame, total_frames)

            # Calculate available bits in segment
            segment_frames = end_frame - start_frame
            required_bytes = len(load_binary_as_text(payload_path).encode('utf-8'))
            required_bits = required_bytes * 8
            available_bits = segment_frames * n_bits

            if required_bits > available_bits:
                QMessageBox.critical(self, "Capacity Exceeded",
                    f"Payload requires {required_bits} bits.\n"
                    f"Selected segment ({start_sec}-{end_sec}s) can only hold {available_bits} bits with {n_bits} LSBs.\n"
                    f"Try increasing LSBs or extending the segment.")
                return

            output = "stego_output.wav"
            payload_text = load_binary_as_text(payload_path)

            # Perform encoding
            encode_audio(working_cover, payload_text, output, n_bits, int(key_text))
            self.state['audio']['last_stego'] = output
            
            # Update status 
            s['status'].setText(f"✅ Stego created: {output}\nSegment used: {start_sec}–{end_sec}s ({segment_frames} frames)")
            
            # Show stego preview section and load the stego audio
            if 'stego_preview_section' in s:
                s['stego_preview_section'].setVisible(True)
            if 'stego_preview' in s:
                s['stego_preview'].load_audio(output)
            if 'info_label' in s:
                s['info_label'].setVisible(False)

        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _run_decoding_audio(self):
        s = self.state['audio']['decode']
        stego = s.get('stego_path')
        n_bits = s['lsb'].value()
        key_text = s['key'].text().strip() if 'key' in s else ""
        
        if not stego:
            QMessageBox.warning(self, "Error", "Select a stego audio file (WAV or MP3).")
            return
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Key must be an integer.")
            return
            
        # Handle MP3 conversion if needed
        working_stego = stego
        temp_files = []
        
        try:
            if stego.lower().endswith('.mp3'):
                if not HAS_PYDUB:
                    QMessageBox.warning(self, "Error", "MP3 support requires pydub. Please install it or use WAV files.")
                    return
                working_stego = convert_mp3_to_wav(stego)
                temp_files.append(working_stego)
            
            # Decode the audio
            text = decode_audio(working_stego, n_bits, int(key_text))
            
            # Update preview
            if 'preview' not in s:
                s['preview'] = QTextEdit()
                s['preview'].setReadOnly(True)
                style_text_edit(s['preview'])
            s['preview'].setPlainText(text)
            
            # Hide info label
            if 'info_label' in s:
                s['info_label'].setVisible(False)
            
            s['status'] = s.get('status') or QLabel()
            s['status'].setText("✅ Payload extracted (BASE64 TEXT). Use 'Save Extracted...' to write binary.")
            self.state['audio']['last_extracted_text'] = text
            
        except Exception as e:
            QMessageBox.critical(self, "Decoding Failed", str(e))
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    # Save handlers
    def _save_last_stego(self, media):
        path = self.state[media].get('last_stego')
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "No stego file generated yet.")
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
        else:
            text = self.state['audio'].get('last_extracted_text')
            if not text:
                QMessageBox.warning(self, "Error", "Nothing extracted yet.")
                return
            choice = QMessageBox.question(self, "Save As",
                                          "Save raw BASE64 text? (Yes)\n\n"
                                          "Or decode BASE64 back to binary file? (No)",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if choice == QMessageBox.Yes:
                fname, _ = QFileDialog.getSaveFileName(self, "Save Extracted (Base64 Text)", "", "Text (*.txt);;All Files (*)")
                if fname:
                    try:
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write(text)
                        QMessageBox.information(self, "Saved", f"Saved: {fname}")
                    except Exception as e:
                        QMessageBox.critical(self, "Save Failed", str(e))
            else:
                fname, _ = QFileDialog.getSaveFileName(self, "Save Extracted (Decoded Binary)", "", "All Files (*)")
                if fname:
                    try:
                        save_text_as_binary(text, fname)
                        QMessageBox.information(self, "Saved", f"Saved: {fname}")
                    except Exception as e:
                        QMessageBox.critical(self, "Save Failed", str(e))

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