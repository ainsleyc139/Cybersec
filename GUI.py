import sys, os, base64
import cv2
import numpy as np
import wave
from PySide6.QtCore import Qt, QMimeData, Signal
from PySide6.QtGui import QColor, QPalette, QPixmap, QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QLineEdit, QFormLayout,
    QTabWidget, QFrame, QSpinBox, QTextEdit, QFileDialog, QMessageBox,
    QScrollArea, QSizePolicy
)

STOP_MARKER = b"====="  # Use bytes for consistency

# ---------------- Utility Functions ----------------
def to_bin(data):
    if isinstance(data, str):
        return ''.join([format(ord(i), "08b") for i in data])
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return [format(int(i), "08b") for i in data]
    elif isinstance(data, (int, np.integer)):
        return format(int(data), "08b")
    else:
        raise TypeError(f"Type not supported: {type(data)}")

def generate_coords(length, key: int):
    """Generate a reproducible sequence of indices using key as seed"""
    coords = list(range(length))
    rng = np.random.RandomState(int(key))
    rng.shuffle(coords)
    return coords

# ---------------- IMAGE ENCODING/DECODING ----------------
def encode_image(image_path, secret_data, output_path, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for encoding.")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    if isinstance(secret_data, str):
        secret_data = secret_data.encode('utf-8')
    secret_data += STOP_MARKER
    binary_secret_data = to_bin(secret_data)
    data_len = len(binary_secret_data)

    h, w, _ = image.shape
    total_pixels = h * w * 3
    coords = generate_coords(total_pixels, key)

    if data_len > len(coords) * n_bits:
        raise ValueError(f"Payload too large! Capacity: {len(coords) * n_bits} bits, Required: {data_len} bits.")

    mask = ~((1 << n_bits) - 1) & 0xFF
    data_index = 0
    for idx in coords:
        if data_index >= data_len:
            break
        y = idx // (w * 3)
        x = (idx // 3) % w
        c = idx % 3
        bits = int(binary_secret_data[data_index:data_index+n_bits].ljust(n_bits, '0'), 2)
        image[y, x, c] = np.uint8((int(image[y, x, c]) & mask) | bits)
        data_index += n_bits

    cv2.imwrite(output_path, image)

def decode_image(image_path, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for decoding.")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    h, w, _ = image.shape
    total_pixels = h * w * 3
    coords = generate_coords(total_pixels, key)

    binary_data = ""
    decoded_bytes = bytearray()

    for idx in coords:
        y = idx // (w * 3)
        x = (idx // 3) % w
        c = idx % 3
        binary_data += to_bin(image[y, x, c])[-n_bits:]

        while len(binary_data) >= 8:
            byte = binary_data[:8]
            binary_data = binary_data[8:]
            decoded_bytes.append(int(byte, 2))

            if decoded_bytes.endswith(STOP_MARKER):
                return decoded_bytes[:-len(STOP_MARKER)]

    raise ValueError("Stop marker not found.")

def compare_images(orig_path, stego_path):
    orig = cv2.imread(orig_path)
    stego = cv2.imread(stego_path)
    if orig is None or stego is None:
        return "Could not open one of the files."
    if orig.shape != stego.shape:
        return "Images have different dimensions."
    diff_mask = orig != stego
    diff_count = np.sum(diff_mask)
    diff_map = np.zeros_like(orig)
    diff_map[diff_mask] = [0, 0, 255]
    diff_map_path = "diff_map.bmp"
    cv2.imwrite(diff_map_path, diff_map)
    return f"Number of pixel values changed: {diff_count}\nDifference map saved as {diff_map_path}"

# ---------------- AUDIO ENCODING/DECODING ----------------
def encode_audio(wav_path, secret_data, output_path, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for encoding.")
    
    with wave.open(wav_path, "rb") as audio:
        params = audio.getparams()
        frames = bytearray(audio.readframes(audio.getnframes()))
    
    if isinstance(secret_data, str):
        secret_data = secret_data.encode('utf-8')
    secret_data += STOP_MARKER
    data_len_bits = len(secret_data) * 8
    max_bits = len(frames) * n_bits

    if data_len_bits > max_bits:
        raise ValueError(f"Payload too large! Capacity: {max_bits} bits, Required: {data_len_bits} bits.")

    binary_secret_data = ''.join(format(byte, '08b') for byte in secret_data)
    coords = generate_coords(len(frames), key)

    for idx in coords:
        if not binary_secret_data:
            break
        bits_to_set = binary_secret_data[:n_bits]
        binary_secret_data = binary_secret_data[n_bits:]
        bits_value = int(bits_to_set.ljust(n_bits, '0'), 2)
        mask = ~((1 << n_bits) - 1) & 0xFF
        frames[idx] = (frames[idx] & mask) | bits_value

    with wave.open(output_path, "wb") as output:
        output.setparams(params)
        output.writeframes(frames)

def decode_audio(wav_path, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for decoding.")
    
    with wave.open(wav_path, "rb") as audio:
        frames = bytearray(audio.readframes(audio.getnframes()))

    binary_data = ""
    decoded_bytes = bytearray()

    coords = generate_coords(len(frames), key)

    for idx in coords:
        byte_val = frames[idx]
        binary_data += format(byte_val & ((1 << n_bits) - 1), f'0{n_bits}b')

        while len(binary_data) >= 8:
            byte_str = binary_data[:8]
            binary_data = binary_data[8:]
            decoded_bytes.append(int(byte_str, 2))

            if decoded_bytes.endswith(STOP_MARKER):
                return decoded_bytes[:-len(STOP_MARKER)]

    raise ValueError("Stop marker not found.")

# ---------------- Payload Helper ----------------
def load_binary_as_text(file_path):
    with open(file_path, "rb") as f:
        binary_data = f.read()
    return base64.b64encode(binary_data).decode('utf-8')

def save_text_as_binary(text, output_path):
    binary_data = base64.b64decode(text.encode('utf-8'))
    with open(output_path, "wb") as f:
        f.write(binary_data)

# ---------------- DRAGGABLE LABEL ----------------
class DraggableLabel(QLabel):
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            border: 2px dashed #aaa;
            border-radius: 8px;
            background-color: #1e1e1e;
            color: #ccc;
            font-size: 14px;
            padding: 10px;
            min-height: 150px;
        """)
        self.setText("Drag & Drop File Here\nor click to browse")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.fileDropped.emit(path)

# ---------------- GUI ----------------
class StegoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔒 LSB Steganography & Steganalysis Tool (AY25-ACW1)")
        self.resize(1200, 850)  # Start at reasonable size

        self.cover_path = ""
        self.payload_path = ""
        self.stego_path = ""
        self.extracted_data = ""
        self.cover_type = ""  # "image", "audio"
        self.payload_type = ""  # "text", "binary"

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        tab_widget.setStyleSheet("""
            QTabBar::tab { min-width: 180px; padding: 12px; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #333; }
        """)

        self.encode_tab = QWidget()
        self.setup_encode_tab()
        tab_widget.addTab(self.encode_tab, "📝 Encode")

        self.decode_tab = QWidget()
        self.setup_decode_tab()
        tab_widget.addTab(self.decode_tab, "📂 Decode")

        main_layout.addWidget(tab_widget)
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

    def setup_encode_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Cover Object Section
        cover_group = QGroupBox("📁 Cover Object (Image or Audio)")
        cover_layout = QHBoxLayout()
        cover_layout.setSpacing(20)
        cover_layout.setContentsMargins(15, 15, 15, 15)

        self.cover_label = DraggableLabel()
        self.cover_label.setMinimumSize(300, 200)
        self.cover_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cover_layout.addWidget(self.cover_label, 1)

        right_controls = QVBoxLayout()
        right_controls.setSpacing(10)
        self.cover_button = QPushButton("📁 Browse Cover File")
        self.cover_button.clicked.connect(self.load_cover_file)
        self.play_cover_button = QPushButton("▶️ Play Cover Audio")
        self.play_cover_button.clicked.connect(self.play_cover_audio)
        self.play_cover_button.setEnabled(False)
        right_controls.addWidget(self.cover_button)
        right_controls.addWidget(self.play_cover_button)
        right_controls.addStretch()
        cover_layout.addLayout(right_controls, 0)

        cover_group.setLayout(cover_layout)

        # Payload Section
        payload_group = QGroupBox("📥 Payload (Text, .exe, .pdf, etc.)")
        payload_layout = QHBoxLayout()
        payload_layout.setSpacing(20)
        payload_layout.setContentsMargins(15, 15, 15, 15)

        self.payload_label = DraggableLabel()
        self.payload_label.setMinimumSize(300, 200)
        self.payload_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        payload_layout.addWidget(self.payload_label, 1)

        right_payload = QVBoxLayout()
        right_payload.setSpacing(10)
        self.payload_button = QPushButton("📁 Browse Payload File")
        self.payload_button.clicked.connect(self.load_payload_file)
        right_payload.addWidget(self.payload_button)
        right_payload.addStretch()
        payload_layout.addLayout(right_payload, 0)

        payload_group.setLayout(payload_layout)

        # Settings Group
        settings_group = QGroupBox("⚙️ Stego Settings")
        form = QFormLayout()
        form.setSpacing(12)
        form.setContentsMargins(15, 15, 15, 15)

        self.lsb_spinbox = QSpinBox()
        self.lsb_spinbox.setRange(1, 8)
        self.lsb_spinbox.setValue(1)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter integer key (required)")
        self.key_input.setEchoMode(QLineEdit.Password)

        self.capacity_label = QLabel("Capacity: — bits")
        self.capacity_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.capacity_label.setStyleSheet("color: #4CAF50;")

        form.addRow("Number of LSBs:", self.lsb_spinbox)
        form.addRow("Encryption Key:", self.key_input)
        form.addRow("Capacity:", self.capacity_label)

        settings_group.setLayout(form)

        # Actions
        button_layout = QHBoxLayout()
        self.encode_button = QPushButton("🚀 Embed Payload")
        self.save_stego_button = QPushButton("💾 Save Stego File")
        self.encode_button.clicked.connect(self.run_encoding)
        self.save_stego_button.clicked.connect(self.save_stego_file)
        self.save_stego_button.setEnabled(False)
        button_layout.addWidget(self.encode_button)
        button_layout.addWidget(self.save_stego_button)
        button_layout.addStretch()

        # Visualization
        self.visualization_label = QLabel("📊 Difference Map / Preview")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        self.visualization_label.setMinimumSize(400, 200)
        self.visualization_label.setStyleSheet("""
            border: 1px dashed gray;
            background-color: #1a1a1a;
            color: #999;
            font-style: italic;
            padding: 10px;
        """)
        self.visualization_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(cover_group)
        layout.addWidget(payload_group)
        layout.addWidget(settings_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.visualization_label)
        layout.addStretch()

        self.encode_tab.setLayout(layout)

    def setup_decode_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Stego Object
        stego_group = QGroupBox("🔍 Stego Object")
        stego_layout = QHBoxLayout()
        stego_layout.setSpacing(20)
        stego_layout.setContentsMargins(15, 15, 15, 15)

        self.stego_label = DraggableLabel()
        self.stego_label.setMinimumSize(300, 200)
        self.stego_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stego_layout.addWidget(self.stego_label, 1)

        right_stego = QVBoxLayout()
        right_stego.setSpacing(10)
        self.stego_button = QPushButton("📁 Browse Stego File")
        self.stego_button.clicked.connect(self.load_stego_file)
        self.play_stego_button = QPushButton("▶️ Play Stego Audio")
        self.play_stego_button.clicked.connect(self.play_stego_audio)
        self.play_stego_button.setEnabled(False)
        right_stego.addWidget(self.stego_button)
        right_stego.addWidget(self.play_stego_button)
        right_stego.addStretch()
        stego_layout.addLayout(right_stego, 0)

        stego_group.setLayout(stego_layout)

        # Settings
        settings_group = QGroupBox("⚙️ Decode Settings")
        form = QFormLayout()
        form.setSpacing(12)
        form.setContentsMargins(15, 15, 15, 15)

        self.lsb_spinbox_decode = QSpinBox()
        self.lsb_spinbox_decode.setRange(1, 8)
        self.lsb_spinbox_decode.setValue(1)

        self.key_input_decode = QLineEdit()
        self.key_input_decode.setPlaceholderText("Enter same key used during encoding")
        self.key_input_decode.setEchoMode(QLineEdit.Password)

        form.addRow("Number of LSBs:", self.lsb_spinbox_decode)
        form.addRow("Decryption Key:", self.key_input_decode)
        settings_group.setLayout(form)

        # Actions
        button_layout = QHBoxLayout()
        self.decode_button = QPushButton("🔍 Extract Payload")
        self.save_payload_button = QPushButton("💾 Save Extracted File")
        self.play_button = QPushButton("▶️ Play Extracted Payload")
        self.decode_button.clicked.connect(self.run_decoding)
        self.save_payload_button.clicked.connect(self.save_decoded_file)
        self.play_button.clicked.connect(self.play_extracted_payload)
        self.save_payload_button.setEnabled(False)
        self.play_button.setEnabled(False)
        button_layout.addWidget(self.decode_button)
        button_layout.addWidget(self.save_payload_button)
        button_layout.addWidget(self.play_button)
        button_layout.addStretch()

        # Preview
        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setPlaceholderText("Extracted payload content preview...")
        self.preview_box.setStyleSheet("""
            border: 1px solid gray;
            background: #222;
            color: #ddd;
            font-family: monospace;
            padding: 10px;
        """)
        self.preview_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(stego_group)
        layout.addWidget(settings_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.preview_box)
        layout.addStretch()

        self.decode_tab.setLayout(layout)

    def on_cover_dropped(self, path):
        if os.path.isfile(path):
            self.cover_path = path
            self.update_cover_display()

    def on_payload_dropped(self, path):
        if os.path.isfile(path):
            self.payload_path = path
            self.update_payload_display()

    def on_stego_dropped(self, path):
        if os.path.isfile(path):
            self.stego_path = path
            self.update_stego_display()

    def update_cover_display(self):
        ext = self.cover_path.lower()
        if ext.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            self.cover_type = "image"
            pixmap = QPixmap(self.cover_path).scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cover_label.setPixmap(pixmap)
            self.cover_label.setText("")
            self.play_cover_button.setEnabled(False)
        elif ext.endswith('.wav'):
            self.cover_type = "audio"
            self.cover_label.setText(f"🔊 WAV Audio Selected:\n{os.path.basename(self.cover_path)}")
            self.play_cover_button.setEnabled(True)
        else:
            self.cover_label.setText("⚠ Unsupported file type")
            self.cover_path = ""
            self.play_cover_button.setEnabled(False)
            return
        self.calculate_capacity()

    def update_payload_display(self):
        ext = self.payload_path.lower()
        if ext.endswith('.txt'):
            self.payload_type = "text"
            with open(self.payload_path, 'r', encoding='utf-8') as f:
                content = f.read(100)
                self.payload_label.setText(f"📄 Text Payload:\n{content}{'...' if len(content) == 100 else ''}")
        else:
            self.payload_type = "binary"
            size = os.path.getsize(self.payload_path)
            self.payload_label.setText(f"📦 Binary Payload:\n{os.path.basename(self.payload_path)}\n({size:,} bytes)")

    def update_stego_display(self):
        ext = self.stego_path.lower()
        if ext.endswith(('.bmp', '.png')):
            self.cover_type = "image"
            pixmap = QPixmap(self.stego_path).scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.stego_label.setPixmap(pixmap)
            self.stego_label.setText("")
            self.play_stego_button.setEnabled(False)
        elif ext.endswith('.wav'):
            self.cover_type = "audio"
            self.stego_label.setText(f"🔊 Stego WAV:\n{os.path.basename(self.stego_path)}")
            self.play_stego_button.setEnabled(True)
        else:
            self.stego_label.setText("⚠ Unsupported file")
            self.stego_path = ""
            self.play_stego_button.setEnabled(False)
            return

    def calculate_capacity(self):
        if not self.cover_path:
            self.capacity_label.setText("Capacity: — bits")
            return

        try:
            if self.cover_type == "image":
                img = cv2.imread(self.cover_path)
                if img is None:
                    raise ValueError("Invalid image")
                h, w, _ = img.shape
                total_bits = h * w * 3 * 8
                n_bits = self.lsb_spinbox.value()
                capacity = total_bits * n_bits // 8
            elif self.cover_type == "audio":
                with wave.open(self.cover_path, "rb") as f:
                    frames = f.readframes(f.getnframes())
                n_bits = self.lsb_spinbox.value()
                capacity = len(frames) * n_bits // 8

            self.capacity_label.setText(f"Capacity: {capacity:,} bytes ({capacity * 8} bits)")
        except Exception as e:
            self.capacity_label.setText(f"Capacity: Error ({str(e)})")

    def play_cover_audio(self):
        if self.cover_path and self.cover_type == "audio":
            os.startfile(self.cover_path)
            QMessageBox.information(self, "Playing", f"Opening {os.path.basename(self.cover_path)} in default player...")

    def play_stego_audio(self):
        if self.stego_path and self.cover_type == "audio":
            os.startfile(self.stego_path)
            QMessageBox.information(self, "Playing", f"Opening {os.path.basename(self.stego_path)} in default player...")

    def play_extracted_payload(self):
        if not self.extracted_data:
            QMessageBox.warning(self, "Error", "No payload extracted yet!")
            return
        if self.payload_type == "text":
            QMessageBox.information(self, "Extracted Text", self.extracted_data[:1000] + ("..." if len(self.extracted_data) > 1000 else ""))
        else:
            temp_path = "temp_extracted"
            save_text_as_binary(self.extracted_data, temp_path)
            if temp_path.endswith(".wav"):
                os.startfile(temp_path)
                QMessageBox.information(self, "Playing", f"Playing extracted audio: {os.path.basename(temp_path)}")
            else:
                os.startfile(temp_path)

    def load_cover_file(self):
        filters = "Image Files (*.bmp *.png *.jpg *.jpeg);;WAV Audio Files (*.wav);;All Files (*)"
        fname, _ = QFileDialog.getOpenFileName(self, "Select Cover File", "", filters)
        if fname:
            self.cover_path = fname
            self.update_cover_display()

    def load_payload_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Payload File", "", "All Files (*)")
        if fname:
            self.payload_path = fname
            self.update_payload_display()

    def load_stego_file(self):
        filters = "Image Files (*.bmp *.png);;WAV Audio Files (*.wav);;All Files (*)"
        fname, _ = QFileDialog.getOpenFileName(self, "Select Stego File", "", filters)
        if fname:
            self.stego_path = fname
            self.update_stego_display()

    def run_encoding(self):
        if not self.cover_path:
            QMessageBox.warning(self, "Error", "Please select a cover object!")
            return
        if not self.payload_path:
            QMessageBox.warning(self, "Error", "Please select a payload file!")
            return
        key_text = self.key_input.text().strip()
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Please enter a valid integer key!")
            return
        n_bits = self.lsb_spinbox.value()

        if self.payload_type == "text":
            with open(self.payload_path, 'r', encoding='utf-8') as f:
                payload_content = f.read()
        else:
            payload_content = load_binary_as_text(self.payload_path)

        if self.cover_type == "audio":
            required_bytes = len(payload_content.encode('utf-8'))
            with wave.open(self.cover_path, "rb") as f:
                total_frames = f.getnframes()
            available_bits = total_frames * n_bits
            required_bits = required_bytes * 8
        else:
            required_bits = len(payload_content.encode('utf-8')) * 8
            img = cv2.imread(self.cover_path)
            h, w, _ = img.shape
            available_bits = h * w * 3 * n_bits

        if required_bits > available_bits:
            QMessageBox.critical(self, "Capacity Exceeded",
                f"Payload requires {required_bits} bits.\n"
                f"Cover can only hold {available_bits} bits with {n_bits} LSBs.\n"
                f"Try increasing LSBs or using a larger cover.")
            return

        try:
            output_ext = "wav" if self.cover_type == "audio" else "bmp"
            self.stego_path = f"stego_output.{output_ext}"
            if self.cover_type == "image":
                encode_image(self.cover_path, payload_content, self.stego_path, n_bits, int(key_text))
                diff_msg = compare_images(self.cover_path, self.stego_path)
                if "diff_map.bmp" in diff_msg:
                    self.visualization_label.setPixmap(QPixmap("diff_map.bmp").scaled(400, 200, Qt.KeepAspectRatio))
                else:
                    self.visualization_label.setText(diff_msg)
            else:
                encode_audio(self.cover_path, payload_content, self.stego_path, n_bits, int(key_text))
                diff_msg = compare_wavs(self.cover_path, self.stego_path)
                self.visualization_label.setText(f"Audio samples altered: {diff_msg}")

            self.save_stego_button.setEnabled(True)
            QMessageBox.information(self, "Success", f"Stego file created: {self.stego_path}")

        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))

    def save_stego_file(self):
        if not self.stego_path:
            QMessageBox.warning(self, "Error", "No stego file generated yet!")
            return
        ext = "wav" if self.cover_type == "audio" else "bmp"
        fname, _ = QFileDialog.getSaveFileName(self, "Save Stego File", "", f"{ext.upper()} Files (*.{ext})")
        if fname:
            if not fname.lower().endswith(f".{ext}"):
                fname += f".{ext}"
            os.replace(self.stego_path, fname)
            QMessageBox.information(self, "Saved", f"Stego file saved as {fname}")

    def run_decoding(self):
        if not self.stego_path:
            QMessageBox.warning(self, "Error", "Please select a stego file!")
            return
        key_text = self.key_input_decode.text().strip()
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Please enter the same integer key used during encoding!")
            return
        n_bits = self.lsb_spinbox_decode.value()

        try:
            if self.cover_type == "image":
                hidden_bytes = decode_image(self.stego_path, n_bits, int(key_text))
                self.extracted_data = hidden_bytes.decode('utf-8')
            else:
                hidden_bytes = decode_audio(self.stego_path, n_bits, int(key_text))
                self.extracted_data = hidden_bytes.decode('utf-8')

            self.preview_box.setPlainText(self.extracted_data)
            self.save_payload_button.setEnabled(True)
            self.play_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Payload extracted successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Decoding Failed", str(e))

    def save_decoded_file(self):
        if not self.extracted_data:
            QMessageBox.warning(self, "Error", "No payload extracted yet!")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Extracted Payload", "", "All Files (*)")
        if fname:
            if self.payload_type == "text":
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(self.extracted_data)
            else:
                save_text_as_binary(self.extracted_data, fname)
            QMessageBox.information(self, "Saved", f"Payload saved as {fname}")

    def closeEvent(self, event):
        for tmp in ["diff_map.bmp", "temp_extracted"]:
            if os.path.exists(tmp):
                os.remove(tmp)
        event.accept()

# ---------------- DARK THEME ----------------
def apply_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

# ---------------- ADD THIS FOR AUDIO COMPARISON ----------------
def compare_wavs(orig_path, stego_path):
    try:
        with wave.open(orig_path, "rb") as f1:
            frames1 = f1.readframes(f1.getnframes())
        with wave.open(stego_path, "rb") as f2:
            frames2 = f2.readframes(f2.getnframes())
        if len(frames1) != len(frames2):
            return "Audio lengths differ."
        diff = sum(a != b for a, b in zip(frames1, frames2))
        return f"{diff} samples altered"
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = StegoMainWindow()
    window.show()
    sys.exit(app.exec())