import sys, os
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QLineEdit, QFormLayout,
    QTabWidget, QFrame, QSpinBox, QTextEdit, QFileDialog, QMessageBox
)

STOP_MARKER = "====="

# ---------------- Utility Functions ----------------
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

def generate_coords(h, w, key: int):
    """Generate a reproducible sequence of (y, x, c) using key as seed"""
    coords = [(y, x, c) for y in range(h) for x in range(w) for c in range(3)]
    rng = np.random.RandomState(int(key))
    rng.shuffle(coords)
    return coords

def encode(image_name, secret_data, output_name, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for encoding.")
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_name}")

    # Add stop marker
    secret_data = (secret_data or "") + STOP_MARKER
    binary_secret_data = to_bin(secret_data)
    data_len = len(binary_secret_data)

    h, w, _ = image.shape
    coords = generate_coords(h, w, key)

    if data_len > len(coords) * n_bits:
        raise ValueError("Insufficient space, need bigger image or fewer bits per channel.")

    mask = ~((1 << n_bits) - 1) & 0xFF
    data_index = 0
    for (y, x, c) in coords:
        if data_index >= data_len:
            break
        bits = int(binary_secret_data[data_index:data_index+n_bits].ljust(n_bits, '0'), 2)
        image[y, x, c] = np.uint8((int(image[y, x, c]) & mask) | bits)
        data_index += n_bits

    cv2.imwrite(output_name, image)

def decode(image_name, n_bits=1, key=None):
    if key is None:
        raise ValueError("Key is required for decoding.")
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_name}")

    h, w, _ = image.shape
    coords = generate_coords(h, w, key)

    binary_data = ""
    decoded_data = ""
    for (y, x, c) in coords:
        binary_data += to_bin(image[y, x, c])[-n_bits:]
        while len(binary_data) >= 8:
            byte = binary_data[:8]
            binary_data = binary_data[8:]
            decoded_data += chr(int(byte, 2))
            if decoded_data.endswith(STOP_MARKER):
                return decoded_data[:-len(STOP_MARKER)]
    return decoded_data

def compare_images(orig_path, stego_path):
    orig = cv2.imread(orig_path)
    stego = cv2.imread(stego_path)
    if orig is None or stego is None:
        return "Could not open one of the files."
    if orig.shape != stego.shape:
        return "Images have different dimensions."
    diff = np.sum(orig != stego)
    return f"Number of pixel values changed: {diff}"

# ---------------- GUI ----------------
class StegoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔒 LSB Steganography & Steganalysis Tool")
        self.setMinimumSize(1100, 750)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.cover_path = ""
        self.payload_data = ""
        self.stego_path = ""
        self.extracted_data = ""

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        tab_widget.setStyleSheet("QTabBar::tab { min-width: 150px; padding: 10px; font-weight: bold; }")

        self.encode_tab = QWidget()
        self.setup_encode_tab()
        tab_widget.addTab(self.encode_tab, "📝 Encode")

        self.decode_tab = QWidget()
        self.setup_decode_tab()
        tab_widget.addTab(self.decode_tab, "📂 Decode")

        main_layout.addWidget(tab_widget)
        self.central_widget.setLayout(main_layout)

    # ---------------- Encode Tab ----------------
    def setup_encode_tab(self):
        layout = QVBoxLayout()

        cover_group = QGroupBox("Cover Object")
        cover_layout = QHBoxLayout()
        self.cover_label = QLabel("No file selected")
        self.cover_label.setFixedSize(280, 280)
        self.cover_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.cover_label.setAlignment(Qt.AlignCenter)
        self.cover_button = QPushButton("📁 Select Cover File")
        self.cover_button.clicked.connect(self.load_cover_file)
        cover_layout.addWidget(self.cover_label)
        cover_layout.addWidget(self.cover_button)
        cover_group.setLayout(cover_layout)

        payload_group = QGroupBox("Payload")
        payload_layout = QHBoxLayout()
        self.payload_button = QPushButton("📁 Select Payload File")
        self.payload_path = QLineEdit()
        self.payload_path.setReadOnly(True)
        self.payload_button.clicked.connect(self.load_payload_file)
        payload_layout.addWidget(self.payload_button)
        payload_layout.addWidget(self.payload_path)
        payload_group.setLayout(payload_layout)

        settings_group = QGroupBox("Stego Settings")
        form = QFormLayout()
        self.lsb_spinbox = QSpinBox()
        self.lsb_spinbox.setRange(1, 8)
        self.lsb_spinbox.setValue(1)
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        form.addRow("Number of LSBs:", self.lsb_spinbox)
        form.addRow("Key (required integer):", self.key_input)
        settings_group.setLayout(form)

        self.encode_button = QPushButton("🚀 Embed Payload")
        self.save_stego_button = QPushButton("💾 Save Stego File")
        self.encode_button.clicked.connect(self.run_encoding)
        self.save_stego_button.clicked.connect(self.save_stego_file)

        self.visualization_label = QLabel("Visualization / Difference Map")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        self.visualization_label.setStyleSheet("border: 1px dashed gray; padding: 20px; font-style: italic;")

        layout.addWidget(cover_group)
        layout.addWidget(payload_group)
        layout.addWidget(settings_group)
        layout.addWidget(self.encode_button)
        layout.addWidget(self.save_stego_button)
        layout.addWidget(self.visualization_label)

        self.encode_tab.setLayout(layout)

    # ---------------- Decode Tab ----------------
    def setup_decode_tab(self):
        layout = QVBoxLayout()

        stego_group = QGroupBox("Stego Object")
        stego_layout = QHBoxLayout()
        self.stego_label = QLabel("No file selected")
        self.stego_label.setFixedSize(280, 280)
        self.stego_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stego_label.setAlignment(Qt.AlignCenter)
        self.stego_button = QPushButton("📁 Select Stego File")
        self.stego_button.clicked.connect(self.load_stego_file)
        stego_layout.addWidget(self.stego_label)
        stego_layout.addWidget(self.stego_button)
        stego_group.setLayout(stego_layout)

        settings_group = QGroupBox("Decode Settings")
        form = QFormLayout()
        self.lsb_spinbox_decode = QSpinBox()
        self.lsb_spinbox_decode.setRange(1, 8)
        self.lsb_spinbox_decode.setValue(1)
        self.key_input_decode = QLineEdit()
        self.key_input_decode.setEchoMode(QLineEdit.Password)
        form.addRow("Number of LSBs:", self.lsb_spinbox_decode)
        form.addRow("Key (required integer):", self.key_input_decode)
        settings_group.setLayout(form)

        self.decode_button = QPushButton("🔍 Extract Payload")
        self.decode_button.clicked.connect(self.run_decoding)
        self.save_payload_button = QPushButton("💾 Save Extracted File")
        self.save_payload_button.clicked.connect(self.save_decoded_file)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setPlaceholderText("Extracted payload content preview...")
        self.preview_box.setStyleSheet("border: 1px solid gray; background: #222; color: #ddd;")

        layout.addWidget(stego_group)
        layout.addWidget(settings_group)
        layout.addWidget(self.decode_button)
        layout.addWidget(self.save_payload_button)
        layout.addWidget(self.preview_box)

        self.decode_tab.setLayout(layout)

    # ---------------- Encode Logic ----------------
    def load_cover_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Cover Image", "", "BMP Images (*.bmp)")
        if fname:
            self.cover_path = fname
            pixmap = QPixmap(fname).scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cover_label.setPixmap(pixmap)

    def load_payload_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Payload File", "", "Text Files (*.txt)")
        if fname:
            with open(fname, "r", encoding="utf-8") as f:
                self.payload_data = f.read()
            self.payload_path.setText(fname)

    def run_encoding(self):
        if not self.cover_path or not self.payload_data:
            QMessageBox.warning(self, "Error", "Please select cover file and payload first!")
            return
        n_bits = self.lsb_spinbox.value()
        key_text = self.key_input.text().strip()
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Please enter a valid integer key.")
            return
        key_val = int(key_text)
        try:
            output_file = "encoded_output.bmp"
            encode(self.cover_path, self.payload_data, output_file, n_bits, key_val)
            self.stego_path = output_file
            diff_msg = compare_images(self.cover_path, self.stego_path)
            self.visualization_label.setText(f"✅ Payload embedded!\n{diff_msg}")
        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))

    def save_stego_file(self):
        if not self.stego_path:
            QMessageBox.warning(self, "Error", "No stego file generated yet!")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Stego File", "", "BMP Images (*.bmp)")
        if fname:
            os.replace(self.stego_path, fname)
            QMessageBox.information(self, "Saved", f"Stego file saved as {fname}")

    # ---------------- Decode Logic ----------------
    def load_stego_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Stego Image", "", "BMP Images (*.bmp)")
        if fname:
            self.stego_path = fname
            pixmap = QPixmap(fname).scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.stego_label.setPixmap(pixmap)

    def run_decoding(self):
        if not self.stego_path:
            QMessageBox.warning(self, "Error", "Please select a stego file first!")
            return
        n_bits = self.lsb_spinbox_decode.value()
        key_text = self.key_input_decode.text().strip()
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Please enter the same integer key used during encoding.")
            return
        key_val = int(key_text)
        try:
            hidden_message = decode(self.stego_path, n_bits, key_val)
            self.preview_box.setPlainText(hidden_message)
            self.extracted_data = hidden_message
        except Exception as e:
            QMessageBox.critical(self, "Decoding Failed", str(e))

    def save_decoded_file(self):
        if not self.extracted_data:
            QMessageBox.warning(self, "Error", "No payload extracted yet!")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Payload", "", "Text Files (*.txt)")
        if fname:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(self.extracted_data)
            QMessageBox.information(self, "Saved", f"Payload saved as {fname}")

# ---------------- Dark Theme ----------------
def apply_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = StegoMainWindow()
    window.show()
    sys.exit(app.exec())
