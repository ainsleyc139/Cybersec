import sys, os, base64, wave, importlib.util
import cv2, numpy as np
from PySide6.QtCore import Qt, QRect, QSize, QPoint, Signal
from PySide6.QtGui import QColor, QPalette, QPainter, QBrush, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QFormLayout, QTabWidget, QSpinBox, QTextEdit,
    QFileDialog, QMessageBox, QStackedWidget, QSizePolicy, QRadioButton,
    QRubberBand, QSplitter
)

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

        # Key (optional UI for consistency)
        s['key'] = QLineEdit(); s['key'].setEchoMode(QLineEdit.Password); s['key'].setPlaceholderText("(Optional) Not used for image backend")
        style_line_edit(s['key'])
        form.addRow("Key:", s['key'])

        if media == "image":
            s['mode_text'] = QRadioButton("Text")
            s['mode_file'] = QRadioButton("File")
            s['mode_file'].setChecked(True)
            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel("Payload Type:"))
            mode_row.addWidget(s['mode_text'])
            mode_row.addWidget(s['mode_file'])
            form.addRow(mode_row)

            s['text_input'] = QLineEdit()
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

        settings.setLayout(form)

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

        # RIGHT PANE (large preview)
        right = QWidget()
        rv = QVBoxLayout()
        if media == "image":
            s['preview'] = ImagePreviewSelector()
            s['preview'].regionSelected.connect(lambda x1,y1,x2,y2, m=media: self._apply_region_from_preview(m, x1,y1,x2,y2))
            rv.addWidget(s['preview'])
            reset_btn = QPushButton("Reset Region"); reset_btn.clicked.connect(lambda checked=False, m=media: self._reset_region(m))
            rv.addWidget(reset_btn, alignment=Qt.AlignRight)
        else:
            placeholder = QLabel("Audio does not have a visual preview.\nUse WAV cover/stego and set LSBs/Key.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color:#aaa; border:1px dashed #333; min-height:360px;")
            rv.addWidget(placeholder)
        right.setLayout(rv)

        # Splitter arrangement
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)  # left (compact)
        splitter.setStretchFactor(1, 3)  # right (big preview)

        page_layout = QVBoxLayout()
        page_layout.addWidget(splitter)
        page.setLayout(page_layout)

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
            s['preview'] = QTextEdit(); s['preview'].setReadOnly(True)
            s['preview'].setPlaceholderText("Extracted payload (BASE64 text preview)…")
            style_text_edit(s['preview'])
            rv.addWidget(s['preview'])
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
            filt = "WAV Audio (*.wav);;All Files (*)" if which in ("cover","stego") else "All Files (*)"
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
        key_text = s['key'].text().strip() if 'key' in s else ""
        if not cover:
            QMessageBox.warning(self, "Error", "Select a WAV cover.")
            return
        if not payload_path:
            QMessageBox.warning(self, "Error", "Select a payload file (will be embedded as text).")
            return
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Key must be an integer.")
            return
        payload_text = load_binary_as_text(payload_path)
        output = "stego_output.wav"
        try:
            encode_audio(cover, payload_text, output, n_bits, int(key_text))
            self.state['audio']['last_stego'] = output
            s['status'] = s.get('status') or QLabel()
            s['status'].setText(f"✅ Stego created: {output}")
        except Exception as e:
            QMessageBox.critical(self, "Encoding Failed", str(e))

    def _run_decoding_audio(self):
        s = self.state['audio']['decode']
        stego = s.get('stego_path')
        n_bits = s['lsb'].value()
        key_text = s['key'].text().strip() if 'key' in s else ""
        if not stego:
            QMessageBox.warning(self, "Error", "Select a stego WAV.")
            return
        if not key_text.isdigit():
            QMessageBox.warning(self, "Error", "Key must be an integer.")
            return
        try:
            text = decode_audio(stego, n_bits, int(key_text))
            if 'preview' not in s:
                s['preview'] = QTextEdit(); s['preview'].setReadOnly(True)
                style_text_edit(s['preview'])
            s['preview'].setPlainText(text)
            s['status'] = s.get('status') or QLabel()
            s['status'].setText("✅ Payload extracted (BASE64 TEXT). Use 'Save Extracted…' to write binary.")
            self.state['audio']['last_extracted_text'] = text
        except Exception as e:
            QMessageBox.critical(self, "Decoding Failed", str(e))

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
