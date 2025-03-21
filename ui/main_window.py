import os
import sys
import io
import random
import math
import time
import numpy as np
from PIL import Image, ImageQt
from PIL.ExifTags import TAGS

# qt imports
from PySide6.QtWidgets import (
    QMainWindow, QFrame, QLabel, QSlider, 
    QCheckBox, QPushButton, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit,
    QGroupBox, QWidget, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor

# custom drop zone
from ui.drop_area import DropArea

# fx package
from effects import EFFECTS, EFFECT_SEQUENCE, perf_logger

class DatabenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pybend - image glitch tool")
        self.resize(800, 667) 
        self.setMinimumSize(800, 667)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.apply_dark_theme()
        self.input_image = None
        self.output_image = None
        self.original_image_data = None
        self.glitched_image_data = None
        
        # metadata dict
        self.image_metadata = {
            'width': 0,
            'height': 0,
            'mode': '',
            'format': '',
            'exif': {}
        }
        
        # hook perf logger to ui
        self.perf_logger = perf_logger
        
        # warm jit funcs
        self.warmup_effects()
        
        # build ui
        self.create_ui()
        
        # hide glitched preview until we load an image
        self.glitched_preview_frame.hide()
        
        # let original preview take full width at start
        self.original_preview_frame.setMinimumWidth(self.width() - 40)  # margins
    
    def apply_dark_theme(self):
        """set up dark ui theme"""
        # colors
        dark_bg = "#121212"
        darker_bg = "#0a0a0a"
        mid_gray = "#1e1e1e"
        light_gray = "#353535"
        accent = "#2c5dba"  # blue accent
        text = "#e0e0e0"
        
        # create palette
        dark_palette = QPalette()
        
        # map to qt elements
        dark_palette.setColor(QPalette.Window, QColor(dark_bg))
        dark_palette.setColor(QPalette.WindowText, QColor(text))
        dark_palette.setColor(QPalette.Base, QColor(darker_bg))
        dark_palette.setColor(QPalette.AlternateBase, QColor(mid_gray))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(accent))
        dark_palette.setColor(QPalette.ToolTipText, QColor(text))
        dark_palette.setColor(QPalette.Text, QColor(text))
        dark_palette.setColor(QPalette.Button, QColor(light_gray))
        dark_palette.setColor(QPalette.ButtonText, QColor(text))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(accent))
        dark_palette.setColor(QPalette.Highlight, QColor(accent))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        # apply to app
        self.setPalette(dark_palette)
        
        # css overrides
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #353535;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #1a1a1a;
            }
            QLabel {
                color: #e0e0e0;
            }
            QTextEdit {
                background-color: #1a1a1a;
                border: 1px solid #353535;
                border-radius: 4px;
                color: #e0e0e0;
            }
            QScrollBar:vertical {
                width: 12px;
                background: transparent;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #2c5dba;
                min-height: 20px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #3a6fd2;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
    
    def create_ui(self):
        """build main ui components"""
        # root widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # button styling
        button_style = """
            QPushButton {
                background-color: #2c5dba;
                color: #f0f0f0;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 16px;
                min-height: 36px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a6fd2;
            }
            QPushButton:pressed {
                background-color: #1e4293;
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #808080;
            }
        """
        
        # slider styling
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #454545;
                height: 8px;
                background: #2a2a2a;
                margin: 2px 0;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #2c5dba;
                border: 1px solid #2c5dba;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #3a6fd2;
                border: 1px solid #3a6fd2;
            }
        """
        
        # root layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # preview container (65% height)
        preview_frame = QWidget()
        preview_frame.setMinimumHeight(400)
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)
        
        # source img area
        self.original_preview_frame = QGroupBox("")
        original_layout = QVBoxLayout(self.original_preview_frame)
        original_layout.setContentsMargins(5, 0, 5, 5)
        
        # drop target zone
        self.original_preview = DropArea()
        self.original_preview.dropped.connect(self.load_image)
        self.original_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        original_layout.addWidget(self.original_preview, 1)
        
        # output img area
        self.glitched_preview_frame = QGroupBox("")
        glitched_layout = QVBoxLayout(self.glitched_preview_frame)
        glitched_layout.setContentsMargins(5, 0, 5, 5)
        
        self.glitched_preview = QLabel()
        self.glitched_preview.setAlignment(Qt.AlignCenter)
        self.glitched_preview.setStyleSheet("""
            border: 1px solid #353535; 
            border-radius: 8px;
            color: #a0a0a0;
            background-color: #1a1a1a;
            padding: 20px;
        """)
        self.glitched_preview.setMinimumSize(300, 200)
        self.glitched_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        glitched_layout.addWidget(self.glitched_preview)
        
        # arrange previews horizontally
        preview_layout.addWidget(self.original_preview_frame, 1)
        preview_layout.addWidget(self.glitched_preview_frame, 1)
        
        # controls panel
        controls_section = QWidget()
        controls_layout = QGridLayout(controls_section)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(5)
        controls_layout.setHorizontalSpacing(15)
        
        # file picker button
        self.browse_button = QPushButton("browse...")
        self.browse_button.clicked.connect(self.browse_image)
        self.browse_button.setStyleSheet(button_style)
        self.browse_button.setCursor(Qt.PointingHandCursor)
        
        # reroll button
        self.reroll_btn = QPushButton("reroll")
        self.reroll_btn.clicked.connect(self.reglitch_image)
        self.reroll_btn.setEnabled(False)
        self.reroll_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c5dba;
                color: #f0f0f0;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 16px;
                font-weight: bold;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #3a6fd2;
            }
            QPushButton:pressed {
                background-color: #1e4293;
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #808080;
            }
        """)
        self.reroll_btn.setCursor(Qt.PointingHandCursor)
        
        self.save_btn = QPushButton("save")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setStyleSheet(button_style)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        
        # intensity slider group
        intensity_container = QWidget()
        intensity_layout = QVBoxLayout(intensity_container)
        intensity_layout.setContentsMargins(0, 4, 0, 4)
        intensity_layout.setSpacing(4)
        
        # intensity label
        intensity_label = QLabel("<b>intensity</b>: 25")
        intensity_label.setStyleSheet("font-size: 16px;")
        intensity_label.setAlignment(Qt.AlignCenter)
        self.intensity_value_label = intensity_label
        intensity_layout.addWidget(intensity_label)
        
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 200)
        self.intensity_slider.setValue(50)
        self.intensity_slider.setStyleSheet(slider_style)
        self.intensity_slider.valueChanged.connect(self.update_intensity_label)
        intensity_layout.addWidget(self.intensity_slider)
        
        intensity_layout.setAlignment(Qt.AlignTop)
        
        # applied fx readout panel
        effects_container = QWidget()
        effects_layout = QVBoxLayout(effects_container)
        effects_layout.setContentsMargins(0, 0, 0, 0)
        effects_layout.setSpacing(0)
        effects_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # applied fx area
        effects_frame = QFrame()
        effects_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
            }
        """)
        effects_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        effects_frame.setFixedHeight(160)
        
        effects_frame_layout = QVBoxLayout(effects_frame)
        effects_frame_layout.setContentsMargins(8, 6, 8, 8)
        effects_frame_layout.setSpacing(6)
        
        effects_label = QLabel("<b>effects applied</b>")
        effects_label.setStyleSheet("font-size: 16px; color: #e0e0e0; background: transparent; margin: 0; padding: 0px;")
        effects_label.setAlignment(Qt.AlignCenter)
        effects_label.setMaximumHeight(24)
        effects_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.effects_label = effects_label
        effects_frame_layout.addWidget(effects_label)
        
        # fx list area
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.debug_text.setFixedHeight(120)
        self.debug_text.setContentsMargins(0, 0, 0, 0)
        self.debug_text.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
                color: #e0e0e0;
                font-size: 14px;
                padding: 8px 8px 12px 8px;
                margin: 0;
            }
            QScrollBar:vertical {
                width: 12px;
                background: transparent;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #2c5dba;
                min-height: 20px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #3a6fd2;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        
        # remove extra space
        doc = self.debug_text.document()
        doc.setDocumentMargin(0)
        
        effects_frame_layout.addWidget(self.debug_text)
        effects_layout.addWidget(effects_frame)
        
        # layout grid
        controls_layout.addWidget(self.browse_button, 0, 0)
        controls_layout.addWidget(self.reroll_btn, 0, 1)
        controls_layout.addWidget(self.save_btn, 0, 2)
        
        controls_layout.addWidget(intensity_container, 1, 1)
        controls_layout.addWidget(effects_container, 1, 2)
        
        # col sizes - 30:40:30
        controls_layout.setColumnStretch(0, 3)
        controls_layout.setColumnStretch(1, 4)
        controls_layout.setColumnStretch(2, 3)
        
        # main layout
        main_layout.addWidget(preview_frame, 3)
        main_layout.addWidget(controls_section, 1)
        
        # resize handler
        self.resizeEvent = self.on_resize
    
    def warmup_effects(self):
        """precompile jit funcs to avoid startup lag"""
        print("warming up jit funcs...")
        start_time = time.time()
        
        # tiny test img
        width, height = 50, 50
        dummy_data = bytearray(np.zeros((height, width, 3), dtype=np.uint8).tobytes())
        row_length = width * 3
        bytes_per_pixel = 3
        
        # compile each fx
        for effect_name, effect in EFFECTS.items():
            print(f"  compiling {effect_name}...")
            for intensity in [0.1, 0.5, 0.9]:
                try:
                    effect.apply(
                        dummy_data, 
                        width, 
                        height, 
                        row_length, 
                        bytes_per_pixel, 
                        intensity
                    )
                except Exception as e:
                    print(f"  [!] failed jit for {effect_name}: {str(e)}")
        
        elapsed = time.time() - start_time
        print(f">>> jit warmup done in {elapsed:.2f}s")
    
    def browse_image(self):
        """open file picker dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "select image",
            "",
            "image files (*.jpg *.jpeg *.png *.bmp);;all files (*.*)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """load source img and prep for glitching"""
        try:
            self.input_image = Image.open(file_path)
            
            # get img metadata
            self.image_metadata = {
                'width': self.input_image.width,
                'height': self.input_image.height,
                'mode': self.input_image.mode,
                'format': self.input_image.format or 'PNG',
                'exif': {}
            }
            
            # extract exif if any
            if hasattr(self.input_image, '_getexif') and self.input_image._getexif():
                exif_data = self.input_image._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        self.image_metadata['exif'][tag] = value
            
            # show output preview now that we have an image
            if not self.glitched_preview_frame.isVisible():
                self.glitched_preview_frame.show()
                self.original_preview_frame.setMinimumWidth(0)
            
            self.display_original_image()
            
            # convert to raw bytes
            self.original_image_data = self.convert_to_raw(self.input_image)
            
            # enable glitch button
            self.reroll_btn.setEnabled(True)
            
            # run first glitch
            self.process_image()
            
        except Exception as e:
            QMessageBox.critical(self, "error", f"failed to load image: {str(e)}")
    
    def convert_to_raw(self, image):
        """convert image to raw bytes"""
        # force rgb mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            self.image_metadata['mode'] = 'RGB'
        
        return bytearray(image.tobytes())
    
    def reconstruct_image(self, raw_data):
        """rebuild image from glitched bytes"""
        width = self.image_metadata['width']
        height = self.image_metadata['height']
        mode = self.image_metadata['mode']
        
        try:
            # bytes -> image
            new_image = Image.frombytes(mode, (width, height), bytes(raw_data))
        
            # keep original format
            if self.image_metadata['format']:
                new_image.format = self.image_metadata['format']
                
            # exif would be restored here for jpg/tiff
            if self.image_metadata['format'] in ['JPEG', 'JPG', 'TIFF'] and self.image_metadata['exif']:
                pass
                
            return new_image
            
        except Exception as e:
            # fallback on error
            print(f"img reconstruction failed: {str(e)}")
            return Image.new(mode, (width, height), (0, 0, 0))
    
    def randomize_intensity(self, intensity):
        """add ±25% random variance to intensity"""
        dev = 0.25
        factor = 1.0 + random.uniform(-dev, dev)
        return max(1, intensity * factor)
    
    def apply_multiple_glitches(self, raw_data, intensity):
        """stack random fx on image data"""
        start_time = time.time()
        
        # clone source data
        data = bytearray(raw_data)
        
        # fx config
        effect_parameters = {
            "reverser": {
                "strength": 1.0,
                "weight": 1.0
            },
            "bit_depth_reduction": {
                "strength": 1.0,
                "weight": 1.0
            },
            "selective_channel_corruption": {
                "strength": 1.0,
                "weight": 1.0
            },
            "chunk_duplication": {
                "strength": 1.0,
                "weight": 1.0
            },
            "palette_swap": {
                "strength": 1.0,
                "weight": 1.0
            },
            "color_range_shifter": {
                "strength": 1.0,
                "weight": 1.0
            },
            "zeta_invert": {
                "strength": 1.0,
                "weight": 1.0
            },
            "wave_propagation": { 
                "strength": 1.0,
                "weight": 1.0
            },
            "phantom_regions": {
                "strength": 1.0,
                "weight": 1.0
            },
            "mirror": {
                "strength": 1.0, # 1.0 str = 1 region, 2.0 = 2 regions, etc
                "weight": 1.0
            },
            "bleed_blur": {
                "strength": 1.0,
                "weight": 1.0
            },
        }
        
        # scale fx count by intensity
        min_effects = max(1, int(1 + (intensity / 50)))
        max_effects = max(min_effects + 1, int(3 + (intensity / 50)))
        num_effects = random.randint(min_effects, max_effects)
        
        print(f"applying {num_effects} fx @ intensity {intensity:.1f}")
        
        # get available fx
        available_effects = [name for name in EFFECT_SEQUENCE if name in EFFECTS and 
                            effect_parameters.get(name, {"weight": 0})["weight"] > 0]
        
        # fallback if nothing enabled
        if not available_effects:
            print("warning: no fx with non-zero weights, using channel corruption as fallback")
            available_effects = ["selective_channel_corruption"]
            effect_parameters["selective_channel_corruption"]["weight"] = 1.0
        
        # build weighted selection pool
        weighted_effects = []
        for effect_name in available_effects:
            params = effect_parameters.get(effect_name, {"strength": 1.0, "weight": 1.0})
            weight = params["weight"]
            # add copies based on weight
            for _ in range(int(weight * 10)):
                weighted_effects.append(effect_name)
        
        # cap fx count to what's available
        if num_effects > len(available_effects):
            num_effects = len(available_effects)
            print(f"capping at {num_effects} available fx")
        
        # select fx
        if num_effects >= len(available_effects):
            selected_effects = available_effects.copy()
            print("using all available fx")
        else:
            # pick unique fx from weighted pool
            random.shuffle(weighted_effects)
            selected_effects = []
            for effect in weighted_effects:
                if effect not in selected_effects:
                    selected_effects.append(effect)
                    if len(selected_effects) >= num_effects:
                        break
        
        # track what we apply
        applied_effects = []
        
        # get img dimensions
        width = self.image_metadata['width']
        height = self.image_metadata['height']
        bytes_per_pixel = 3  # RGB
        row_length = width * bytes_per_pixel
        
        # run each fx
        for effect_name in selected_effects:
            # get config
            params = effect_parameters.get(effect_name, {"strength": 1.0, "weight": 1.0})
            strength_multiplier = params["strength"]
            
            # get fx implementation
            effect = EFFECTS[effect_name]
            
            # randomize intensity
            effect_intensity = self.randomize_intensity(intensity * strength_multiplier)
            
            # scale to 0-1
            normalized_intensity = min(1.0, effect_intensity / 100.0)
            
            print(f"applying {effect_name} @ {effect_intensity:.1f}")
            
            # run fx
            data = effect.apply(data, width, height, row_length, bytes_per_pixel, normalized_intensity)
            
            # log for ui
            applied_effects.append((effect_name, effect_intensity))
        
        # timing
        total_time = (time.time() - start_time) * 1000
        print(f"glitch stack done in {total_time:.2f}ms")
        
        return data, applied_effects
    
    def display_original_image(self):
        """show source img in preview"""
        if self.input_image:
            # get container size
            preview_size = self.original_preview.size()
            
            if preview_size.width() > 0 and preview_size.height() > 0:
                # copy for display
                preview_img = self.input_image.copy()
                
                # calc scaled size
                img_width = self.input_image.width
                img_height = self.input_image.height
                preview_width = preview_size.width() - 10
                preview_height = preview_size.height() - 10
                
                # scale factor
                width_ratio = preview_width / img_width
                height_ratio = preview_height / img_height
                scale_ratio = min(width_ratio, height_ratio)
                
                # new size
                new_width = int(img_width * scale_ratio)
                new_height = int(img_height * scale_ratio)
                
                # resize
                preview_img = preview_img.resize((new_width, new_height), Image.LANCZOS)
                
                # convert for qt
                img_data = preview_img.convert("RGBA").tobytes("raw", "RGBA")
                qimage = QImage(img_data, new_width, new_height, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
                self.original_preview.setPixmap(pixmap)
    
    def display_glitched_image(self):
        """show glitched img in preview"""
        if self.output_image:
            # get container size
            preview_size = self.glitched_preview.size()
            
            if preview_size.width() > 0 and preview_size.height() > 0:
                # copy for display
                preview_img = self.output_image.copy()
                
                # calc scaled size
                img_width = self.output_image.width
                img_height = self.output_image.height
                preview_width = preview_size.width() - 10
                preview_height = preview_size.height() - 10
                
                # scale factor
                width_ratio = preview_width / img_width
                height_ratio = preview_height / img_height
                scale_ratio = min(width_ratio, height_ratio)
                
                # new size
                new_width = int(img_width * scale_ratio)
                new_height = int(img_height * scale_ratio)
                
                # resize
                preview_img = preview_img.resize((new_width, new_height), Image.LANCZOS)
                
                # convert for qt
                img_data = preview_img.convert("RGBA").tobytes("raw", "RGBA")
                qimage = QImage(img_data, new_width, new_height, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
                self.glitched_preview.setPixmap(pixmap)
                
                # keep styling
                self.glitched_preview.setStyleSheet("""
                    border: 1px solid #353535; 
                    border-radius: 8px;
                    background-color: #1a1a1a;
                    padding: 20px;
                """)
    
    def update_debug_info(self, applied_effects):
        """update fx panel with what was applied"""
        # clear old
        self.debug_text.clear()
        
        # update header
        self.effects_label.setText(f"<b>{len(applied_effects)} effects applied</b>")
        
        # build list text
        debug_text = ""
        
        for i, (effect, intensity) in enumerate(applied_effects, 1):
            # prettier name
            effect_name = effect.replace("_", " ").title()
            debug_text += f"{i}. {effect_name}: {intensity:.1f}\n"
        
        self.debug_text.setText(debug_text)
        
        # update window title
        intensity = self.intensity_slider.value()
        displayed_intensity = intensity // 2
        self.setWindowTitle(f"pybend - {len(applied_effects)} effects applied - intensity: {displayed_intensity}")
        # update slider label
        self.intensity_value_label.setText(f"<b>intensity</b>: {displayed_intensity}")
    
    def process_image(self):
        """apply fx stack to image"""
        self.glitched_image_data, self.applied_effects = self.apply_multiple_glitches(self.original_image_data, self.intensity_slider.value())
        self.output_image = self.reconstruct_image(self.glitched_image_data)
        self.display_glitched_image()
        self.update_debug_info(self.applied_effects)
    
    def reglitch_image(self):
        """generate new random fx stack"""
        self.process_image()
    
    def save_image(self):
        """save output to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "save image",
            "",
            "image files (*.jpg *.jpeg *.png *.bmp);;all files (*.*)"
        )
        
        if file_path:
            self.output_image.save(file_path)
    
    def toggle_perf_logging(self):
        """toggle perf metrics"""
        # always on now
        self.perf_logger.enable()
        
    def update_intensity_label(self, value):
        """update slider label"""
        displayed_value = value // 2
        self.intensity_value_label.setText(f"<b>intensity</b>: {displayed_value}")
    
    def on_resize(self, event):
        """handle window resize"""
        # parent handler
        super().resizeEvent(event)
        
        # make original preview full width if needed
        if not self.glitched_preview_frame.isVisible():
            self.original_preview_frame.setMinimumWidth(self.width() - 40)
        
        # refresh previews
        if self.input_image:
            self.display_original_image()
        if self.output_image:
            self.display_glitched_image()
            
    def changeEvent(self, event):
        """handle window state changes"""
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() == Qt.WindowNoState:
                # reset width constraints
                self.original_preview_frame.setMinimumWidth(0)
                self.glitched_preview_frame.setMinimumWidth(0)
                
                if self.glitched_preview_frame.isVisible():
                    self.adjustSize()
                
        super().changeEvent(event) 