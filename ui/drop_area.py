from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont

class DropArea(QLabel):
    dropped = Signal(str)  # signal fires when file drops
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        
        font = QFont()
        font.setPointSize(16)
        self.setFont(font)
        
        self.setText("drop image here or click browse...")
        
        self.setStyleSheet("""
            border: 1px solid #353535; 
            border-radius: 8px;
            color: #a0a0a0;
            background-color: #1a1a1a;
            padding: 20px;
        """)
        
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def dragEnterEvent(self, event):
        """handle image files being dragged in"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # highlight to show valid drop
                    self.setStyleSheet("""
                        border: 2px solid #2c5dba;
                        border-radius: 8px;
                        color: #e0e0e0;
                        background-color: rgba(44, 93, 186, 0.2);
                        padding: 20px;
                    """)
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """reset style when drag exits"""
        self.setStyleSheet("""
            border: 1px solid #353535; 
            border-radius: 8px;
            color: #a0a0a0;
            background-color: #1a1a1a;
            padding: 20px;
        """)
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event):
        """process dropped image file"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            # reset style
            self.setStyleSheet("""
                border: 1px solid #353535; 
                border-radius: 8px;
                color: #a0a0a0;
                background-color: #1a1a1a;
                padding: 20px;
            """)
            self.dropped.emit(file_path)
            
    def setPixmap(self, pixmap):
        """keep styling when showing image"""
        super().setPixmap(pixmap)
        self.setStyleSheet("""
            border: 1px solid #353535; 
            border-radius: 8px;
            background-color: #1a1a1a;
            padding: 20px;
        """)