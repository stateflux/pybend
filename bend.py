#!/usr/bin/env python3
"""
pybend: manipulate raw pixel data to create glitch art
"""

import sys
from PySide6.QtWidgets import QApplication

from effects import EFFECTS, EFFECT_SEQUENCE, perf_logger

from ui import DatabenderApp

def main():
    """init + run app"""
    app = QApplication(sys.argv)
    window = DatabenderApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 