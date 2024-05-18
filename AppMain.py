import sys

from UI.mainWindow import UI_mainWindow
from PyQt5.QtWidgets import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = UI_mainWindow()
    mainwindow.show()

    sys.exit(app.exec_())