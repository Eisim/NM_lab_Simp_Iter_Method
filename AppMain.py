import sys

from UI.mainWindow import UI_mainWindow
from PyQt5.QtWidgets import *
import os
if not os.path.exists(os.path.join('data')):
    os.mkdir('data')
if not os.path.exists(os.path.join('experiments')):
    os.mkdir('experiments')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = UI_mainWindow()
    mainwindow.show()

    sys.exit(app.exec_())