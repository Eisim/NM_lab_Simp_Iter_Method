from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout

from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtCore import QUrl

import os

ui_file = './UI/help_window.ui'

class UI_infoWindow(QMainWindow):
    def __init__(self,file_name):
        super(UI_infoWindow, self).__init__()
        uic.loadUi(ui_file, self)

        self.info_widget = QWidget()
        self.layout = QVBoxLayout()

        settings = QWebEngineSettings.globalSettings()
        settings.setAttribute(QWebEngineSettings.PluginsEnabled, True)

        self.info_widget.setLayout(self.layout)
        self.view = QWebEngineView()

        self.view.setUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join('info', file_name))))

        self.layout.addWidget(self.view)

        self.setCentralWidget(self.view)

        # self.info_widget.show()