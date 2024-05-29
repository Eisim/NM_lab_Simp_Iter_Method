from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from UI.infoWindow import UI_infoWindow
import numpy as np
import pandas as pd
import ctypes
import os

from PyQt5.QtCore import QThread, pyqtSignal

from threading import Thread

is_busy = False

ui_file = './UI/MainWindow.ui'
import time

class ProgressThread(QThread):
    progress_signal  = pyqtSignal(int)
    def __init__(self,progress_func, max_progress_value = 10000):
        super().__init__()
        self.progress = 0
        self.progress_func = progress_func
        self.max_progress_value = max_progress_value
    def run(self):
        while True:
            self.progress = int(self.progress_func()*self.max_progress_value)
            self.progress_signal.emit(self.progress)
            time.sleep(0.5)
def create_plot(parent):
    parent.fig = Figure(figsize=(parent.width() / 100, parent.height() / 100))
    parent.canvas = FigureCanvas(parent.fig)
    parent.plot = parent.fig.add_subplot(projection='3d')
    return parent.plot


class UI_mainWindow(QMainWindow):
    def __init__(self):
        super(UI_mainWindow, self).__init__()
        uic.loadUi(ui_file, self)

        # создание окон для графиков
        self.plt = create_plot(self.plot_widget_func)
        self.plt_widget_dif = create_plot(self.plot_widget_dif)
        # присвоение мест для окон
        self.plot_widget_func.canvas.setParent(self.plot_widget_func)
        self.plot_widget_dif.canvas.setParent(self.plot_widget_dif)
        self.tabWidget.currentChanged.connect(
            self.toolBar_changing)  # задание функционала. В данной строке: Меняет тулбар при переходе на другую вклвдку
        self.plot_toolBar = NavigationToolbar(self.plot_widget_func.canvas, self)

        self.addToolBar(self.plot_toolBar)  # создание тулбара

        # функционал кнопок
        self.threads = {"calculating":None, "progress_bar":None}

        self.calculate_button.clicked.connect(self.new_thread_calculating)

        self.plot_button.clicked.connect(
            self.plotting)  # задание функционала. В данной строке: построение графика при нажатии на кнопку "Построить"
        self.delete_plot.clicked.connect(
            self.clear_plots)  # задание функционала. В данной строке: очистка окон от ВСЕХ графиков (чистит все окна(графики и таблицу))
        # функционал списков
        self.standart_params()
        # Названия осей
        self.plot_widget_func.plot.set_xlabel("x")
        self.plot_widget_func.plot.set_ylabel("y")
        self.plot_widget_func.plot.set_zlabel("$V(x,y)$")

        self.plot_widget_dif.plot.set_xlabel("x")
        self.plot_widget_dif.plot.set_ylabel("y")
        self.plot_widget_dif.plot.set_zlabel("$U(x,y) - V(x,y)$")

        # подключение библиотеки
        lib_dir = os.path.join(os.curdir, 'dll', 'Release', "libNM1_lib.dll")  # Что запускаем
        lib = ctypes.windll.LoadLibrary(lib_dir)

        self.calculating_func = lib.main_f
        self.calculating_func.argtypes = [ctypes.c_int, ctypes.c_int,  # n, m
                            ctypes.c_int,  # N_max
                            ctypes.c_double,  # eps
                            ctypes.c_double,  # accur
                            ctypes.c_int,  # accur_exit
                            ctypes.c_int  # eps_exit
                            ]
        self.calculating_func.restype = ctypes.c_void_p

        self.progress_func = lib.get_iteration
        self.progress_func.argtypes = []
        self.progress_func.restype = ctypes.c_float


        self.progressBar.setMaximum(10000)
        self.progress_thread = ProgressThread(self.progress_func,10000)
        self.progress_thread.progress_signal.connect(self.update_progress_bar)
        self.progress_thread.start()
        # настройка включения второго окна
        # self.info_button.triggered.connect(lambda: self.info_window("my_info.pdf"))

    def update_progress_bar(self,val):
        self.progressBar.setValue(val)
    def new_thread_calculating(self):
        global is_busy
        if is_busy:
            return
        is_busy = True
        self.threads['calculating'] = Thread(target=self.calculating)
        self.threads['calculating'].start()
        #self.threads['progress_bar'] = Thread(target=self.run_progressbar)
        #self.threads['progress_bar'].start()
    def run_progressbar(self):
        global is_busy
        while self.progressBar.value() < self.progressBar_MaxValue and is_busy:
            cur_step = self.progress_func()
            cur_percent = int(cur_step * self.progressBar_MaxValue)
            self.progressBar.setValue(cur_percent)
        self.progressBar.setValue(0)

    def standart_params(self):
        self.n, self.m, self.N_max, self.eps, self.accur = (10, 10, 100, 5e-7, 1e-12)
        self.input_n.setText(str(self.n))
        self.input_m.setText(str(self.m))
        self.input_N_max.setText(str(self.N_max))
        self.input_eps.setText(str(self.eps))
        self.input_accur.setText(str(self.accur))

    def clear_plots(self):
        self.clear_plot(self.plot_widget_func)
        self.clear_plot(self.plot_widget_dif)
        self.clear_table(self.info_table_v1)
        self.clear_table(self.info_table_v2)
        self.clear_table(self.info_table_dif1)
        self.clear_table(self.info_table_dif2)
        self.clear_table(self.info_table_r1)
        self.clear_table(self.info_table_r2)
        self.clear_exrta_info_table()

    def clear_plot(self, cur_plot_widget):
        cur_plot_widget.plot.cla()
        cur_plot_widget.canvas.draw()  # обновление окна
        # Названия осей
        cur_plot_widget.plot.set_xlabel("x")
        cur_plot_widget.plot.set_ylabel("y")
        cur_plot_widget.canvas.draw()

    def toolBar_changing(self, index):  # изменение привязки тулбара
        self.removeToolBar(self.plot_toolBar)
        if index == 0:  # тулбал для вкладки # Функция и сплайн на одном графике
            self.plot_toolBar = NavigationToolbar(self.plot_widget_func.canvas, self)
        elif index == 1:  # тулбар для вкладки # График первых производных функции и сплайна
            self.plot_toolBar = NavigationToolbar(self.plot_widget_dif.canvas, self)
        self.addToolBar(self.plot_toolBar)

    def file_to_table(self, file_name):  # из str делает list(list(str))
        if len(file_name.split('.')) == 1:
            file_name += '.txt'
        table = []
        with open(file_name, 'r') as f:
            for line in f:
                table.append(line.split(' '))
        return table

    def clear_exrta_info_table(self):
        while self.extra_info_layout.count():
            item = self.extra_info_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_extra_info_table(self, df):
        self.clear_exrta_info_table()
        i = 0
        cols = df.columns
        for i in range(len(cols)):
            cur_text = f"{cols[i]} {df.iloc[0, i]}"
            self.extra_info_layout.addWidget(QLabel(cur_text, self))
            i += 1

    def calculating(self):
        global is_busy
        # lib_dir = os.path.join(os.curdir, "libNM1_lib.dll")

        self.n = int(self.input_n.text())
        self.m = int(self.input_m.text())
        self.N_max = int(self.input_N_max.text())
        self.eps = float(self.input_eps.text())
        self.accur = float(self.input_accur.text())
        self.accur_exit = int(self.check_accur.isChecked())
        self.eps_exit = int(self.check_eps.isChecked())
        self.statusBar.showMessage(f'Расчёт начался')
        self.calculating_func(self.n, self.m, self.N_max, self.eps, self.accur, self.accur_exit, self.eps_exit)
        is_busy = False
        self.statusBar.showMessage(f'Расчёт завершен')


    def plotting(self):
        self.statusBar.showMessage(f'Строятся графики')
        a = 0
        b = 1
        c = 0
        d = 1

        f_v1 = "data/v_part1.csv"
        f_v2 = "data/v_part2.csv"

        f_r1 = "data/r_part1.csv"
        f_r2 = "data/r_part2.csv"

        f_dif1 = "data/dif_part1.csv"
        f_dif2 = "data/dif_part2.csv"

        f_extra_info = "data/extra_info.csv"



        self.clear_exrta_info_table()
        self.update_extra_info_table(pd.read_csv(f_extra_info, encoding='cp1251'))

        table_v1 = np.genfromtxt(f_v1, delimiter=',')
        table_v2 = np.genfromtxt(f_v2, delimiter=',')
        table_dif1 = np.genfromtxt(f_dif1, delimiter=',')
        table_dif2 = np.genfromtxt(f_dif2, delimiter=',')
        table_r1 = np.genfromtxt(f_r1, delimiter=',')
        table_r2 = np.genfromtxt(f_r2, delimiter=',')
        n = table_v1.shape[0]+table_v2.shape[0]-1
        m =table_v1.shape[1] - 1
        first_part2_index = m // 2 + 1
        x_arr = np.linspace(a, b, n + 1)
        y_arr = np.linspace(c, d, m + 1)

        x1_arr = x_arr[0:n // 2 + 1]
        y1_arr = y_arr
        x2_arr = x_arr[n // 2:]
        y2_arr = y_arr[:first_part2_index]

        x1, y1 = np.meshgrid(x1_arr, y1_arr)
        z1 = table_v1.T
        dif1 = table_dif1.T

        x2, y2 = np.meshgrid(x2_arr, y2_arr)
        dif2 = np.insert(table_dif2, 0, dif1[:first_part2_index, -1], axis=0).T
        z2 = np.insert(table_v2, 0, z1[:first_part2_index, -1], axis=0).T

        if self.plot_type.currentIndex() == 0:
            self.plot_widget_func.plot.plot_surface(x1, y1, z1, cmap=plt.cm.plasma)
            self.plot_widget_dif.plot.plot_surface(x1, y1, dif1, cmap=plt.cm.plasma)
            self.plot_widget_func.plot.plot_surface(x2, y2, z2, cmap=plt.cm.plasma)
            self.plot_widget_dif.plot.plot_surface(x2, y2, dif2, cmap=plt.cm.plasma)
        if self.plot_type.currentIndex() == 1:
            self.plot_widget_func.plot.scatter(x1, y1, z1, c=z1, alpha = 1)
            self.plot_widget_dif.plot.scatter(x1, y1, dif1, c=dif1, alpha = 1)
            self.plot_widget_func.plot.scatter(x2, y2, z2, c=z2, alpha = 1)
            self.plot_widget_dif.plot.scatter(x2, y2, dif2, c=dif2, alpha = 1)

        self.clear_table(self.info_table_v1)
        self.clear_table(self.info_table_v2)
        self.clear_table(self.info_table_dif1)
        self.clear_table(self.info_table_dif2)
        self.clear_table(self.info_table_r1)
        self.clear_table(self.info_table_r2)

        self.set_table(self.info_table_v1, table_v1)
        self.set_table(self.info_table_v2, table_v2)
        self.set_table(self.info_table_dif1, table_dif1)
        self.set_table(self.info_table_dif2, table_dif2)
        self.set_table(self.info_table_r1, table_r1)
        self.set_table(self.info_table_r2, table_r2)

        self.plot_widget_func.canvas.draw()
        self.plot_widget_dif.canvas.draw()
        self.statusBar.showMessage(f'Графики построены')
    def set_row(self, table, row):
        max_row_index = table.rowCount()
        table.insertRow(max_row_index)  # создание строки
        for i in range(len(row)):
            table.setItem(max_row_index, i, QTableWidgetItem(str(row[i])))  # заполнение элементами

    def set_table(self, table, data):
        if isinstance(data, np.ndarray):
            table.setColumnCount(data.shape[0])
        elif isinstance(data, pd.DataFrame):
            table.setColumnCount(data.shape[0])
            table.setHorizontalHeaderLabels(data.columns)
        for row in data.T:
            self.set_row(table, row)

    def clear_table(self, table):
        while (table.rowCount() > 0):
            table.removeRow(0)

