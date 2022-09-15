
# the original repository: https://github.com/hofmann-tobias/cpselect

import sys, os
import qtawesome as qta
import numpy as np

from typing import List
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

def cpselect(img_1, img_2):
    """
    Tool for selection a individual number of control points in any two pictures

    :param img_path1: path to the first image
    :param img_path2: path to the second image

    :return: list with a dictionary for each control point
    """
    global img1
    global img2
    img1 = img_1
    img2 = img_2

    app = QApplication(sys.argv)
    cps = _MainWindow()
    cps.raise_()
    app.exec_()

    points = []
    for cp in cps.wp.canvas.CPlist:
        points.append(cp.getdict)

    del img1, img2

    return points

class _MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.createWindow()
        self.createConn()

        self.showMaximized()
        self.show()
        self.wp.canvas.fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)

        q1 = QTimer(self)
        q1.setSingleShot(False)
        q1.timeout.connect(self.updateGUI)
        q1.start(100)

    def createWindow(self):
        self.setWindowTitle('Control Point Selection Tool')
        self.setWindowIcon(qta.icon('fa5s.object-group'))

        # Main Tab Frame
        main_widget = QTabWidget(self)
        self.setCentralWidget(main_widget)

        # Modalities Tab
        widget = QWidget()
        hlay = QHBoxLayout(widget)
        vlay = QVBoxLayout()
        vlay2 = QVBoxLayout()
        vlay2.setSpacing(10)
        hlay_buttons = QHBoxLayout()

        hlay.addLayout(vlay)
        hlay.addLayout(vlay2)

        self.wp = _WidgetPlot(self)
        vlay.addWidget(self.wp)

        main_widget.addTab(widget, qta.icon('fa5s.layer-group'), 'Modalities')

        self.help = QTextEdit()
        self.help.setReadOnly(True)
        self.help.setMaximumWidth(350)
        self.help.setMinimumHeight(540)

        help_header = '''
            <!DOCTYPE html>
                <html lang="de" id="main">
                    <head>
                        <meta charset="UTF-8">
                        <title>About the tool</title>
                        <style>td,th{font-size:14px;}p{font-size: 14px;}</style>
                    </head>
        '''
        help_body = '''
            <body>
                <p>&nbsp;</p>
                <p><br /><img src="{}" alt="ulaval" width="280" height="150" /></p>
                <h2>LeManchot-Fusion&emsp;</h2>
                <p>The fusion subsystem of the LeManchot platform. It provides the required tools for fusion and registration of RGBD&amp;T data.</p>
                <p></p>
                <table style="border-collapse: collapse; width: 59.8011%; height: 90px;" border="1">
                <tbody>
                <tr style="height: 18px;">
                <td style="width: 14.6307%; height: 18px;"><strong>Researcher</strong></td>
                <td style="width: 45.1704%; height: 18px;">Parham Nooralishahi</td>
                </tr>
                <tr style="height: 18px;">
                <td style="width: 14.6307%; height: 18px;">&nbsp;</td>
                <td style="width: 45.1704%; height: 18px;">Sandra Pozzer</td>
                </tr>
                <tr style="height: 18px;">
                <td style="width: 14.6307%; height: 18px;">&nbsp;</td>
                <td style="width: 45.1704%; height: 18px;">Gabriel Ramos</td>
                </tr>
                <tr style="height: 18px;">
                <td style="width: 14.6307%; height: 18px;"><strong>Supervisor</strong></td>
                <td style="width: 45.1704%; height: 18px;">Professor Xavier Maldague</td>
                </tr>
                <tr style="height: 18px;">
                <td style="width: 14.6307%; height: 18px;"><strong>Organization</strong></td>
                <td style="width: 45.1704%; height: 18px;">Universit&eacute; Laval</td>
                </tr>
                </tbody>
                </table>
                <p>&nbsp;</p>
                <h4>Control Point Selector toolbox</h4>
                <p>This toolbox enables you to select the control points in order to align the two modalities.</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
            </body>
            </html>
        '''
        help_html = help_header + help_body.format(os.path.join('resources/ulaval.jpg'))
        self.help.insertHtml(help_html)
        
        self.cpTabelModel = QStandardItemModel(self)
        self.cpTable = QTableView(self)
        self.cpTable.setModel(self.cpTabelModel)
        self.cpTable.setMaximumWidth(350)

        self.delButton = QPushButton('Delete Selected CPs')
        self.delButton.setStyleSheet("font-size: 16px")

        self.pickButton = QPushButton("Pick CPs")
        self.pickButton.setFixedHeight(60)
        self.pickButton.setStyleSheet("color: red; font-size: 16px;")

        self.exitButton = QPushButton('Done')
        self.exitButton.setFixedHeight(60)
        self.exitButton.setStyleSheet("font-size: 16px;")

        vlay2.addWidget(self.help)
        vlay2.addWidget(self.cpTable)
        vlay2.addWidget(self.delButton)

        vlay2.addLayout(hlay_buttons)
        hlay_buttons.addWidget(self.pickButton)
        hlay_buttons.addWidget(self.exitButton)

        self.updateCPtable()
        self.statusBar().showMessage('Ready')

    def createConn(self):
        self.pickButton.clicked.connect(self.pickmodechange)
        self.exitButton.clicked.connect(self.menu_quit)
        self.delButton.clicked.connect(self.delCP)

    def menu_quit(self):
        self.close()

    def pickmodechange(self):
        if self.wp.canvas.toolbar.mode.value in ['', None]:
            if self.wp.canvas.pickmode == True:
                self.wp.canvas.pickMode_changed = True
                self.wp.canvas.pickmode = False
                self.statusBar().showMessage('Pick Mode deactivate.')
                self.wp.canvas.cursorGUI = 'arrow'
                self.wp.canvas.cursorChanged = True
            else:
                self.wp.canvas.pickMode_changed = True
                self.wp.canvas.pickmode = True
                self.wp.canvas.toolbar._active = ''
                self.statusBar().showMessage('Pick Mode activate. Select Control Points.')
        else:
            self.statusBar().showMessage(
                f'Please, first deactivate the selected navigation tool {self.wp.canvas.toolbar.mode.value}', 3000)

    def delCP(self):
        rows = self.cpTable.selectionModel().selectedRows()
        for row in rows:
            try:
                idp = int(row.data())
                for cp in self.wp.canvas.CPlist:
                    if cp.idp == idp:
                        index = self.wp.canvas.CPlist.index(cp)
                        self.wp.canvas.CPlist.pop(index)
            except:
                pass

        self.wp.canvas.updateCanvas()
        self.wp.canvas.cpChanged = True

    def updateGUI(self):

        # if self.wp.canvas.toolbar._active not in ['', None]:
        if self.wp.canvas.toolbar.mode.value not in ['', None]:
            self.wp.canvas.pickmode = False
            self.wp.canvas.pickMode_changed = True

        if self.wp.canvas.pickMode_changed:
            if not self.wp.canvas.pickmode:
                self.pickButton.setStyleSheet("color: red; font-size: 20px;")
            elif self.wp.canvas.pickmode:
                self.pickButton.setStyleSheet("color: green; font-size: 20px;")
            self.wp.canvas.pickMode_changed = False

        if self.wp.canvas.cursorChanged:
            if self.wp.canvas.cursorGUI == 'cross':
                QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
            elif self.wp.canvas.cursorGUI == 'arrow':
                QApplication.restoreOverrideCursor()
            self.wp.canvas.cursorChanged = False

        if self.wp.canvas.cpChanged:
            self.updateCPtable()

    def updateCPtable(self):
        self.wp.canvas.cpChanged = False
        self.cpTable.clearSelection()
        self.cpTabelModel.clear()
        self.cpTabelModel.setHorizontalHeaderLabels(
            ['#ID', 'x (M1)', 'y (M1)', 'x (M2)', 'y (M2)'])

        for cp in self.wp.canvas.CPlist:
            idp, x1, y1, x2, y2 = cp.coordText

            c1 = QStandardItem(idp)
            c2 = QStandardItem(x1)
            c3 = QStandardItem(y1)
            c4 = QStandardItem(x2)
            c5 = QStandardItem(y2)

            row = [c1, c2, c3, c4, c5]

            for c in row:
                c.setTextAlignment(Qt.AlignCenter)
                c.setFlags(Qt.ItemIsEditable)
                c.setFlags(Qt.ItemIsSelectable)

            self.cpTabelModel.appendRow(row)

        self.cpTable.resizeColumnsToContents()

class _WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = _PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

class _PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
        self.createConn()

        self.figureActive = False
        self.axesActive = None
        self.CPactive = None
        self.pickmode = False
        self.pickMode_changed = True
        self.cpChanged = False
        self.cursorGUI = 'arrow'
        self.cursorChanged = False
        self.CPlist = []
        self.lastIDP = 0

    def plot(self):
        gs0 = self.fig.add_gridspec(1, 2)

        self.ax11 = self.fig.add_subplot(gs0[0], xticks=[], yticks=[], title='Modality 1 (M1)')
        self.ax12 = self.fig.add_subplot(gs0[1], xticks=[], yticks=[], title='Modality 2 (M2)')

        self.ax11.imshow(img1)
        self.ax12.imshow(img2)

    def updateCanvas(self, event=None):
        ax11_xlim = self.ax11.get_xlim()
        ax11_xvis = ax11_xlim[1] - ax11_xlim[0]
        ax12_xlim = self.ax12.get_xlim()
        ax12_xvis = ax12_xlim[1] - ax12_xlim[0]

        while len(self.ax11.patches) > 0:
            [p.remove() for p in self.ax11.patches]
        while len(self.ax12.patches) > 0:
            [p.remove() for p in self.ax12.patches]
        while len(self.ax11.texts) > 0:
            [t.remove() for t in self.ax11.texts]
        while len(self.ax12.texts) > 0:
            [t.remove() for t in self.ax12.texts]

        ax11_units = ax11_xvis * 0.003
        ax12_units = ax12_xvis * 0.003

        for cp in self.CPlist:
            x1 = cp.img1x
            y1 = cp.img1y
            x2 = cp.img2x
            y2 = cp.img2y
            idp = str(cp.idp)

            if x1:
                symb1 = plt.Circle((x1, y1), ax11_units * 8, fill=False, color='red')
                symb2 = plt.Circle((x1, y1), ax11_units * 1, fill=True, color='red')
                self.ax11.text(x1 + ax11_units * 5, y1 + ax11_units * 5, idp)
                self.ax11.add_patch(symb1)
                self.ax11.add_patch(symb2)

            if x2:
                symb1 = plt.Circle((x2, y2), ax12_units * 8, fill=False, color='red')
                symb2 = plt.Circle((x2, y2), ax12_units * 1, fill=True, color='red')
                self.ax12.text(x2 + ax12_units * 5, y2 + ax12_units * 5, idp)
                self.ax12.add_patch(symb1)
                self.ax12.add_patch(symb2)

        self.fig.canvas.draw()

    def createConn(self):
        self.fig.canvas.mpl_connect('figure_enter_event', self.activeFigure)
        self.fig.canvas.mpl_connect('figure_leave_event', self.leftFigure)
        self.fig.canvas.mpl_connect('axes_enter_event', self.activeAxes)
        self.fig.canvas.mpl_connect('button_press_event', self.mouseClicked)
        self.ax11.callbacks.connect('xlim_changed', self.updateCanvas)
        self.ax12.callbacks.connect('xlim_changed', self.updateCanvas)

    def activeFigure(self, event):
        self.figureActive = True
        if self.pickmode and self.cursorGUI != 'cross':
            self.cursorGUI = 'cross'
            self.cursorChanged = True

    def leftFigure(self, event):
        self.figureActive = False
        if self.cursorGUI != 'arrow':
            self.cursorGUI = 'arrow'
            self.cursorChanged = True

    def activeAxes(self, event):
        self.axesActive = event.inaxes

    def mouseClicked(self, event):
        x = event.xdata
        y = event.ydata

        if self.toolbar.mode != '':
            self.pickmode = False

        if self.pickmode and (event.inaxes == self.ax11 or event.inaxes == self.ax12):

            if self.CPactive and not self.CPactive.status_complete:
                self.CPactive.appendCoord(x, y)
                self.cpChanged = True
            else:
                idp = self.lastIDP + 1
                cp = _ControlPoint(idp, x, y, self)
                self.CPlist.append(cp)
                self.cpChanged = True
                self.lastIDP += 1

            self.updateCanvas()

class _ControlPoint:
    def __init__(self, idp, x, y, other):
        self.img1x = None
        self.img1y = None
        self.img2x = None
        self.img2y = None
        self.status_complete = False
        self.idp = idp

        self.mn = other
        self.mn.CPactive = self

        self.appendCoord(x, y)

    def appendCoord(self, x, y):

        if self.mn.axesActive == self.mn.ax11 and self.img1x is None:
            self.img1x = x
            self.img1y = y
        elif self.mn.axesActive == self.mn.ax12 and self.img2x is None:
            self.img2x = x
            self.img2y = y

        else:
            raise Exception("Please, select the control point in the other image")

        if self.img1x and self.img2x:
            self.status_complete = True
            self.mn.cpActive = None

    @property
    def coord(self):
        return self.idp, self.img1x, self.img1y, self.img2x, self.img2y

    @property
    def coordText(self):
        if self.img1x and not self.img2x:
            return str(round(self.idp, 2)), str(round(self.img1x, 2)), str(round(self.img1y, 2)), '', ''
        elif not self.img1x and self.img2x:
            return str(round(self.idp, 2)), '', '', str(round(self.img2x, 2)), str(round(self.img2y, 2))
        else:
            return str(round(self.idp, 2)), str(round(self.img1x, 2)), str(round(self.img1y, 2)), str(
                round(self.img2x, 2)), str(round(self.img2y, 2))

    def __str__(self):
        return f"CP {self.idp}: {self.coord}"

    @property
    def getdict(self):

        dict = {
            'point_id': self.idp,
            'img1_x': self.img1x,
            'img1_y': self.img1y,
            'img2_x': self.img2x,
            'img2_y': self.img2y,
        }

        return dict