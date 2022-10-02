#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import os.path
from scipy.misc import imread
import numpy as np
from PyQt4 import QtCore, QtGui, uic
from seam_carve import seam_carve
Ui_MainWindow, QtBaseClass = uic.loadUiType('./guiwindow.ui')


class Viewer(QtGui.QWidget):
    def __init__(self, parent=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.INIT_IMAGE = np.ones([350, 350, 3], np.uint8) * 255
        self.INIT_MASK = np.zeros(self.INIT_IMAGE.shape, np.int8)
        self.SAVE_COLOR = np.array([0, 127, 0])
        self.DEL_COLOR = np.array([127, 0, 0])
        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.image = self.INIT_IMAGE.copy()
        self.n_rows = self.INIT_IMAGE.shape[0]
        self.n_cols = self.INIT_IMAGE.shape[1]
        self.qimage = QtGui.QImage(self.image.data, self.n_cols, 
                                   self.n_rows, 
                                   QtGui.QImage.Format_Indexed8)
        self.mask = self.INIT_MASK.copy()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = event.rect()
        painter.drawImage(rect, self.qimage, rect)

    def updateImage(self):
        expanded_mask = (self.mask == 1)[:, :, np.newaxis] * self.SAVE_COLOR + \
                        (self.mask == -1)[:, :, np.newaxis] * self.DEL_COLOR
        buffer = np.require(self.image + expanded_mask, np.uint8, ['A', 'O', 'C'])
        buffer[self.image.astype(np.uint16) + expanded_mask > 255] = 240
        self.qimage = QtGui.QImage(buffer.data, self.n_cols, self.n_rows, 
                                   buffer.shape[1] * 3, QtGui.QImage.Format_RGB888)

    def loadImage(self, img, mask = None):
        self.image = img
        self.qimage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.n_rows, self.n_cols, _ = img.shape
        self.resize(self.n_cols, self.n_rows)
        if mask is None:
            self.mask = np.zeros(img.shape[:2], np.int8)
        else:
            self.mask = mask.copy()
            self.updateImage()
        self.update()

    def clearMask(self):
        self.mask.fill(0)
        self.updateImage()
        self.update()

    def changeMask(self, pos, value, radius):
        for row_i in range(-radius, radius):
            for col_i in range(-radius, radius):
                row = pos.y() + row_i
                col = pos.x() + col_i
                if row >= 0 and row < self.n_rows and \
                        col >= 0 and col < self.n_cols:
                    self.mask[row, col] = value
        self.updateImage()
        self.update()

    def handleScaleBtn(self, btn):
        # Flags: 1-large/small, 2-up/down, 4-vert/hor
        if btn & 4:
            mode = 'vertical '
        else:
            mode = 'horizontal '
        if btn & 2:
            mode += 'expand'
        else:
            mode += 'shrink'
        image = self.image.copy()
        mask = self.mask.copy()

        if btn & 1:
            seam_count = 10
        else:
            seam_count = 1

        for i in range(seam_count):
            image, mask, _ = seam_carve(image, mode, mask=mask)
        self.n_rows, self.n_cols, _ = image.shape
        self.image = np.require(image, np.uint8, ['A', 'O', 'C'])
        if mask is None or not mask.shape == image.shape[:2]:
            self.mask = np.zeros([self.n_rows, self.n_cols], np.int8)
        else:
            self.mask = mask.astype(np.int8).copy()
        del image, mask
        self.updateImage()
        self.update()
        self.parent().alignToImage(self.image.shape)


class Gui(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self, cfgpath):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)

        self.configpath = cfgpath
        self.paint = Viewer(self)
        self.setupUi(self)

        self.loadButton.clicked.connect(self.loadImage)
        self.maskClearButton.clicked.connect(self.paint.clearMask)
        self.brushSizeSB.valueChanged.connect(self.brushSizeChange)
        btnlist = [self.horDownBtn, self.horDownLargeBtn, self.horUpBtn, self.horUpLargeBtn,
                   self.vertDownBtn, self.vertDownLargeBtn, self.vertUpBtn, self.vertUpLargeBtn]
        sigmap = QtCore.QSignalMapper(self)
        for i in range(len(btnlist)):
            self.connect(btnlist[i], QtCore.SIGNAL("clicked()"), sigmap, QtCore.SLOT("map()"))
            sigmap.setMapping(btnlist[i], i)
        self.connect(sigmap, QtCore.SIGNAL("mapped(int)"), self.paint.handleScaleBtn)

        self.alignToImage(self.paint.image.shape)
        self.brushsize = self.brushSizeSB.value()
        self.imagepath = ''

    def mouseMoveEvent(self, event):
        ex = event.x() - self.paint.x()
        ey = event.y() - self.paint.y()
        pos = QtCore.QPoint(ex, ey)
        if ex >= 0 and ex < self.paint.width() and \
                ey >= 0 and ey < self.paint.height():
            if event.buttons() & QtCore.Qt.LeftButton:
                if self.brushSaveRB.isChecked():
                    value = 1
                else:
                    value = -1
                self.paint.changeMask(pos, value, self.brushsize)
            elif event.buttons() & QtCore.Qt.RightButton:
                self.paint.changeMask(pos, 0, self.brushsize)
        return QtGui.QMainWindow.mouseMoveEvent(self, event)

    def mousePressEvent(self, event):
        return self.mouseMoveEvent(event)

    def alignToImage(self, shape):
        self.resize(shape[1] + self.controlFrame.width(), shape[0])
        self.paint.setGeometry(self.controlFrame.width(), 0, shape[1], shape[0])
        self.controlFrame.setGeometry(0, 0, self.controlFrame.width(), shape[0])

    def brushSizeChange(self):
        self.brushsize = self.brushSizeSB.value()

    def loadImage(self, filename=''):
        if type(filename) != str or filename == '':
            filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 
                                                         QtCore.QDir.currentPath())
        if filename != '':
            img = imread(filename)
            self.alignToImage(img.shape)
            fname,fext = os.path.splitext(filename)
            maskpath = fname+'_mask'+fext
            mask = None
            if os.path.isfile(maskpath):
                mask_img = imread(maskpath)
                if (np.array_equal(mask_img.shape[:2],img.shape[:2])):
                    mask = ((mask_img[:,:,0]!=0)*(-1) + (mask_img[:,:,1]!=0)).astype(np.int8)
                    
            self.paint.loadImage(img,mask)
            self.imagepath = filename

    def loadParams(self, params):
        if params[0] != '' and os.path.isfile(params[0]):
            self.loadImage(params[0])
        self.brushsize = params[1]
        self.brushSizeSB.setValue(params[1])

    def saveParams(self):
        params = (self.imagepath, self.brushsize)
        saveConfig(self.configpath, params)


def loadConfig(filename):
    with open(filename) as fhandle:
        for line in fhandle:
            if len(line) != 0 and line[0] != '#':
                if line[:6] == 'image=':
                    imgpath = line[6:].rstrip()
                elif line[:6] == 'brush=':
                    bsize = int(line[6:])
    return (imgpath, bsize)


def saveConfig(filename, params):
    with open(filename, 'w') as fhandle:
        print('image=%s\nbrush=%d' % params, file=fhandle)

app = QtGui.QApplication.instance()
if not app:
    app = QtGui.QApplication(sys.argv)
app.aboutToQuit.connect(app.deleteLater)
configpath = os.path.dirname(os.path.abspath(__file__)) + '/gui.config'
window = Gui(configpath)
app.aboutToQuit.connect(window.saveParams)
window.show()
if os.path.isfile(configpath):
    params = loadConfig(configpath)
    window.loadParams(params)
if len(sys.argv) > 1:
    window.loadImage(sys.argv[1])
app.exec_()
