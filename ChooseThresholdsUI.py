
import sys
import os
import cv2
import numpy as np
from PyQt4.QtCore import pyqtSignal, pyqtSlot, QObject, SIGNAL, Qt
from PyQt4.QtGui import QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QWidget, QMainWindow, QLabel, \
    QCheckBox, QSlider, QFileDialog, QImage, QPixmap, QSpinBox, QComboBox

import pipeline


class ChooseThresholdsUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.baseImages = []

        self.SobelXTh = [0,255]

        self.SobelYTh = [0,255]

        self.SobelMTh = [0,255]

        self.SobelATh = [0,255]

        self.initUI()


    @pyqtSlot()
    def openNewFileDialog(self, index):
        """ @brief slot with open new file dialog
        """
        directory = os.path.dirname(self.loadFileLine[index].text())
        fileName = QFileDialog.getOpenFileName(self, 'Choose image', directory, 'All files (*)')
        if isinstance(fileName, tuple):  # for PyQt5
            if fileName[0]:
                self.loadFileLine[index].setText(fileName[0])
        else:  # for PyQt4
            if fileName:
                self.loadFileLine[index].setText(fileName)

    @pyqtSlot()
    def loadImage(self):

        self.baseImages = []
        for i in range(0,4):
            if (i>0):
                if not self.loadFileCB[i-1].isChecked():
                    continue
            image = cv2.imread(self.loadFileLine[i].text())
            if image is not None:
                self.baseImages.append(image)
        if len(self.baseImages) !=0:
            self.updateImage()

    @pyqtSlot()
    def sliderChange(self):
        self.SobelXTh[0] = self.slider1SobelX.value()
        self.SobelXTh[1] = self.slider2SobelX.value()
        self.labelSobelX.setText("SobelX th: {:03d} {:03d}".format(self.SobelXTh[0], self.SobelXTh[1]))

        self.SobelYTh[0] = self.slider1SobelY.value()
        self.SobelYTh[1] = self.slider2SobelY.value()
        self.labelSobelY.setText("SobelY th: {:03d} {:03d}".format(self.SobelYTh[0], self.SobelYTh[1]))

        self.SobelMTh[0] = self.slider1SobelM.value()
        self.SobelMTh[1] = self.slider2SobelM.value()
        self.labelSobelM.setText("SobelM th: {:03d} {:03d}".format(self.SobelMTh[0], self.SobelMTh[1]))

        self.SobelATh[0] = self.slider1SobelA.value()*np.pi/510
        self.SobelATh[1] = self.slider2SobelA.value()*np.pi/510
        self.labelSobelA.setText("SobelA th: {:1.3f} {:1.3f}".format(self.SobelATh[0], self.SobelATh[1]))

        self.updateImage()

    def initUI(self):

        # Choose file dialog
        loadFileBtn = QPushButton("Load files:")
        loadFileBtn.clicked.connect(self.loadImage)

        self.loadFileLine = []
        loadFileDialogBtn = []
        self.loadFileCB = []
        for i in range(0,4):
            self.loadFileLine.append(QLineEdit(""))
            loadFileDialogBtn.append(QPushButton("..."))
            loadFileDialogBtn[i].setMaximumWidth(40)
            if (i>0):
                self.loadFileCB.append(QCheckBox())
                # self.loadFileCB[i-1].stateChanged.connect(self.updateImage)
        loadFileDialogBtn[0].clicked.connect(lambda: self.openNewFileDialog(0))
        loadFileDialogBtn[1].clicked.connect(lambda: self.openNewFileDialog(1))
        loadFileDialogBtn[2].clicked.connect(lambda: self.openNewFileDialog(2))
        loadFileDialogBtn[3].clicked.connect(lambda: self.openNewFileDialog(3))
        loadFileDialogBtn[1].clicked.connect(lambda: self.loadFileCB[0].setChecked(True))
        loadFileDialogBtn[2].clicked.connect(lambda: self.loadFileCB[1].setChecked(True))
        loadFileDialogBtn[3].clicked.connect(lambda: self.loadFileCB[2].setChecked(True))

        self.labelImage = QLabel()

        labelChannels = QLabel("Channels:")
        self.comboChannel0 = QComboBox()
        self.comboChannel0.addItems(['G', 'L', 'S'])
        self.comboChannel0.currentIndexChanged.connect(self.updateImage)
        self.comboChannel1 = QComboBox()
        self.comboChannel1.addItems(['G', 'L', 'S'])
        self.comboChannel1.currentIndexChanged.connect(self.updateImage)
        self.comboChannel2 = QComboBox()
        self.comboChannel2.addItems(['G', 'L', 'S'])
        self.comboChannel2.currentIndexChanged.connect(self.updateImage)

        labelKernels = QLabel("Kernels:")
        self.kernel0SB = QSpinBox()
        self.kernel0SB.setMinimum(1)
        self.kernel0SB.setMaximum(31)
        self.kernel0SB.setValue(15)
        self.kernel0SB.setSingleStep(2)
        self.kernel0SB.valueChanged.connect(self.updateImage)
        self.kernel1SB = QSpinBox()
        self.kernel1SB.setMinimum(1)
        self.kernel1SB.setMaximum(31)
        self.kernel1SB.setValue(15)
        self.kernel1SB.setSingleStep(2)
        self.kernel1SB.valueChanged.connect(self.updateImage)
        self.kernel2SB = QSpinBox()
        self.kernel2SB.setMinimum(1)
        self.kernel2SB.setMaximum(31)
        self.kernel2SB.setValue(15)
        self.kernel2SB.setSingleStep(2)
        self.kernel2SB.valueChanged.connect(self.updateImage)


        self.labelSobelX = QLabel()
        self.chboxSobelX0 = QCheckBox()
        self.chboxSobelX0.stateChanged.connect(self.updateImage)
        self.chboxSobelX1 = QCheckBox()
        self.chboxSobelX1.stateChanged.connect(self.updateImage)
        self.chboxSobelX2 = QCheckBox()
        self.chboxSobelX2.stateChanged.connect(self.updateImage)
        self.slider1SobelX = QSlider(Qt.Horizontal)
        self.slider1SobelX.setMinimum(0)
        self.slider1SobelX.setMaximum(255)
        self.slider1SobelX.setValue(20)
        self.slider1SobelX.valueChanged.connect(self.sliderChange)
        self.slider2SobelX = QSlider(Qt.Horizontal)
        self.slider2SobelX.setMinimum(0)
        self.slider2SobelX.setMaximum(255)
        self.slider2SobelX.setValue(100)
        self.slider2SobelX.valueChanged.connect(self.sliderChange)

        self.labelSobelY = QLabel()
        self.chboxSobelY0 = QCheckBox()
        self.chboxSobelY0.stateChanged.connect(self.updateImage)
        self.chboxSobelY1 = QCheckBox()
        self.chboxSobelY1.stateChanged.connect(self.updateImage)
        self.chboxSobelY2 = QCheckBox()
        self.chboxSobelY2.stateChanged.connect(self.updateImage)
        self.slider1SobelY = QSlider(Qt.Horizontal)
        self.slider1SobelY.setMinimum(0)
        self.slider1SobelY.setMaximum(255)
        self.slider1SobelY.setValue(20)
        self.slider1SobelY.valueChanged.connect(self.sliderChange)
        self.slider2SobelY = QSlider(Qt.Horizontal)
        self.slider2SobelY.setMinimum(0)
        self.slider2SobelY.setMaximum(255)
        self.slider2SobelY.setValue(100)
        self.slider2SobelY.valueChanged.connect(self.sliderChange)

        self.labelSobelM = QLabel()
        self.chboxSobelM0 = QCheckBox()
        self.chboxSobelM0.stateChanged.connect(self.updateImage)
        self.chboxSobelM1 = QCheckBox()
        self.chboxSobelM1.stateChanged.connect(self.updateImage)
        self.chboxSobelM2 = QCheckBox()
        self.chboxSobelM2.stateChanged.connect(self.updateImage)
        self.slider1SobelM = QSlider(Qt.Horizontal)
        self.slider1SobelM.setMinimum(0)
        self.slider1SobelM.setMaximum(255)
        self.slider1SobelM.setValue(30)
        self.slider1SobelM.valueChanged.connect(self.sliderChange)
        self.slider2SobelM = QSlider(Qt.Horizontal)
        self.slider2SobelM.setMinimum(0)
        self.slider2SobelM.setMaximum(255)
        self.slider2SobelM.setValue(100)
        self.slider2SobelM.valueChanged.connect(self.sliderChange)

        self.labelSobelA = QLabel()
        self.chboxSobelA0 = QCheckBox()
        self.chboxSobelA0.stateChanged.connect(self.updateImage)
        self.chboxSobelA1 = QCheckBox()
        self.chboxSobelA1.stateChanged.connect(self.updateImage)
        self.chboxSobelA2 = QCheckBox()
        self.chboxSobelA2.stateChanged.connect(self.updateImage)
        self.slider1SobelA = QSlider(Qt.Horizontal)
        self.slider1SobelA.setMinimum(0)
        self.slider1SobelA.setMaximum(255)
        self.slider1SobelA.setValue(int(0.7/np.pi*510))
        self.slider1SobelA.valueChanged.connect(self.sliderChange)
        self.slider2SobelA = QSlider(Qt.Horizontal)
        self.slider2SobelA.setMinimum(0)
        self.slider2SobelA.setMaximum(255)
        self.slider2SobelA.setValue(int(1.3/np.pi*510))
        self.slider2SobelA.valueChanged.connect(self.sliderChange)

        self.sliderChange()

        # Layouts
        layoutMain = QVBoxLayout()
        layoutMain2 = QHBoxLayout()
        layoutSettings = QVBoxLayout()
        layoutChannels = QHBoxLayout()
        layoutKernelSB = QHBoxLayout()
        layoutChboxSobelX = QHBoxLayout()
        layoutChboxSobelY = QHBoxLayout()
        layoutChboxSobelM = QHBoxLayout()
        layoutChboxSobelA = QHBoxLayout()


        layoutsChooseFile = []
        for i in range(0,4):
            layoutsChooseFile.append(QHBoxLayout())
            if i == 0:
                layoutsChooseFile[i].addWidget(loadFileBtn)
            else:
                layoutsChooseFile[i].addWidget(self.loadFileCB[i-1])
            layoutsChooseFile[i].addWidget(self.loadFileLine[i])
            layoutsChooseFile[i].addWidget(loadFileDialogBtn[i])
            layoutMain.addLayout(layoutsChooseFile[i])

        layoutMain.addLayout(layoutMain2, 1)


        layoutMain2.addWidget(self.labelImage, 1)
        layoutMain2.addLayout(layoutSettings)

        layoutSettings.addWidget(labelChannels)
        layoutSettings.addLayout(layoutChannels)
        layoutChannels.addWidget(self.comboChannel0)
        layoutChannels.addWidget(self.comboChannel1)
        layoutChannels.addWidget(self.comboChannel2)
        layoutSettings.addWidget(labelKernels)
        layoutSettings.addLayout(layoutKernelSB)
        layoutKernelSB.addWidget(self.kernel0SB)
        layoutKernelSB.addWidget(self.kernel1SB)
        layoutKernelSB.addWidget(self.kernel2SB)
        layoutSettings.addWidget(self.labelSobelX)
        layoutSettings.addLayout(layoutChboxSobelX)
        layoutChboxSobelX.addWidget(self.chboxSobelX0)
        layoutChboxSobelX.addWidget(self.chboxSobelX1)
        layoutChboxSobelX.addWidget(self.chboxSobelX2)
        layoutSettings.addWidget(self.slider1SobelX)
        layoutSettings.addWidget(self.slider2SobelX)
        layoutSettings.addWidget(self.labelSobelY)
        layoutSettings.addLayout(layoutChboxSobelY)
        layoutChboxSobelY.addWidget(self.chboxSobelY0)
        layoutChboxSobelY.addWidget(self.chboxSobelY1)
        layoutChboxSobelY.addWidget(self.chboxSobelY2)
        layoutSettings.addWidget(self.slider1SobelY)
        layoutSettings.addWidget(self.slider2SobelY)
        layoutSettings.addWidget(self.labelSobelM)
        layoutSettings.addLayout(layoutChboxSobelM)
        layoutChboxSobelM.addWidget(self.chboxSobelM0)
        layoutChboxSobelM.addWidget(self.chboxSobelM1)
        layoutChboxSobelM.addWidget(self.chboxSobelM2)
        layoutSettings.addWidget(self.slider1SobelM)
        layoutSettings.addWidget(self.slider2SobelM)
        layoutSettings.addWidget(self.labelSobelA)
        layoutSettings.addLayout(layoutChboxSobelA)
        layoutChboxSobelA.addWidget(self.chboxSobelA0)
        layoutChboxSobelA.addWidget(self.chboxSobelA1)
        layoutChboxSobelA.addWidget(self.chboxSobelA2)
        layoutSettings.addWidget(self.slider1SobelA)
        layoutSettings.addWidget(self.slider2SobelA)
        layoutSettings.addStretch(1)

        mainWidget = QWidget()
        mainWidget.setLayout(layoutMain)
        self.setCentralWidget(mainWidget)
        self.showMaximized()

    def selectChannel(self, combo, N):
        if combo.currentText() == 'G':
            im = cv2.cvtColor(self.baseImages[N], cv2.COLOR_BGR2GRAY)
        elif combo.currentText() == 'L':
            im = cv2.cvtColor(self.baseImages[N], cv2.COLOR_BGR2HLS)[:, :, 1]
        elif combo.currentText() == 'S':
            im = cv2.cvtColor(self.baseImages[N], cv2.COLOR_BGR2HLS)[:, :, 2]
        return im

    def updateOneImg(self, imN):

        changed = False

        if self.chboxSobelX0.isChecked() or self.chboxSobelY0.isChecked() or self.chboxSobelM0.isChecked() or \
                self.chboxSobelA0.isChecked():
            changed = True
            im = self.selectChannel(self.comboChannel0, imN)
            th = pipeline.Tresholds(im, kernel=self.kernel0SB.value())

            im0 = np.ones_like(im, dtype='uint8')
            if self.chboxSobelX0.isChecked():
                im0 = im0 & th.abs_sobel_thresh(orient='x', thresh=self.SobelXTh)
            if self.chboxSobelY0.isChecked():
                im0 = im0 & th.abs_sobel_thresh(orient='y', thresh=self.SobelYTh)
            if self.chboxSobelM0.isChecked():
                im0 = im0 & th.mag_thresh(thresh=self.SobelMTh)
            if self.chboxSobelA0.isChecked():
                im0 = im0 & th.dir_threshold(thresh=self.SobelATh)
        else:
            im0 = np.zeros_like(self.baseImages[imN][:,:,0], dtype='uint8')

        if self.chboxSobelX1.isChecked() or self.chboxSobelY1.isChecked() or self.chboxSobelM1.isChecked() or \
                self.chboxSobelA1.isChecked():
            changed = True
            im = self.selectChannel(self.comboChannel1, imN)
            th = pipeline.Tresholds(im, kernel=self.kernel1SB.value())

            im1 = np.ones_like(im, dtype='uint8')
            if self.chboxSobelX1.isChecked():
                im1 = im1 & th.abs_sobel_thresh(orient='x', thresh=self.SobelXTh)
            if self.chboxSobelY1.isChecked():
                im1 = im1 & th.abs_sobel_thresh(orient='y', thresh=self.SobelYTh)
            if self.chboxSobelM1.isChecked():
                im1 = im1 & th.mag_thresh(thresh=self.SobelMTh)
            if self.chboxSobelA1.isChecked():
                im1 = im1 & th.dir_threshold(thresh=self.SobelATh)
        else:
            im1 = np.zeros_like(self.baseImages[imN][:,:,0], dtype='uint8')

        if self.chboxSobelX2.isChecked() or self.chboxSobelY2.isChecked() or self.chboxSobelM2.isChecked() or \
                self.chboxSobelA2.isChecked():
            changed = True
            im = self.selectChannel(self.comboChannel2, imN)
            th = pipeline.Tresholds(im, kernel=self.kernel2SB.value())

            im2 = np.ones_like(im, dtype='uint8')
            if self.chboxSobelX2.isChecked():
                im2 = im2 & th.abs_sobel_thresh(orient='x', thresh=self.SobelXTh)
            if self.chboxSobelY2.isChecked():
                im2 = im2 & th.abs_sobel_thresh(orient='y', thresh=self.SobelYTh)
            if self.chboxSobelM2.isChecked():
                im2 = im2 & th.mag_thresh(thresh=self.SobelMTh)
            if self.chboxSobelA2.isChecked():
                im2 = im2 & th.dir_threshold(thresh=self.SobelATh)
        else:
            im2 = np.zeros_like(self.baseImages[imN][:,:,0], dtype='uint8')

        if not changed:
            imf = cv2.cvtColor(self.baseImages[imN], cv2.COLOR_BGR2RGB)
        else:
            imf = np.dstack((im0*255, im1*255, im2*255))

        return  imf

    def updateImage(self):

        if len(self.baseImages) == 0:
            return

        imf = []
        for i in range(len(self.baseImages)):
            im = self.updateOneImg(i)
            imf.append(im)


        frameSize = (self.labelImage.size().width(), self.labelImage.size().height(), 3)
        bytesPerLine = 3 * frameSize[0]

        if len(self.baseImages) == 1:
            frameImg = cv2.resize(imf[0], frameSize[0:2])
        else:
            frameImg = np.zeros([frameSize[1], frameSize[0], frameSize[2]], dtype='uint8')
            midy = frameSize[1]//2
            midx = frameSize[0]//2
            # frameImg[0:midy, 0:midx, :] = 255
            frameImg[:midy,:midx,:] = cv2.resize(imf[0], (midx, midy))
            frameImg[:midy,midx:midx*2,:] = cv2.resize(imf[1], (midx, midy))
            if len(self.baseImages) >= 3:
                frameImg[midy:midy*2, :midx, :] = cv2.resize(imf[2], (midx, midy))
            if len(self.baseImages) >= 4:
                frameImg[midy:midy*2, midx:midx*2, :] = cv2.resize(imf[3], (midx, midy))

        qImg = QImage(frameImg, frameSize[0], frameSize[1], bytesPerLine, QImage.Format_RGB888)
        self.labelImage.setPixmap(QPixmap.fromImage(qImg))


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ctUI = ChooseThresholdsUI()
    sys.exit(app.exec_())
