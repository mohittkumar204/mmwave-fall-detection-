"""Texas Instruments mmWave People Counting & Fall Detection GUI.

Integrated with Trained mmWave Fall-Detection & Posture Classification Models:
1. Primary: 9-Kinematic Hybrid 1D-CNN + LSTM Model (ONNX / PyTorch)
2. Fallback: Approach A Statistical 23-Feature Classifier (RandomForest / GradientBoosting)
3. 5-Class Posture Estimation: STANDING, SITTING, WALKING, FALLING, LYING

Preserves the original Texas Instruments mmWave People Counting Visualizer layout,
3D OpenGL point cloud, ellipsoids/bounding boxes, target statistics, and height plots.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import sys
import time
from collections import OrderedDict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
GUI_DIR = Path(__file__).resolve().parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import numpy as np
# Pre-import torch before PyQt5 to avoid Windows DLL initialization conflict on Python 3.13
try:
    import torch
except Exception:
    pass

from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QStyleFactory, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QColor, QFont, QPixmap
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from src.gui.oob_parser import uartParserSDK
from src.gui.gui_threads import *
from src.gui.graphUtilities import *
from src.gui.gl_classes import GLTextItem

# Import project inference engine
from src.inference.live_fall_detector import LiveFallDetector
from src.radar.packet_parser import ParsedFrame, Point, Target
from src.radar.serial_acquisition import RadarReplayAcquisition


class LoggerTee:
    """Safely mirrors stdout to both console and a logfile."""
    def __init__(self, filename: str | Path):
        self.terminal = sys.stdout
        try:
            self.log = open(filename, "a", encoding="utf-8", errors="replace")
        except Exception:
            self.log = None

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            pass
        if self.log:
            try:
                self.log.write(message)
                self.log.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        if self.log:
            try:
                self.log.flush()
            except Exception:
                pass


class Window(QDialog):
    def __init__(self, parent=None, size=None):
        super(Window, self).__init__(parent)
        self.Gree = 1

        self.setWindowFlags(
            Qt.Window |
            Qt.CustomizeWindowHint |
            Qt.WindowTitleHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("mmWave People Counting")

        # Set up safe logging directory
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.localtime()
        log_file = log_dir / f"ti_gui_logfile_{ts[2]}{ts[1]}{ts[0]}_{ts[3]}{ts[4]}.txt"
        sys.stdout = LoggerTee(log_file)

        print("Python is ", struct.calcsize("P") * 8, " bit")
        print("Python version: ", sys.version)

        self.frameTime = 55
        self.graphFin = 1
        self.hGraphFin = 1
        self.threeD = 1
        self.lastFramePoints = np.zeros((5, 1))
        self.plotTargets = 1
        self.frameNum = 0
        self.lastTID = []
        self.profile = {
            "startFreq": 60.25,
            "numLoops": 64,
            "numTx": 3,
            "sensorHeight": 1.5,
            "maxRange": 10,
            "az_tilt": 0,
            "elev_tilt": 0,
            "leftX": -3.0,
            "rightX": 3.0,
            "nearY": 0.5,
            "farY": 6.0,
            "bottomZ": -1.5,
            "topZ": 1.5,
        }
        self.lastFrameHadTargets = False
        self.sensorHeight = 1.5
        self.numFrameAvg = 20
        self.configSent = 0
        self.previousFirstZ = -1
        self.yzFlip = 0

        # Timer to reset fall alert
        self.fallTimer = QTimer()
        self.fallTimer.setSingleShot(True)
        self.fallTimer.timeout.connect(self.resetFallText)
        self.fallResetTimerOn = 0
        self.fallThresh = -0.22

        # Timer to monitor Telegram remote control commands (/fall, /upright)
        self.telegramTimer = QTimer()
        self.telegramTimer.timeout.connect(self._check_telegram_remote_override)
        self.telegramTimer.start(250)

        # Replay mode state
        self.replay_acquisition: RadarReplayAcquisition | None = None
        self.replay_timer = QTimer()
        self.replay_timer.timeout.connect(self._replay_step)
        self.replay_generator = None

        # Color gradients
        self.Gradients = OrderedDict([
            ('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
            ('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)), (0.11348314606741573, (0, 68, 255, 255)), (0.6797752808988764, (255, 255, 0, 255)), (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)), (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
            ('summer', {'ticks': [(1, (255, 255, 0, 255)), (0.0, (0, 170, 127, 255))], 'mode': 'rgb'}),
            ('space', {'ticks': [(0.562, (75, 215, 227, 255)), (0.087, (255, 170, 0, 254)), (0.332, (0, 255, 0, 255)), (0.77, (85, 0, 255, 255)), (0.0, (255, 0, 0, 255)), (1.0, (255, 0, 127, 255))], 'mode': 'rgb'}),
            ('winter', {'ticks': [(1, (0, 255, 127, 255)), (0.0, (0, 0, 255, 255))], 'mode': 'rgb'}),
            ('spectrum2', {'ticks': [(1.0, (255, 0, 0, 255)), (0.0, (255, 0, 255, 255))], 'mode': 'hsv'}),
        ])
        cmap = 'spectrum2'
        self.gradientMode = self.Gradients.get(cmap, self.Gradients['bw'])
        self.zRange = [-3, 3]
        self.plotHeights = 1

        if size:
            left = 40
            top = 40
            width = math.ceil(size.width() * 0.92)
            height = math.ceil(size.height() * 0.92)
            self.setGeometry(left, top, width, height)

        self.previousCloud = np.zeros((6, 1150, 10))
        self.previousPointCount = np.zeros((10, 1))

        # Resolve asset image paths
        images_dir = PROJECT_ROOT / "images"
        standing_path = images_dir / "stickFigureStanding.png"
        falling_path = images_dir / "stickFigureFalling.png"
        if not standing_path.exists():
            standing_path = GUI_DIR / "images" / "stickFigureStanding.png"
        if not falling_path.exists():
            falling_path = GUI_DIR / "images" / "stickFigureFalling.png"

        self.standingPicture = QPixmap(str(standing_path)) if standing_path.exists() else QPixmap()
        self.fallingPicture = QPixmap(str(falling_path)) if falling_path.exists() else QPixmap()
        self.bbox = [-1000, 1000, -1000, 1000, -1000, 1000]

        # Initialize trained Machine Learning Fall Detection Engine
        self.fall_detector = None
        self.ml_engine_name = "Rule Heuristic"
        self._init_trained_fall_detector()

        # Build original GUI widgets
        self.plot3DQTGraph()
        self.colorGradient()
        self.heightPlots()
        self.fallDetData()

        self.setConnectionLayout()
        self.setStatsLayout()
        self.setPlotControlLayout()
        self.setConfigLayout()
        self.setUpBoundaryBoxControls()
        self.setSensorPositionControls()

        # Tabs for graphing options
        self.graphTabs = QTabWidget()
        self.graphTabs.addTab(self.pcplot, '3D Plot')
        self.graphTabs.currentChanged.connect(self.whoVisible)

        gridlay = QGridLayout()
        gridlay.addWidget(self.comBox, 0, 0, 1, 1)
        gridlay.addWidget(self.statBox, 1, 0, 1, 1)
        gridlay.addWidget(self.configBox, 2, 0, 1, 1)
        gridlay.addWidget(self.plotControlBox, 3, 0, 1, 1)
        gridlay.addWidget(self.boxTab, 4, 0, 1, 1)
        gridlay.addWidget(self.spBox, 5, 0, 1, 1)
        gridlay.addWidget(self.graphTabs, 0, 1, 6, 1)
        gridlay.addWidget(self.gw, 0, 2, 6, 1)
        gridlay.addWidget(self.demoData, 0, 3, 1, 2)
        gridlay.addWidget(self.hPlot, 1, 3, 5, 2)
        gridlay.setColumnStretch(0, 1)
        gridlay.setColumnStretch(1, 3)
        self.setLayout(gridlay)

        # Preload default tuned configuration if available
        default_cfg = PROJECT_ROOT / "configs" / "ODS_6m_posture_tuned.cfg"
        if default_cfg.exists():
            try:
                self.parseCfg(str(default_cfg))
                print(f"[GUI] Preloaded configuration: {default_cfg.name}")
            except Exception as e:
                print(f"[GUI] Note loading default cfg: {e}")

    def _init_trained_fall_detector(self):
        """Load the trained machine-learning model engine."""
        try:
            # Check single-person mode from environment (defaults to True for residential rooms)
            single_person = os.getenv("RADAR_SINGLE_PERSON_MODE", "True").strip().lower() in ("true", "1", "yes")
            self.fall_detector = LiveFallDetector(
                metadata_path=PROJECT_ROOT / "artifacts" / "best_model_metadata.json",
                scaler_path=PROJECT_ROOT / "artifacts" / "scaler.joblib",
                model_path=PROJECT_ROOT / "models" / "best_fall_model.joblib",
                single_person_mode=single_person,
                min_snr=8.0,
                min_track_age=1,
            )
            if self.fall_detector.use_cnn_lstm:
                self.ml_engine_name = "Hybrid 1D-CNN+LSTM (Active)"
            elif self.fall_detector.model is not None:
                self.ml_engine_name = f"{self.fall_detector.metadata.get('best_model', 'RandomForest')} (Approach A)"
            else:
                self.ml_engine_name = "Kinematic Stabilizer"
            print(f"[GUI] Successfully loaded trained ML fall detector: {self.ml_engine_name} (SinglePersonMode: {single_person})")
        except Exception as e:
            print(f"[GUI] Warning initializing trained model: {e}")
            self.fall_detector = None
            self.ml_engine_name = "Rule Heuristic (Model Unavailable)"

    # ---------------------------------------------------------
    # LEFT CONTROLS PANE LAYOUT
    # ---------------------------------------------------------
    def setConnectionLayout(self):
        self.comBox = QGroupBox('Connect to Com Ports')
        self.uartCom = QLineEdit('11')
        self.dataCom = QLineEdit('14')
        self.uartLabel = QLabel('UART COM:')
        self.dataLabel = QLabel('DATA COM:')
        self.connectStatus = QLabel('Not Connected')
        self.connectButton = QPushButton('Connect')
        self.connectButton.clicked.connect(self.connectCom)
        self.configLabel = QLabel('Config Type:')
        self.configType = QComboBox()
        self.configType.addItems([
            "3D People Counting",
            "SDK Out of Box Demo",
            "Long Range People Detection",
            "Sense and Detect HVAC Control",
            "(Legacy) 2D People Counting",
            "(Legacy): Overhead People Counting",
            "Capon3DAOP",
            "Replay",
        ])
        self.comLayout = QGridLayout()
        self.comLayout.addWidget(self.uartLabel, 0, 0)
        self.comLayout.addWidget(self.uartCom, 0, 1)
        self.comLayout.addWidget(self.dataLabel, 1, 0)
        self.comLayout.addWidget(self.dataCom, 1, 1)
        self.comLayout.addWidget(self.configLabel, 2, 0)
        self.comLayout.addWidget(self.configType, 2, 1)
        self.comLayout.addWidget(self.connectButton, 3, 0)
        self.comLayout.addWidget(self.connectStatus, 3, 1)
        self.comBox.setLayout(self.comLayout)

    def setStatsLayout(self):
        self.statBox = QGroupBox('Statistics')
        self.frameNumDisplay = QLabel('Frame: 0')
        self.plotTimeDisplay = QLabel('Average Plot Time: 0 ms')
        self.numPointsDisplay = QLabel('Points: 0')
        self.numTargetsDisplay = QLabel('Targets: 0')
        self.statsLayout = QVBoxLayout()
        self.statsLayout.addWidget(self.frameNumDisplay)
        self.statsLayout.addWidget(self.plotTimeDisplay)
        self.statsLayout.addWidget(self.numPointsDisplay)
        self.statsLayout.addWidget(self.numTargetsDisplay)
        self.statBox.setLayout(self.statsLayout)

    def setPlotControlLayout(self):
        self.plotControlBox = QGroupBox('Plot Controls')
        self.staticclutter = QCheckBox('Display Static Points')
        self.plotByIndex = QCheckBox('Plot Point Color by Index')
        self.plotByHeight = QCheckBox('Plot Point Color By Height')
        self.plotTracks = QCheckBox('Plot Tracks')
        self.pointColorGroup = QButtonGroup()
        self.pointColorGroup.addButton(self.plotByIndex)
        self.pointColorGroup.addButton(self.plotByHeight)
        self.pointColorGroup.setExclusive(True)
        self.persistentFramesInput = QComboBox()
        self.persistentFramesInput.addItems(['1','2','3','4','5','6','7','8','9','10'])
        self.persistentFramesInput.setCurrentIndex(2)
        self.pFILabel = QLabel('# of Persistent Frames')
        self.orientationSelection = QComboBox()
        self.orientationSelection.addItems(['Side Mount', 'Overhead Mount'])
        self.orientationSelection.currentIndexChanged.connect(self.swapOrientations)
        self.oriLabel = QLabel('Orientation')
        self.fallThreshInput = QLineEdit(str(self.fallThresh))
        self.fallThreshInput.textEdited.connect(self.updateFallThresh)
        self.fallTLabel = QLabel('Fall Detection Threshold')
        self.plotControlLayout = QGridLayout()
        self.plotControlLayout.addWidget(self.plotByIndex, 0, 0, 1, 1)
        self.plotControlLayout.addWidget(self.plotByHeight, 1, 0, 1, 1)
        self.plotControlLayout.addWidget(self.plotTracks, 2, 0, 1, 1)
        self.plotControlLayout.addWidget(self.persistentFramesInput, 4, 0, 1, 1)
        self.plotControlLayout.addWidget(self.pFILabel, 4, 1, 1, 1)
        self.plotControlLayout.addWidget(self.staticclutter, 3, 0, 1, 1)
        self.plotControlLayout.addWidget(self.orientationSelection, 5, 0, 1, 1)
        self.plotControlLayout.addWidget(self.oriLabel, 5, 1, 1, 1)
        self.plotControlLayout.addWidget(self.fallThreshInput, 6, 0, 1, 1)
        self.plotControlLayout.addWidget(self.fallTLabel, 6, 1, 1, 1)
        self.plotControlBox.setLayout(self.plotControlLayout)
        self.plotByHeight.setChecked(True)
        self.plotByIndex.setChecked(False)
        self.plotTracks.setChecked(True)

    def setConfigLayout(self):
        self.configBox = QGroupBox('Configuration')
        self.selectConfig = QPushButton('Select Configuration')
        self.sendConfig = QPushButton('Send Configuration')
        self.selectConfig.clicked.connect(self.selectCfg)
        self.sendConfig.clicked.connect(self.sendCfg)
        self.configTable = QTableWidget(5, 2)
        self.configTable.setItem(0, 0, QTableWidgetItem('Radar Parameter'))
        self.configTable.setItem(0, 1, QTableWidgetItem('Value'))
        self.configTable.setItem(1, 0, QTableWidgetItem('Max Range'))
        self.configTable.setItem(2, 0, QTableWidgetItem('Range Resolution'))
        self.configTable.setItem(3, 0, QTableWidgetItem('Max Velocity'))
        self.configTable.setItem(4, 0, QTableWidgetItem('Velocity Resolution'))
        self.configLayout = QVBoxLayout()
        self.configLayout.addWidget(self.selectConfig)
        self.configLayout.addWidget(self.sendConfig)
        self.configLayout.addWidget(self.configTable)
        self.configBox.setLayout(self.configLayout)

    def setBoxControlLayout(self, name):
        boxControl = QGroupBox(name)
        lx = QLineEdit('-3')
        rx = QLineEdit('3')
        ny = QLineEdit('0.5')
        fy = QLineEdit('6')
        bz = QLineEdit('-1.5')
        tz = QLineEdit('1.5')
        enable = QCheckBox()
        lxL = QLabel('Left X')
        rxL = QLabel('Right X')
        nyL = QLabel('Near Y')
        fyL = QLabel('Far Y')
        bzL = QLabel('Bottom Z')
        tzL = QLabel('Top Z')
        enableL = QLabel('Enable Box')
        boxConLayout = QGridLayout()
        boxConLayout.addWidget(lxL, 0, 0, 1, 1)
        boxConLayout.addWidget(lx, 0, 1, 1, 1)
        boxConLayout.addWidget(rxL, 0, 2, 1, 1)
        boxConLayout.addWidget(rx, 0, 3, 1, 1)
        boxConLayout.addWidget(nyL, 1, 0, 1, 1)
        boxConLayout.addWidget(ny, 1, 1, 1, 1)
        boxConLayout.addWidget(fyL, 1, 2, 1, 1)
        boxConLayout.addWidget(fy, 1, 3, 1, 1)
        boxConLayout.addWidget(bzL, 2, 0, 1, 1)
        boxConLayout.addWidget(bz, 2, 1, 1, 1)
        boxConLayout.addWidget(tzL, 2, 2, 1, 1)
        boxConLayout.addWidget(tz, 2, 3, 1, 1)
        boxConLayout.addWidget(enableL, 3, 0, 1, 1)
        boxConLayout.addWidget(enable, 3, 1, 1, 1)
        boxControl.setLayout(boxConLayout)
        boundList = [lx, rx, ny, fy, bz, tz]
        for text in boundList:
            text.textEdited.connect(self.changeBoundaryBox)
        enable.stateChanged.connect(self.changeBoundaryBox)
        return {'boxCon': boxControl, 'boundList': boundList, 'checkEnable': enable, 'boxNum': -1}

    def setSensorPositionControls(self):
        self.az_tilt = QLineEdit('0')
        self.az_tiltL = QLabel('Azimuth Tilt')
        self.elev_tilt = QLineEdit('0')
        self.elev_tiltL = QLabel('Elevation Tilt')
        self.s_height = QLineEdit(str(self.profile['sensorHeight']))
        self.s_heightL = QLabel('Sensor Height')
        self.spLayout = QGridLayout()
        self.spLayout.addWidget(self.az_tilt, 0, 1, 1, 1)
        self.spLayout.addWidget(self.az_tiltL, 0, 0, 1, 1)
        self.spLayout.addWidget(self.elev_tilt, 1, 1, 1, 1)
        self.spLayout.addWidget(self.elev_tiltL, 1, 0, 1, 1)
        self.spLayout.addWidget(self.s_height, 2, 1, 1, 1)
        self.spLayout.addWidget(self.s_heightL, 2, 0, 1, 1)
        self.spBox = QGroupBox('Sensor Position')
        self.spBox.setLayout(self.spLayout)
        self.s_height.textEdited.connect(self.updateSensorPosition)
        self.az_tilt.textEdited.connect(self.updateSensorPosition)
        self.elev_tilt.textEdited.connect(self.updateSensorPosition)

    def updateSensorPosition(self):
        try:
            float(self.s_height.text())
            float(self.az_tilt.text())
            float(self.elev_tilt.text())
        except Exception:
            return
        command = f"sensorPosition {self.s_height.text()} {self.az_tilt.text()} {self.elev_tilt.text()} \n"
        if hasattr(self, "parser") and hasattr(self.parser, "sendLine"):
            try:
                self.cThread = sendCommandThread(self.parser, command)
                self.cThread.start(priority=QThread.HighestPriority - 2)
            except Exception:
                pass
        self.gz.translate(dx=0, dy=0, dz=self.profile['sensorHeight'])
        self.profile['sensorHeight'] = float(self.s_height.text())
        self.gz.translate(dx=0, dy=0, dz=-self.profile['sensorHeight'])

    def setUpBoundaryBoxControls(self):
        self.boundaryBoxes = []
        self.boxTab = QTabWidget()
        for i in range(2):
            name = f'Box{i}'
            self.boundaryBoxes.append(self.setBoxControlLayout(name))
            toAdd = self.boundaryBoxes[i]
            toAdd['boxNum'] = i
            self.boxTab.addTab(toAdd['boxCon'], name)
        self.boundaryBoxes[0]['checkEnable'].setChecked(True)

    def changeBoundaryBox(self):
        numBoxes = 0
        for box in self.boundaryBoxes:
            if box['checkEnable'].isChecked():
                numBoxes += 1
        boundaryString = f"LiveScenery {numBoxes} "
        for box in self.boundaryBoxes:
            if box['checkEnable'].isChecked():
                for text in box['boundList']:
                    val = text.text().replace(" ", "")
                    try:
                        float(val)
                    except Exception:
                        return
                    boundaryString += text.text() + " "
                self.drawBoundaryBox3d(box['boxNum'])
            else:
                self.boundaryBoxViz[box['boxNum']].setVisible(False)
                self.bottomSquare[box['boxNum']].setVisible(False)
        boundaryString += "\n"
        if self.configSent and hasattr(self, "parser") and hasattr(self.parser, "sendLine"):
            try:
                self.cThread = sendCommandThread(self.parser, boundaryString)
                self.cThread.start(priority=QThread.HighestPriority - 2)
            except Exception:
                pass

    def setBoundaryTextVals(self, profile):
        for box in self.boundaryBoxes:
            bList = box['boundList']
            bList[0].setText(str(profile.get('leftX', -3.0)))
            bList[1].setText(str(profile.get('rightX', 3.0)))
            bList[2].setText(str(profile.get('nearY', 0.5)))
            bList[3].setText(str(profile.get('farY', 6.0)))
            bList[4].setText(str(profile.get('bottomZ', -1.5)))
            bList[5].setText(str(profile.get('topZ', 1.5)))

    def drawBoundaryBox3d(self, index):
        bList = self.boundaryBoxes[index]['boundList']
        try:
            xl = float(bList[0].text())
            xr = float(bList[1].text())
            yl = float(bList[2].text())
            yr = float(bList[3].text())
            zl = float(bList[4].text()) - self.profile['sensorHeight']
            zr = float(bList[5].text()) - self.profile['sensorHeight']
        except Exception:
            return
        self.bbox = [xl, xr, yl, yr, zl, zr]
        boxLines = getBoxLines(xl, yl, zl, xr, yr, zr)
        squareLine = getSquareLines(xl, yl, xr, yr, zl)
        if not self.boundaryBoxViz[index].visible():
            self.boundaryBoxViz[index].setVisible(True)
            self.bottomSquare[index].setVisible(True)
        self.boundaryBoxViz[index].setData(pos=boxLines, color=pg.glColor('r'), width=2, antialias=True, mode='lines')
        self.bottomSquare[index].setData(pos=squareLine, color=pg.glColor('b'), width=2, antialias=True, mode='line_strip')

    def colorGradient(self):
        self.gw = pg.GradientWidget(orientation='right')
        self.gw.restoreState(self.gradientMode)

    def swapOrientations(self):
        if self.orientationSelection.currentText() == 'Overhead Mount':
            self.yzFlip = 1
            self.evmBox.rotate(90, 1, 0, 0)
            self.gz.resetTransform()
            self.gz.setSize(x=8, y=8)
            self.gz.translate(dx=0, dy=0, dz=-self.profile['sensorHeight'])
            self.changeBoundaryBox()
            self.zRange = [-1 * self.profile['sensorHeight'], 0]
        else:
            self.yzFlip = 0
            self.evmBox.rotate(-90, 1, 0, 0)
            self.gz.resetTransform()
            self.gz.setSize(x=self.profile['maxRange'] * 2, y=self.profile['maxRange'])
            self.gz.translate(dx=0, dy=self.profile['maxRange'] / 2, dz=-self.profile['sensorHeight'])
            self.changeBoundaryBox()
            self.zRange = [-3, 3]

    # ---------------------------------------------------------
    # RIGHT SIDE: FALL DETECTION STATUS & HEIGHT PLOTS
    # ---------------------------------------------------------
    def fallDetData(self):
        """Construct the original Fall Detection Demo pane with model badges."""
        self.demoData = QGroupBox('Fall Detection Demo')
        self.demoName = QLabel('Fall Detection Demo')
        self.demoName.setStyleSheet("font-weight: bold; font-size: 11pt;")

        self.numDetPeople = QLabel('Number of Detected People: 0')
        self.numDetPeople.setStyleSheet("font-size: 10pt;")

        self.fallDetEnabled = QLabel('Fall Detection Disabled - No People Detected')
        self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #555555;")

        self.mlModelStatus = QLabel(f'Engine: {self.ml_engine_name}')
        self.mlModelStatus.setStyleSheet("font-size: 9pt; color: #0277BD; font-weight: bold;")

        self.confidenceLabel = QLabel('Confidence: --')
        self.confidenceLabel.setStyleSheet("font-size: 9pt; color: #333333;")

        self.latencyLabel = QLabel('Inference Latency: --')
        self.latencyLabel.setStyleSheet("font-size: 9pt; color: #333333;")

        self.fallAlert = QLabel('Upright')
        self.fallAlert.setAlignment(Qt.AlignCenter)
        self.fallAlert.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E7D32; padding: 4px;")

        self.fallPic = QLabel()
        self.fallPic.setAlignment(Qt.AlignCenter)
        if not self.standingPicture.isNull():
            self.fallPic.setPixmap(self.standingPicture)

        self.alertStatusLabel = QLabel('Alerts: ESP32 Buzzer (Auto) | Telegram (Armed)')
        self.alertStatusLabel.setStyleSheet("font-size: 8.5pt; color: #555555;")

        self.ackButton = QPushButton('Acknowledge / Silence')
        self.ackButton.setStyleSheet("font-size: 8.5pt; padding: 2px 6px;")
        self.ackButton.clicked.connect(self.acknowledgeAlert)

        layoutDD = QGridLayout()
        layoutDD.addWidget(self.demoName, 0, 0, 1, 2)
        layoutDD.addWidget(self.numDetPeople, 1, 0)
        layoutDD.addWidget(self.fallAlert, 1, 1)
        layoutDD.addWidget(self.fallDetEnabled, 2, 0)
        layoutDD.addWidget(self.fallPic, 2, 1, 3, 1)
        layoutDD.addWidget(self.mlModelStatus, 3, 0)
        layoutDD.addWidget(self.confidenceLabel, 4, 0)
        layoutDD.addWidget(self.latencyLabel, 5, 0)
        layoutDD.addWidget(self.alertStatusLabel, 6, 0)
        layoutDD.addWidget(self.ackButton, 6, 1)

        self.demoData.setLayout(layoutDD)
        self._update_alert_channel_status()

    def heightPlots(self):
        """Construct the original 3 Height Plots (Instantaneous, Average, Delta)."""
        self.heightPlotsDict = {'instant': [], 'mean': [], 'deriv': []}
        yMin = -3
        yMax = 3
        # 100 frames of data, 256 tracks (supports full TI tracker TID range 0..255)
        self.targetSize = np.zeros((7, 256, 100))
        self.hPlot = QGroupBox('Height Plots')

        # Instantaneous Height
        self.iHPlot = pg.PlotWidget()
        self.iHPlot.setYRange(yMin, yMax, padding=0.1)
        self.iHPlot.setTitle('Instantaneous Height')
        self.iHPlot.showGrid(x=False, y=True)
        self.iHPlot.setBackground('w')
        self.iHPlot.setMouseEnabled(False, False)
        self.iHPlot.hideAxis('bottom')
        self.heightPlotsDict['instant'] = pg.PlotCurveItem(pen=pg.mkPen(width=1.5, color='b'))
        self.iHPlot.addItem(self.heightPlotsDict['instant'])

        # Average Height
        self.mHPlot = pg.PlotWidget()
        self.mHPlot.setTitle('Average Height')
        self.mHPlot.setYRange(yMin, yMax, padding=0.1)
        self.mHPlot.showGrid(x=False, y=True)
        self.mHPlot.setBackground('w')
        self.mHPlot.setMouseEnabled(False, False)
        self.mHPlot.hideAxis('bottom')
        self.heightPlotsDict['mean'] = pg.PlotCurveItem(pen=pg.mkPen(width=1.5, color='b'))
        self.mHPlot.addItem(self.heightPlotsDict['mean'])

        # Delta Height
        self.dHPlot = pg.PlotWidget()
        self.dHPlot.setTitle('Delta Height')
        self.dHPlot.setYRange(-1.5, 1.5, padding=0.1)
        self.dHPlot.showGrid(x=False, y=True)
        self.dHPlot.setBackground('w')
        self.dHPlot.setMouseEnabled(False, False)
        self.dHPlot.hideAxis('bottom')
        self.heightPlotsDict['deriv'] = pg.PlotCurveItem(pen=pg.mkPen(width=1.5, color='b'))
        self.dHPlot.addItem(self.heightPlotsDict['deriv'])

        self.fallThreshMarker = pg.InfiniteLine(pos=self.fallThresh, angle=0, pen=pg.mkPen(width=1.5, color='r', style=Qt.DashLine))
        self.dHPlot.addItem(self.fallThreshMarker)

        self.heightLayout = QVBoxLayout()
        self.heightLayout.addWidget(self.iHPlot)
        self.heightLayout.addWidget(self.mHPlot)
        self.heightLayout.addWidget(self.dHPlot)
        self.hPlot.setLayout(self.heightLayout)

    # ---------------------------------------------------------
    # 3D POINT CLOUD & TRACKING VISUALIZATION
    # ---------------------------------------------------------
    def plot3DQTGraph(self):
        self.pcplot = gl.GLViewWidget()
        dummy = np.zeros((1, 3))

        self.gz = gl.GLGridItem()
        self.gz.translate(0, 0, -1 * self.profile['sensorHeight'])
        self.boundaryBoxViz = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
        self.bottomSquare = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
        for box in self.boundaryBoxViz:
            box.setVisible(False)
        self.scatter = gl.GLScatterPlotItem(size=5)
        self.scatter.setData(pos=dummy)
        self.pcplot.addItem(self.gz)
        self.pcplot.addItem(self.boundaryBoxViz[0])
        self.pcplot.addItem(self.boundaryBoxViz[1])
        self.pcplot.addItem(self.bottomSquare[0])
        self.pcplot.addItem(self.bottomSquare[1])
        self.pcplot.addItem(self.scatter)

        # Device representative box
        verX = 0.0625
        verY = 0.05
        verZ = 0.125
        verts = np.empty((2, 3, 3))
        verts[0, 0, :] = [-verX, 0, verZ]
        verts[0, 1, :] = [-verX, 0, -verZ]
        verts[0, 2, :] = [verX, 0, -verZ]
        verts[1, 0, :] = [-verX, 0, verZ]
        verts[1, 1, :] = [verX, 0, verZ]
        verts[1, 2, :] = [verX, 0, -verZ]
        self.evmBox = gl.GLMeshItem(vertexes=verts, smooth=False, drawEdges=True, edgeColor=pg.glColor('r'), drawFaces=False)
        self.pcplot.addItem(self.evmBox)

        self.coordStr = []
        self.ellipsoids = []
        for m in range(20):
            mesh = gl.GLLinePlotItem()
            mesh.setVisible(False)
            self.pcplot.addItem(mesh)
            self.ellipsoids.append(mesh)

            text = GLTextItem()
            text.setGLViewWidget(self.pcplot)
            text.setVisible(False)
            self.pcplot.addItem(text)
            self.coordStr.append(text)

    # ---------------------------------------------------------
    # MAIN DATA & INFERENCE PIPELINE
    # ---------------------------------------------------------
    def updateGraph(self, parsedData):
        if not parsedData:
            return

        pointCloud = parsedData[0]
        targets = parsedData[1]
        indexes = parsedData[2]
        numPoints = parsedData[3]
        numTargets = parsedData[4]
        self.frameNum = parsedData[5]
        fail = parsedData[6]
        classifierOutput = parsedData[7] if len(parsedData) > 7 else []

        if fail:
            return

        fallDetEn = 0
        rotTargetDataX, rotTargetDataY, rotTargetDataZ = rotX(
            targets[1], targets[2], targets[3], -1 * self.profile['elev_tilt']
        )
        targets[1] = rotTargetDataX
        targets[2] = rotTargetDataY
        targets[3] = rotTargetDataZ

        for i in range(numPoints):
            rotPointDataX, rotPointDataY, rotPointDataZ = rotX(
                [pointCloud[0, i]], [pointCloud[1, i]], [pointCloud[2, i]], -1 * self.profile['elev_tilt']
            )
            pointCloud[0, i] = float(np.asarray(rotPointDataX).item())
            pointCloud[1, i] = float(np.asarray(rotPointDataY).item())
            pointCloud[2, i] = float(np.asarray(rotPointDataZ).item())

        # Update basic counts
        self.numPointsDisplay.setText(f'Points: {numPoints}')
        self.numTargetsDisplay.setText(f'Targets: {numTargets}')
        self.numDetPeople.setText(f'Number of Detected People: {numTargets}')

        forced = self.fall_detector.get_forced_state() if self.fall_detector else None
        if numTargets == 0:
            if forced == "FALL":
                self.fallAlert.setText('FALLEN!')
                self.fallAlert.setStyleSheet(
                    "font-size: 16pt; font-weight: bold; color: red; "
                    "background-color: #ffcccc; border: 2px solid red; "
                    "border-radius: 4px; padding: 4px;"
                )
                if not self.fallingPicture.isNull():
                    self.fallPic.setPixmap(self.fallingPicture)
                self.fallDetEnabled.setText('🚨 FALL OVERRIDE (Telegram Command /fall)')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #D32F2F; font-weight: bold;")
                self.confidenceLabel.setText('Confidence: 100.0% (Remote)')
                self.latencyLabel.setText('Inference Latency: 0.0 ms')
                if getattr(self.fall_detector, "buzzer", None):
                    self.fall_detector.buzzer.trigger_fall_alarm()
            else:
                fdestr = 'Fall Detection Disabled - No People Detected'
                self.fallDetEnabled.setText(fdestr)
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #555555;")
                self.confidenceLabel.setText('Confidence: --')
                self.latencyLabel.setText('Inference Latency: --')
                self.fallAlert.setText('Upright')
                self.fallAlert.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E7D32; padding: 4px;")
                if not self.standingPicture.isNull():
                    self.fallPic.setPixmap(self.standingPicture)
                if getattr(self.fall_detector, "buzzer", None):
                    self.fall_detector.buzzer.clear_alarm()
        else:
            fallDetEn = 1

        if len(targets) < 13:
            targets = []

        # =========================================================
        # RUN TRAINED MACHINE-LEARNING MODEL INFERENCE
        # =========================================================
        ml_predictions: list[dict[str, Any]] = []
        if self.fall_detector is not None and numTargets > 0:
            try:
                frame_targets = []
                for t_idx in range(numTargets):
                    col = t_idx if targets.shape[1] > t_idx else 0
                    tid = int(targets[0, col])
                    px = float(targets[1, col])
                    py = float(targets[2, col])
                    pz = float(targets[3, col])
                    vx = float(targets[4, col]) if targets.shape[0] > 4 else 0.0
                    vy = float(targets[5, col]) if targets.shape[0] > 5 else 0.0
                    vz = float(targets[6, col]) if targets.shape[0] > 6 else 0.0
                    ax = float(targets[10, col]) if targets.shape[0] > 10 else 0.0
                    ay = float(targets[11, col]) if targets.shape[0] > 11 else 0.0
                    az = float(targets[12, col]) if targets.shape[0] > 12 else 0.0
                    frame_targets.append(Target(
                        tid=tid, pos_x=px, pos_y=py, pos_z=pz,
                        vel_x=vx, vel_y=vy, vel_z=vz,
                        acc_x=ax, acc_y=ay, acc_z=az
                    ))

                frame_points = []
                for p_idx in range(numPoints):
                    assoc_tid = int(indexes[p_idx]) if p_idx < len(indexes) else 255
                    px = float(pointCloud[0, p_idx])
                    py = float(pointCloud[1, p_idx])
                    pz = float(pointCloud[2, p_idx])
                    dop = float(pointCloud[3, p_idx]) if pointCloud.shape[0] > 3 else 0.0
                    snr = float(pointCloud[4, p_idx]) if pointCloud.shape[0] > 4 else 15.0
                    rng = float(math.sqrt(px**2 + py**2 + pz**2))
                    frame_points.append(Point(
                        point_index=p_idx, elevation=0.0, azimuth=0.0,
                        doppler=dop, range_val=rng, snr=snr,
                        x=px, y=py, z=pz, track_index=assoc_tid
                    ))

                parsed_frame = ParsedFrame(frame_num=self.frameNum, targets=frame_targets, points=frame_points)
                ml_predictions = self.fall_detector.process_frame(parsed_frame)

                # Synchronize targets array with filtered targets to eliminate ghost bounding boxes
                if parsed_frame.targets and len(parsed_frame.targets) < numTargets:
                    kept_tids = {t.tid for t in parsed_frame.targets}
                    col_indices = [c for c in range(targets.shape[1]) if int(targets[0, c]) in kept_tids]
                    if col_indices:
                        targets = targets[:, col_indices]
                        numTargets = len(col_indices)
            except Exception as e:
                print(f"[GUI] Error in ML inference: {e}")

        # Update Fall Detection UI from model predictions
        if ml_predictions:
            falling_target = next((r for r in ml_predictions if r.get("is_fall_instant") or r.get("active_alert")), None)
            any_fall = (falling_target is not None)
            primary = falling_target if falling_target is not None else ml_predictions[0]
            p_tid = primary.get("tid", 0)
            p_posture = primary.get("posture", "UPRIGHT")
            p_conf = primary.get("confidence", 0.95)
            p_lat = primary.get("latency_ms", 1.0)
            p_desc = primary.get("motion_label", p_posture)

            # Display verified human count
            if getattr(self.fall_detector, "noise_filter", None) and self.fall_detector.noise_filter.single_person_mode:
                self.numDetPeople.setText(f'Number of Detected People: {len(ml_predictions)}')
                self.numTargetsDisplay.setText(f'Targets: {len(ml_predictions)}')

            self.latencyLabel.setText(f'Inference Latency: {p_lat:.1f} ms')
            self.confidenceLabel.setText(f'Confidence: {p_conf * 100:.1f}%')

            if any_fall:
                self.fallAlert.setText('FALLEN!')
                self.fallAlert.setStyleSheet(
                    "font-size: 16pt; font-weight: bold; color: red; "
                    "background-color: #ffcccc; border: 2px solid red; "
                    "border-radius: 4px; padding: 4px;"
                )
                if not self.fallingPicture.isNull():
                    self.fallPic.setPixmap(self.fallingPicture)
                self.fallDetEnabled.setText(f'⚠️ FALL DETECTED (TID: {p_tid}) | High Drop Kinematics')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #D32F2F; font-weight: bold;")
                if self.fallResetTimerOn == 0:
                    self.fallResetTimerOn = 1
                    self.fallTimer.start(5000)
            elif p_posture == "BUFFERING":
                self.fallAlert.setText('Buffering...')
                self.fallAlert.setStyleSheet("font-size: 13pt; font-weight: bold; color: #1976D2; padding: 4px;")
                self.fallDetEnabled.setText('Status: Calibrating 8-frame temporal window...')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #1976D2;")
                if not self.standingPicture.isNull():
                    self.fallPic.setPixmap(self.standingPicture)
            elif p_posture == "LYING":
                self.fallAlert.setText('Lying')
                self.fallAlert.setStyleSheet("font-size: 14pt; font-weight: bold; color: #E65100; padding: 4px;")
                if not self.standingPicture.isNull():
                    self.fallPic.setPixmap(self.standingPicture)
                self.fallDetEnabled.setText(f'Model: LYING (TID: {p_tid}) | {p_desc}')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #E65100;")
            else:
                # UPRIGHT posture
                self.fallAlert.setText('Upright')
                self.fallAlert.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E7D32; padding: 4px;")
                if not self.standingPicture.isNull():
                    self.fallPic.setPixmap(self.standingPicture)
                self.fallDetEnabled.setText(f'Model: UPRIGHT (TID: {p_tid}) | {p_desc}')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #2E7D32;")

        # Mounting position flip
        if self.yzFlip == 1:
            pointCloud[[1, 2]] = pointCloud[[2, 1]]
            pointCloud[2, :] = -1 * pointCloud[2, :]
            if len(targets) > 0:
                targets[[2, 3]] = targets[[3, 2]]
                targets[3, :] = -1 * targets[3, :]

        # Clutter removal
        if self.configType.currentText() in ('3D People Counting', 'Capon3DAOP', 'Sense and Detect HVAC Control'):
            if not self.staticclutter.isChecked() and pointCloud.shape[0] > 3:
                statics = np.where(pointCloud[3, :] == 0)
                try:
                    firstZ = statics[0][0]
                    numPoints = firstZ
                    pointCloud = pointCloud[:, :firstZ]
                    indexes = indexes[:self.previousFirstZ]
                    self.previousFirstZ = firstZ
                except Exception:
                    pass

        # Point cloud persistence
        fNum = self.frameNum % 10
        if numPoints:
            self.previousCloud[:5, :numPoints, fNum] = pointCloud[:5, :numPoints]
            if len(indexes) > 0:
                self.previousCloud[5, :len(indexes), fNum] = indexes[:numPoints]
        self.previousPointCount[fNum] = numPoints

        # 3D Plot Threading
        if self.configType.currentText() == 'SDK3xPeopleCount':
            pointIn = pointCloud
            indicesIn = indexes
        else:
            persistentFrames = int(self.persistentFramesInput.currentText())
            totalPoints = sum(int(np.asarray(self.previousPointCount[(fNum - i) % 10]).item()) for i in range(1, persistentFrames + 1))
            pointIn = np.zeros((5, int(totalPoints)))
            indicesIn = np.ones((1, int(totalPoints))) * 255
            cur_pts = 0
            for i in range(1, persistentFrames + 1):
                prevCount = int(np.asarray(self.previousPointCount[(fNum - i) % 10]).item())
                pointIn[:, cur_pts:cur_pts + prevCount] = self.previousCloud[:5, :prevCount, (fNum - i) % 10]
                if numTargets > 0:
                    indicesIn[0, cur_pts:cur_pts + prevCount] = self.previousCloud[5, :prevCount, (fNum - i) % 10]
                cur_pts += prevCount

        if self.graphFin:
            self.plotstart = int(round(time.time() * 1000))
            self.graphFin = 0
            try:
                indicesIn_arr = indicesIn[0, :] if len(indicesIn.shape) > 1 else indicesIn
            except Exception:
                indicesIn_arr = []
            self.get_thread = updateQTTargetThread3D(
                pointIn, targets, indicesIn_arr, self.scatter, self.pcplot,
                numTargets, self.ellipsoids, self.coordStr, classifierOutput,
                self.zRange, self.gw, self.plotByIndex.isChecked(),
                self.plotTracks.isChecked(), self.bbox,
                self.boundaryBoxes[0]['checkEnable'].isChecked()
            )
            self.get_thread.done.connect(self.graphDone)
            self.get_thread.start(priority=QThread.HighestPriority - 1)

        # Height data calculation for targets (Instantaneous, Average, Delta)
        fNum100 = self.frameNum % 100
        for t in range(numTargets):
            col = t if targets.shape[1] > t else 0
            tid = int(targets[0, col])
            tid_idx = tid % 256

            # Associate points from current frame pointCloud
            tIndices = np.where(np.array(indexes) == tid)[0]
            if len(tIndices) > 0 and pointCloud.shape[1] >= len(indexes):
                tPointsZ = pointCloud[2, tIndices]
                # Filter ceiling reflections by using 90th percentile instead of max
                p_top_z = float(np.percentile(tPointsZ, 90))
                raw_height = p_top_z + float(self.sensorHeight)
                z_spread = float(np.amax(tPointsZ) - np.amin(tPointsZ))
                y_spread = float(np.amax(pointCloud[1, tIndices]) - np.amin(pointCloud[1, tIndices]))
                x_spread = float(np.amax(pointCloud[0, tIndices]) - np.amin(pointCloud[0, tIndices]))
            else:
                # Authoritative tracked vertical position from target centroid z with torso offset
                raw_height = float(targets[3, col]) + float(self.sensorHeight) + 0.35
                z_spread = 0.5
                y_spread = 0.5
                x_spread = 0.5

            # Temporal low-pass filter to eliminate saw-tooth spikes on instantaneous height
            prev_f = (fNum100 - 1) % 100
            prev_instant = float(self.targetSize[0, tid_idx, prev_f])
            if prev_instant > 0:
                cur_height = 0.65 * raw_height + 0.35 * prev_instant
            else:
                cur_height = raw_height

            self.targetSize[0, tid_idx, fNum100] = cur_height
            self.targetSize[1, tid_idx, fNum100] = z_spread
            self.targetSize[2, tid_idx, fNum100] = y_spread
            self.targetSize[3, tid_idx, fNum100] = x_spread

            if tid in self.lastTID:
                self.targetSize[4, tid_idx, 0] = self.targetSize[4, tid_idx, 0] + 1
                age = int(self.targetSize[4, tid_idx, 0])
                prev_f = (fNum100 - 1) % 100
                prev_avg = self.targetSize[5, tid_idx, prev_f]
                if prev_avg == 0:
                    prev_avg = cur_height
                self.targetSize[5, tid_idx, fNum100] = (
                    (1.0 / self.numFrameAvg) * cur_height +
                    ((self.numFrameAvg - 1.0) / self.numFrameAvg) * prev_avg
                )
                if age > 10:
                    past_f = (fNum100 - 10) % 100
                    past_avg = self.targetSize[5, tid_idx, past_f]
                    self.targetSize[6, tid_idx, fNum100] = (
                        self.targetSize[5, tid_idx, fNum100] - past_avg
                    )
                    # Fallback heuristic trigger if ML detector is not active
                    if self.fall_detector is None and self.targetSize[6, tid_idx, fNum100] < self.fallThresh and fallDetEn:
                        self.fallAlert.setText('FALLEN!')
                        self.fallAlert.setStyleSheet("font-size: 16pt; font-weight: bold; color: red; background-color: #ffcccc; padding: 4px;")
                        if not self.fallingPicture.isNull():
                            self.fallPic.setPixmap(self.fallingPicture)
                        if self.fallResetTimerOn == 0:
                            self.fallResetTimerOn = 1
                            self.fallTimer.start(5000)
                else:
                    self.targetSize[6, tid_idx, fNum100] = 0.0
            else:
                self.targetSize[4, tid_idx, 0] = 1
                self.targetSize[5, tid_idx, fNum100] = cur_height
                self.targetSize[6, tid_idx, fNum100] = 0.0

            # Floating 3D status text above target
            if tid < len(self.coordStr):
                tx = targets[1, col]
                ty = targets[2, col]
                tz = targets[3, col]
                t_pred = next((p for p in ml_predictions if p.get("tid") == tid), None)
                if t_pred:
                    posture = t_pred.get("posture", "UPRIGHT")
                    conf = t_pred.get("confidence", 0.95)
                    is_fall = t_pred.get("is_fall_instant") or t_pred.get("active_alert")
                    if is_fall:
                        lbl = f"⚠️ TID:{tid} FALLEN! ({conf*100:.0f}%)"
                    else:
                        lbl = f"TID:{tid} {posture} ({conf*100:.0f}%)"
                else:
                    lbl = f"TID:{tid} z={tz:.2f}m"
                self.coordStr[tid].setText(lbl)

        if numTargets > 0:
            self.lastTID = [targets[0, i] for i in range(numTargets)]
        else:
            self.lastTID = []

        # Trigger Height Graph Thread
        if self.hGraphFin:
            self.hGraphFin = 0
            self.heightThread = updateHeightGraphs(self.targetSize, self.heightPlotsDict, self.frameNum, self.lastTID)
            self.heightThread.done.connect(self.heightGraphDone)
            self.heightThread.start(priority=QThread.HighestPriority - 2)

    def graphDone(self):
        plotend = int(round(time.time() * 1000))
        plotime = plotend - self.plotstart
        try:
            if self.frameNum > 1:
                self.averagePlot = (plotime * 1.0 / self.frameNum) + (self.averagePlot * (self.frameNum - 1.0) / self.frameNum)
            else:
                self.averagePlot = plotime
        except Exception:
            self.averagePlot = plotime
        self.graphFin = 1
        self.frameNumDisplay.setText(f'Frame: {self.frameNum}')
        self.plotTimeDisplay.setText(f'Average Plot time: {str(plotime)[:5]} ms')

    def heightGraphDone(self, toPlot):
        self.hGraphFin = 1
        if toPlot.get('success'):
            height = toPlot['height']
            mH = toPlot['mH']
            dH = toPlot['dH']
            x = toPlot['x']
            self.heightPlotsDict['instant'].setData(x, height)
            self.heightPlotsDict['mean'].setData(x, mH)
            self.heightPlotsDict['deriv'].setData(x, dH)

    def _update_alert_channel_status(self):
        """Update the visual status badge for ESP32 and Telegram alerts."""
        if not hasattr(self, 'alertStatusLabel') or not self.fall_detector:
            return
        parts = []
        if getattr(self.fall_detector, "buzzer", None) and self.fall_detector.buzzer.enabled:
            st = self.fall_detector.buzzer.get_status()
            if st["connected"]:
                parts.append(f"ESP32: Connected ({st['transport']})")
            elif self.fall_detector.buzzer.port or self.fall_detector.buzzer.wifi_ip:
                parts.append("ESP32: Standby")
            else:
                parts.append("ESP32: Port Unset (.env)")
        else:
            parts.append("ESP32: Off")

        if getattr(self.fall_detector, "telegram", None) and self.fall_detector.telegram.enabled:
            tst = self.fall_detector.telegram.get_status()
            forced = self.fall_detector.get_forced_state()
            if forced:
                parts.append(f"Telegram: Override ({forced})")
            elif tst["configured"]:
                parts.append("Telegram: Armed (@testfallbot)")
            else:
                parts.append("Telegram: Token Unset (.env)")
        else:
            parts.append("Telegram: Off")

        self.alertStatusLabel.setText(f"Alerts: {' | '.join(parts)}")

    def _check_telegram_remote_override(self):
        """Periodic check (every 250ms) for remote commands from Telegram (/fall, /upright)."""
        if not self.fall_detector:
            return
        forced = self.fall_detector.get_forced_state()
        self._update_alert_channel_status()

        if forced == "FALL":
            # Force UI to FALLEN! regardless of whether radar tracks targets or not
            if self.fallAlert.text() != "FALLEN!":
                self.fallAlert.setText('FALLEN!')
                self.fallAlert.setStyleSheet(
                    "font-size: 16pt; font-weight: bold; color: red; "
                    "background-color: #ffcccc; border: 2px solid red; "
                    "border-radius: 4px; padding: 4px;"
                )
                if not self.fallingPicture.isNull():
                    self.fallPic.setPixmap(self.fallingPicture)
                self.fallDetEnabled.setText('🚨 FALL OVERRIDE (Telegram Command /fall)')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #D32F2F; font-weight: bold;")
                self.confidenceLabel.setText('Confidence: 100.0% (Remote)')
                self.latencyLabel.setText('Inference Latency: 0.0 ms')
                if getattr(self.fall_detector, "buzzer", None):
                    self.fall_detector.buzzer.trigger_fall_alarm()
        elif forced == "UPRIGHT":
            # Force UI to Upright regardless of radar blips
            if self.fallAlert.text() == "FALLEN!":
                self.fallAlert.setText('Upright')
                self.fallAlert.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E7D32; padding: 4px;")
                if not self.standingPicture.isNull():
                    self.fallPic.setPixmap(self.standingPicture)
                self.fallDetEnabled.setText('Normal Posture: UPRIGHT (Telegram Command /upright)')
                self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #2E7D32;")
                self.confidenceLabel.setText('Confidence: 100.0% (Remote)')
                self.latencyLabel.setText('Inference Latency: 0.0 ms')
                if getattr(self.fall_detector, "buzzer", None):
                    self.fall_detector.buzzer.clear_alarm(force=True)

    def acknowledgeAlert(self):
        """Manually silence active buzzer alarm and reset fall alert badge."""
        if self.fall_detector:
            self.fall_detector.acknowledge_alarm()
            if hasattr(self.fall_detector, "set_forced_state"):
                self.fall_detector.set_forced_state("UPRIGHT")
        self.resetFallText()

    def resetFallText(self):
        self.fallAlert.setText('Upright')
        self.fallAlert.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E7D32; padding: 4px;")
        if not self.standingPicture.isNull():
            self.fallPic.setPixmap(self.standingPicture)
        self.fallDetEnabled.setText('Fall Detection: Active (Normal Tracking)')
        self.fallDetEnabled.setStyleSheet("font-size: 9.5pt; color: #2E7D32;")
        self.fallResetTimerOn = 0
        if self.fall_detector and getattr(self.fall_detector, "buzzer", None):
            self.fall_detector.buzzer.clear_alarm(force=True)

    def closeEvent(self, event):
        """Safely release ML model alert dispatchers on GUI shutdown."""
        if self.fall_detector:
            try:
                self.fall_detector.close()
            except Exception:
                pass
        super().closeEvent(event)

    def updateFallThresh(self):
        try:
            newThresh = float(self.fallThreshInput.text())
            self.fallThresh = newThresh
            self.fallThreshMarker.setPos(self.fallThresh)
        except Exception:
            pass

    # ---------------------------------------------------------
    # SERIAL PORT & REPLAY CONNECTION ROUTINES
    # ---------------------------------------------------------
    def connectCom(self):
        if self.connectButton.text() == 'Disconnect':
            self._disconnect()
            return

        cfg_type = self.configType.currentText()
        if cfg_type == "Replay":
            self._start_replay_connection()
            return

        # Physical serial connection
        self.parser = uartParserSDK(type=cfg_type)
        self.parser.frameTime = self.frameTime

        self.uart_thread = parseUartThread(self.parser)
        self.uart_thread.fin.connect(self.updateGraph)

        self.parseTimer = QTimer()
        self.parseTimer.setSingleShot(False)
        self.parseTimer.timeout.connect(self.parseData)

        u_text = self.uartCom.text().strip()
        d_text = self.dataCom.text().strip()
        uart = u_text if u_text.upper().startswith("COM") else f"COM{u_text}"
        data = d_text if d_text.upper().startswith("COM") else f"COM{d_text}"

        try:
            self.parser.connectComPorts(uart, data)
            self.connectStatus.setText(f'Connected ({uart}/{data})')
            self.connectStatus.setStyleSheet("color: green; font-weight: bold;")
            self.connectButton.setText('Disconnect')
            print(f"[GUI] Connected to Radar CLI on {uart} and Data on {data}")
        except Exception as e:
            print(f"[GUI] Serial Connection Error on {uart}/{data}: {e}")
            self.connectStatus.setText('Unable to Connect')
            self.connectStatus.setStyleSheet("color: red; font-weight: bold;")
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.warning(
                    self,
                    "Radar Connection Failed",
                    f"Could not open radar serial ports:\nCLI Port: {uart}\nData Port: {data}\n\n"
                    f"Error: {e}\n\n"
                    "Please verify that the IWR6843ISK-ODS sensor is plugged into USB, "
                    "the CP2105 driver is installed, and the COM ports match Device Manager.\n\n"
                    "Tip: You can select 'Replay' under Config Type to run live inference on recorded dataset trials without physical hardware."
                )

    def _start_replay_connection(self):
        """Allow testing the visualizer with recorded trial data."""
        fd = QFileDialog()
        trial_dir = fd.getExistingDirectory(
            self,
            "Select Recorded Radar Trial Directory",
            str(PROJECT_ROOT / "datasets" / "my_posture_dataset")
        )
        if not trial_dir:
            # Fall back to dataset search
            trial_path = PROJECT_ROOT / "dataset" / "archive_old_5class_dataset" / "person1" / "falling" / "T01_20261008_132511"
            if trial_path.exists():
                trial_dir = str(trial_path)
            else:
                return

        try:
            self.replay_acquisition = RadarReplayAcquisition(Path(trial_dir), playback_fps=18.18)
            self.replay_generator = self.replay_acquisition.stream_frames()
            self.connectStatus.setText(f'Replay: {Path(trial_dir).name}')
            self.connectStatus.setStyleSheet("color: blue; font-weight: bold;")
            self.connectButton.setText('Disconnect')
            self.replay_timer.start(int(1000.0 / 18.18))
            print(f"[GUI] Started Replay on: {trial_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Replay Error", f"Failed to load replay trial:\n{e}")

    def _replay_step(self):
        """Step one frame forward in replay mode."""
        if not self.replay_generator:
            return
        try:
            frame: ParsedFrame = next(self.replay_generator)
            num_targets = len(frame.targets)
            num_points = len(frame.points)

            # Convert to TI GUI format
            targets_arr = np.zeros((13, max(num_targets, 1)))
            for i, t in enumerate(frame.targets):
                targets_arr[0, i] = t.tid
                targets_arr[1, i] = t.pos_x
                targets_arr[2, i] = t.pos_y
                targets_arr[3, i] = t.pos_z
                targets_arr[4, i] = t.vel_x
                targets_arr[5, i] = t.vel_y
                targets_arr[6, i] = t.vel_z
                targets_arr[10, i] = t.acc_x
                targets_arr[11, i] = t.acc_y
                targets_arr[12, i] = t.acc_z

            pts_arr = np.zeros((5, max(num_points, 1)))
            indices_arr = np.ones((max(num_points, 1),)) * 255
            for j, p in enumerate(frame.points):
                pts_arr[0, j] = p.x
                pts_arr[1, j] = p.y
                pts_arr[2, j] = p.z
                pts_arr[3, j] = p.doppler
                pts_arr[4, j] = p.snr
                indices_arr[j] = p.track_index

            parsedData = (
                pts_arr,
                targets_arr,
                indices_arr,
                num_points,
                num_targets,
                frame.frame_num,
                0,
                []
            )
            self.updateGraph(parsedData)
        except StopIteration:
            # Loop replay
            if self.replay_acquisition:
                self.replay_generator = self.replay_acquisition.stream_frames()

    def _disconnect(self):
        self.replay_timer.stop()
        self.replay_generator = None
        if hasattr(self, "parseTimer"):
            self.parseTimer.stop()
        if hasattr(self, "parser") and hasattr(self.parser, "dataCom"):
            try:
                self.parser.dataCom.close()
            except Exception:
                pass
        self.connectStatus.setText('Not Connected')
        self.connectStatus.setStyleSheet("color: black;")
        self.connectButton.setText('Connect')
        print("[GUI] Disconnected")

    def selectCfg(self):
        try:
            fname = self.selectFile()
            if fname:
                self.parseCfg(fname)
        except Exception as e:
            print(f"[GUI] Error selecting cfg: {e}")

    def selectFile(self):
        default_dir = str(PROJECT_ROOT / "configs")
        fd = QFileDialog()
        filename, _ = fd.getOpenFileName(self, "Select mmWave Radar Chirp Config", default_dir, "Config Files (*.cfg)")
        return filename

    def parseCfg(self, fname):
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            self.cfg = f.readlines()

        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if not args:
                continue
            if args[0] in ('SceneryParam', 'boundaryBox'):
                self.boundaryLine = counter
                self.profile['leftX'] = float(args[1])
                self.profile['rightX'] = float(args[2])
                self.profile['nearY'] = float(args[3])
                self.profile['farY'] = float(args[4])
                self.profile['bottomZ'] = float(args[5]) if len(args) > 5 else -1.5
                self.profile['topZ'] = float(args[6]) if len(args) > 6 else 1.5
                self.setBoundaryTextVals(self.profile)
                self.boundaryBoxes[0]['checkEnable'].setChecked(True)
            elif args[0] == 'profileCfg' and len(args) > 11:
                self.profile['startFreq'] = float(args[2])
                self.profile['idle'] = float(args[3])
                self.profile['adcStart'] = float(args[4])
                self.profile['rampEnd'] = float(args[5])
                self.profile['slope'] = float(args[8])
                self.profile['samples'] = float(args[10])
                self.profile['sampleRate'] = float(args[11])
            elif args[0] == 'frameCfg' and len(args) > 3:
                self.profile['numLoops'] = float(args[3])
                self.profile['numTx'] = float(args[2]) + 1
            elif args[0] == 'chirpCfg':
                chirpCount += 1
            elif args[0] == 'sensorPosition' and len(args) > 3:
                self.profile['sensorHeight'] = float(args[1])
                self.profile['az_tilt'] = float(args[2])
                self.profile['elev_tilt'] = float(args[3])
            counter += 1

        if self.profile.get('slope', 0) > 0 and self.profile.get('sampleRate', 0) > 0:
            self.profile['maxRange'] = self.profile['sampleRate'] * 1e3 * 0.9 * 3e8 / (2 * self.profile['slope'] * 1e12)
            bw = self.profile['samples'] / (self.profile['sampleRate'] * 1e3) * self.profile['slope'] * 1e12
            rangeRes = 3e8 / (2 * bw)
            Tc = (self.profile['idle'] * 1e-6 + self.profile['rampEnd'] * 1e-6) * max(chirpCount, 1)
            lda = 3e8 / (self.profile['startFreq'] * 1e9)
            maxVelocity = lda / (4 * Tc)
            velocityRes = lda / (2 * Tc * self.profile['numLoops'] * self.profile['numTx'])

            self.configTable.setItem(1, 1, QTableWidgetItem(f"{self.profile['maxRange']:.2f} m"))
            self.configTable.setItem(2, 1, QTableWidgetItem(f"{rangeRes:.3f} m"))
            self.configTable.setItem(3, 1, QTableWidgetItem(f"{maxVelocity:.2f} m/s"))
            self.configTable.setItem(4, 1, QTableWidgetItem(f"{velocityRes:.3f} m/s"))

        self.az_tilt.setText(str(self.profile['az_tilt']))
        self.elev_tilt.setText(str(self.profile['elev_tilt']))
        self.s_height.setText(str(self.profile['sensorHeight']))
        self.gz.translate(0, 0, 3 - self.profile['sensorHeight'])
        self.changeBoundaryBox()

    def sendCfg(self):
        if not hasattr(self, "cfg") or not self.cfg:
            QMessageBox.warning(self, "Missing Configuration", "Please select a valid chirp .cfg file first!")
            return
        if not hasattr(self, "parser") or not hasattr(self.parser, "uartCom"):
            QMessageBox.warning(self, "Not Connected", "Please connect to the radar COM ports first!")
            return

        try:
            self.parser.sendCfg(self.cfg)
            self.configSent = 1
            self.parseTimer.start(self.frameTime)
            print("[GUI] Chirp configuration successfully transmitted to sensor!")
        except Exception as e:
            print(f"[GUI] Error sending config: {e}")
            QMessageBox.critical(self, "Config Transmission Failed", f"Could not transmit config to radar:\n{e}")

    def parseData(self):
        if hasattr(self, "uart_thread") and not self.uart_thread.isRunning():
            self.uart_thread.start(priority=QThread.HighestPriority)

    def whoVisible(self):
        self.threeD = 1 if self.threeD == 0 else 0


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size() if screen else None
    window = Window(size=size)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
