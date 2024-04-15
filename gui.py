from PyQt5.QtCore import (
    Qt,
    QObject,
    QThread,
    pyqtSignal,
    pyqtSlot
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget
)
import argparse
import cv2
import numpy as np
import sys

from src.ensemble import EnsembleClassifier

class VideoThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)
    update_results_signal = pyqtSignal(object)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
    
    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        clf = EnsembleClassifier()

        while True:
            ret, frame = cap.read()
            
            if ret:
                clf_output = clf.classify(frame, display=False, as_matlike=True, print_results=False)
                self.update_frame_signal.emit(clf_output['frame'])

                if clf_output['full_result'] is not None:
                    self.update_results_signal.emit(clf_output['full_result'])
                else:
                    self.update_results_signal.emit(None)


class App(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()

        self.camera_index = camera_index
        self.current_frame = None
        
        self.setWindowTitle("Cockatiel Species Classifier")
        self.setup_interface()

        self.vthread = self.create_video_thread(camera_index)
        self.vthread.start()


    def setup_interface(self):
        main = QVBoxLayout()
        self.setLayout(main)

        self.active_frame = QLabel(self)
        self.active_frame.resize(256, 256)
        main.addWidget(self.active_frame, alignment=Qt.AlignCenter)

        results_layout = QGridLayout()

        results_heading_1 = QLabel("Label")
        results_heading_1.setFont(QFont("Arial", 14, 600))
        results_heading_2 = QLabel("Probability")
        results_heading_2.setFont(QFont("Arial", 14, 600))

        results_layout.addWidget(results_heading_1, 0, 0)
        results_layout.addWidget(results_heading_2, 0, 1)

        self.results_label = QLabel("No Detections")
        self.results_label.setFont(QFont("Arial", 14, 400))
        self.results_prob = QLabel("0.00")
        self.results_prob.setFont(QFont("Arial", 14, 400))

        results_layout.addWidget(self.results_label, 1, 0)
        results_layout.addWidget(self.results_prob, 1, 1)
        
        main.addLayout(results_layout)
        
        self.show()


    def create_video_thread(self, camera_index):
        vthread = VideoThread(camera_index)
        vthread.update_frame_signal.connect(self.update_frame)
        vthread.update_results_signal.connect(self.update_results)
        
        return vthread

    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        self.current_frame = frame
        self.update_active_frame(frame)

    
    @pyqtSlot(object)
    def update_results(self, results):
        label = results['label'] if results is not None else "No Detections"
        probability = results['probability'] if results is not None else "0.00"
        self.results_label.setText(f"{label}")
        self.results_prob.setText(f"{probability}")


    def update_active_frame(self, frame):
        pixmap = self.cv2pixmap(frame)
        self.active_frame.setPixmap(pixmap)


    def cv2pixmap(self, cv2_frame):
        frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        image_widget = QImage(frame.data, width, height, channel * width, QImage.Format_RGB888)
        return QPixmap.fromImage(image_widget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", '-c', nargs='?', type=int, default=0, help="Index of the camera that will be used by cv2.VideoCapture()")

    args = parser.parse_args()

    app = QApplication(sys.argv)
    a = App(camera_index=args.camera)
    a.show()
    sys.exit(app.exec_())
