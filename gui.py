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
    QRadioButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget
)
import argparse
import cv2
import numpy as np
import sys

# classifier classes
from src.ensemble import EnsembleClassifier
from src.haar import HaarCascadeClassifier

class VideoThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)
    update_results_signal = pyqtSignal(object)

    def __init__(self, camera_index=0, target_classifier="ensemble"):
        super().__init__()
        self.camera_index = camera_index
        self.target_classifier = target_classifier

    def switch_classifier(self, target_classifier):
        self.target_classifier = target_classifier
    
    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        
        clf = EnsembleClassifier()
        haar_clf = HaarCascadeClassifier()

        while True:
            ret, frame = cap.read()

            if ret and self.target_classifier == "ensemble":
                clf_output = clf.classify(frame, display=False, as_matlike=True, print_results=False)
                self.update_frame_signal.emit(clf_output['frame'])

                if clf_output['full_result'] is not None:
                    self.update_results_signal.emit(clf_output['full_result'])
                else:
                    self.update_results_signal.emit(None)
            
            if ret and self.target_classifier == "haar":
                clf_output = haar_clf.classify(frame, display=False, as_matlike=True)
                self.update_frame_signal.emit(clf_output['frame'])

                if clf_output['result'] is not None:
                    self.update_results_signal.emit(clf_output['result'])
                else:
                    self.update_results_signal.emit(None)


class App(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()

        self.camera_index = camera_index
        self.current_frame = None
        self.active_classifier = "ensemble"
        
        self.setWindowTitle("Cockatiel Species Classifier")
        self.setup_interface()

        self.vthread = self.create_video_thread(camera_index)
        self.vthread.start()


    def setup_interface(self):
        main = QVBoxLayout()
        self.setLayout(main)

        # camera view
        self.active_frame = QLabel(self)
        self.active_frame.resize(256, 256)
        main.addWidget(self.active_frame, alignment=Qt.AlignCenter)

        # classifier switches
        switches_layout = self.setup_classifier_switches_view()
        main.addLayout(switches_layout)
        
        # results view
        self.results_stack = QStackedLayout()

        ensemble_results_layout = self.setup_ensemble_results_view()
        ensemble_results_widget = self.convert_layout_to_widget(ensemble_results_layout)
        self.results_stack.addWidget(ensemble_results_widget)

        haar_results_layout = self.setup_haar_results_view()
        haar_results_widget = self.convert_layout_to_widget(haar_results_layout)
        self.results_stack.addWidget(haar_results_widget)

        main.addLayout(self.results_stack)

        self.show()


    def convert_layout_to_widget(self, layout):
        widget = QWidget()
        widget.setLayout(layout)

        return widget


    def setup_ensemble_results_view(self):
        results_layout = QGridLayout()

        results_heading_1 = QLabel("Label")
        results_heading_1.setFont(QFont("Arial", 14, 600))
        results_heading_2 = QLabel("Probability")
        results_heading_2.setFont(QFont("Arial", 14, 600))

        results_layout.addWidget(results_heading_1, 0, 0)
        results_layout.addWidget(results_heading_2, 0, 1)

        self.ensemble_results_label = QLabel("No Detections")
        self.ensemble_results_label.setFont(QFont("Arial", 14, 400))
        self.ensemble_results_prob = QLabel("0.00")
        self.ensemble_results_prob.setFont(QFont("Arial", 14, 400))

        results_layout.addWidget(self.ensemble_results_label, 1, 0)
        results_layout.addWidget(self.ensemble_results_prob, 1, 1)

        return results_layout
    

    def setup_haar_results_view(self):
        results_layout = QGridLayout()

        results_heading_1 = QLabel("Label")
        results_heading_1.setFont(QFont("Arial", 14, 600))
        results_heading_2 = QLabel("Probability")
        results_heading_2.setFont(QFont("Arial", 14, 600))

        results_layout.addWidget(results_heading_1, 0, 0)
        results_layout.addWidget(results_heading_2, 0, 1)

        self.haar_results_label = QLabel("No Detections")
        self.haar_results_label.setFont(QFont("Arial", 14, 400))
        
        self.haar_results_prob = QLabel("0.00")
        self.haar_results_prob.setFont(QFont("Arial", 14, 400))

        results_layout.addWidget(self.haar_results_label, 1, 0)
        results_layout.addWidget(self.haar_results_prob, 1, 1)

        return results_layout
    

    def setup_classifier_switches_view(self):
        switches_layout = QHBoxLayout()

        toggle_ensemble = QRadioButton()
        toggle_ensemble.setText("Ensemble Classifier (SqueezeNet + Haar)")
        toggle_ensemble.setChecked(True)
        toggle_ensemble.toggled.connect(lambda: self.switch_classifier("ensemble"))

        toggle_haar = QRadioButton()
        toggle_haar.setText("Haar Cascade Classifier")
        toggle_haar.toggled.connect(lambda: self.switch_classifier("haar"))

        switches_layout.addWidget(toggle_ensemble)
        switches_layout.addWidget(toggle_haar)

        return switches_layout
    

    def switch_classifier(self, target_classifier):
        # update local property
        self.active_classifier = target_classifier

        # update the target classifier used by VideoThread
        self.vthread.switch_classifier(self.active_classifier)

        # change view
        if self.active_classifier == "ensemble":
            self.results_stack.setCurrentIndex(0)
        if self.active_classifier == "haar":
            self.results_stack.setCurrentIndex(1)


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
        if self.active_classifier == "ensemble" and self.vthread.target_classifier == "ensemble":
            label = results['label'] if results is not None else "No Detections"
            probability = results['probability'] if results is not None else "0.00"
            self.ensemble_results_label.setText(f"{label}")
            self.ensemble_results_prob.setText(f"{probability}")

        if self.active_classifier == "haar" and self.vthread.target_classifier == "haar":
            label = "No Detections"
            probability = "0.00"

            if 'label' in results:
                label = results['label']

            if 'score' in results:
                probability = results['score']
            
            self.haar_results_label.setText(f"{label}")
            self.haar_results_prob.setText(f"{probability}")


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
