#!/usr/bin/python

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
    QPushButton,
    QRadioButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QMessageBox
)
import qdarktheme
import argparse
import cv2
import numpy as np
import sys
import os
import io
import time

# classifier classes
from src.ensemble import EnsembleClassifier
from src.haar import HaarCascadeClassifier

class VideoThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)
    update_results_signal = pyqtSignal(object)
    done_averaging = pyqtSignal(bool)
    averaging_results = pyqtSignal(dict)

    def __init__(self, camera_index=0, target_classifier="ensemble", averaging_mode=False):
        super().__init__()
        self.camera_index = camera_index
        self.target_classifier = target_classifier
        self.averaging_mode = averaging_mode

        self.clf = None
        self.haar_clf = None

    
    def switch_classifier(self, target_classifier):
        """
        Setter for the `target_classifier` property.
        """
        self.target_classifier = target_classifier


    def toggle_averaging(self, value: bool):
        """
        Setter for the `averaging_mode` property.
        """
        self.averaging_mode = value


    def is_raspberry_pi(self):
        """
        Check if the hardware where this program runs on is using chips
        that are specific to Raspberry Pi. This is used by VideoThread
        to identify whether Picamera2 or OpenCV should be used.
        """
        try:
            if os.name != 'posix':
                return False

            chips = ('BCM2708', 'BCM2709', 'BCM2711', 'BCM2835', 'BCM2836')

            with io.open('/proc/cpuinfo', 'r') as cpuinfo:
                for line in cpuinfo:
                    if line.startswith('Hardware'):
                        _, value = line.strip().split(':', 1)
                        value = value.strip()
                        if value in chips:
                            print("Raspberry Pi chipset detected.")
                            return True
        except Exception:
            pass

        return False
    
    def run(self):
        """
        Primary entry point for VideoThread. Instructions in this method
        will automatically run as soon as a VideoThread instance is
        initialized.
        """
        is_rpi = self.is_raspberry_pi()

        # capture instance for Picamera2
        picam2 = None

        # capture instance for OpenCV
        cap = None
        
        # import and initialize Picamera2 if VideoThread is
        # initialized on a Raspberry Pi
        if is_rpi:
            from picamera2 import Picamera2
            from libcamera import Transform
            
            picam2 = Picamera2()
            
            config = {
                "format": "RGB888",
                "size": (640, 480)
            }

            picam2.configure(picam2.create_preview_configuration(main=config))
        
        # initialize classifiers/models
        self.clf = EnsembleClassifier()
        self.haar_clf = HaarCascadeClassifier()
        
        # Raspberry Pi
        if is_rpi:
            picam2.start()

            while True:
                if self.averaging_mode:
                    self.run_averaging_mode(picam2, is_rpi)
                else:
                    frame = picam2.capture_array()
                    ret = frame.any()

                    self.detect(self.clf, self.haar_clf, frame, ret)
        # Non-Raspberry-Pi Device (e.g. x86)
        else:
            cap = cv2.VideoCapture(self.camera_index)

            while True:
                if self.averaging_mode:
                    self.run_averaging_mode(cap, is_rpi)
                else:
                    ret, frame = cap.read()
                    self.detect(self.clf, self.haar_clf, frame, ret)


    def run_averaging_mode(self, cap, is_rpi: bool):
        labels = []
        probs = []
        ctr = 0

        while ctr < 40:
            if is_rpi:
                frame = cap.capture_array()
                ret = frame.any()
            else:
                ret, frame = cap.read()

            results = self.detect(self.clf, self.haar_clf, frame, ret)

            if results is not None:
                print(f"Frame: {ctr + 1}, Label: {results['label']}, Prob: {results['probability']}")
                labels.append(results['label'])
                probs.append(results['probability'].item())
            else:
                labels.append(None)
                probs.append(0)

            ctr += 1

            if not self.averaging_mode:
                self.done_averaging.emit(True)
                break

        self.averaging_mode = False
        self.done_averaging.emit(True)

        overalls = self.get_overalls(labels, probs)
        filepath = os.path.join(os.path.dirname(__file__), "results", f"averages_{time.time()}.txt")

        with open(filepath, 'a') as f:
            for idx, label in enumerate(labels):
                f.write(f"Frame {idx + 1}: {label} ({probs[idx]})\n")

            f.write("\nOVERALLS:\n")
            for label, prob in overalls.items():
                f.write(f"{label}: {prob}\n")

            f.close()

        print(f"Results written to {filepath}")

        highest_scoring = max(overalls, key=overalls.get)
        self.averaging_results.emit({ f"{highest_scoring}": overalls[highest_scoring] })


    def get_overalls(self, labels, probabilities):
        results = {}
        labels_set = {label for label in labels}

        for label in labels_set:
            probs = []

            for idx, x in enumerate(probabilities):
                if labels[idx] == label:
                    probs.append(x)

            results[label] = sum(probs) / 40

        return results

    
    def detect(self, ensemble_clf, haar_clf, frame, signal):
        """
        Method for performing inference using the Ensemble and 
        Haar Classifiers. This takes a `frame` argument, which 
        refers to the camera output (i.e. in OpenCV, this will
        be of MatLike type. The `signal` argument takes any form
        of data, and it will serve as an indicator that a camera
        output is being received).
        """
        if signal and self.target_classifier == "ensemble":
            clf_output = ensemble_clf.classify(frame, display=False, as_matlike=True, print_results=False)
            self.update_frame_signal.emit(clf_output['frame'])

            if clf_output['full_result'] is not None:
                self.update_results_signal.emit(clf_output['full_result'])
            else:
                self.update_results_signal.emit(None)

            return clf_output['full_result']

        if signal and self.target_classifier == "haar":
            clf_output = haar_clf.classify(frame, display=False, as_matlike=True)
            self.update_frame_signal.emit(clf_output['frame'])

            if clf_output['result'] is not None:
                self.update_results_signal.emit(clf_output['result'])
            else:
                self.update_results_signal.emit(None)

            return clf_output['result']


class App(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()

        self.camera_index = camera_index
        self.current_frame = None
        self.active_classifier = "ensemble"
        self.averaging_mode = False
        
        self.setWindowTitle("Cockatiel Species Classifier")
        self.setup_interface()

        self.vthread = self.create_video_thread(camera_index)
        self.vthread.start()


    def setup_interface(self):
        """
        Entry point for initializing the User Interface.
        """
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
        """
        Converts a PyQt Layout to a QWidget instance.
        """
        widget = QWidget()
        widget.setLayout(layout)

        return widget


    def setup_ensemble_results_view(self):
        """
        This initializes the view which displays the results
        returned by the Ensemble Classifier.
        """
        view_layout = QVBoxLayout()
        results_layout = QGridLayout()

        results_heading_1 = QLabel("Label")
        results_heading_1.setFont(QFont("Manrope", 14, 600))
        results_heading_2 = QLabel("Probability")
        results_heading_2.setFont(QFont("Manrope", 14, 600))

        results_layout.addWidget(results_heading_1, 0, 0)
        results_layout.addWidget(results_heading_2, 0, 1)

        self.ensemble_results_label = QLabel("No Detections")
        self.ensemble_results_label.setFont(QFont("Manrope", 14, 400))
        self.ensemble_results_prob = QLabel("0.00")
        self.ensemble_results_prob.setFont(QFont("Manrope", 14, 400))

        results_layout.addWidget(self.ensemble_results_label, 1, 0)
        results_layout.addWidget(self.ensemble_results_prob, 1, 1)

        view_layout.addLayout(results_layout)

        self.toggle_averaging_btn = QPushButton("Run in Averaging Mode (40 frames)")
        self.toggle_averaging_btn.setCheckable(True)
        self.toggle_averaging_btn.setChecked(False)
        self.toggle_averaging_btn.setStyleSheet("font-size: 16px; font-family: 'Manrope'; font-weight: 600;")
        self.toggle_averaging_btn.clicked.connect(self.toggle_averaging)
        
        view_layout.addWidget(self.toggle_averaging_btn)

        return view_layout


    def toggle_averaging(self):
        self.averaging_mode = True if self.averaging_mode is False else False
        self.toggle_averaging_btn.setChecked(True)
        self.vthread.toggle_averaging(self.averaging_mode)
    

    def setup_haar_results_view(self):
        """
        This initializes the view which displays the results
        returned by the Haar Cascade Classifier.
        """
        results_layout = QGridLayout()

        results_heading_1 = QLabel("Label")
        results_heading_1.setFont(QFont("Manrope", 14, 600))
        results_heading_2 = QLabel("Score")
        results_heading_2.setFont(QFont("Manrope", 14, 600))

        results_layout.addWidget(results_heading_1, 0, 0)
        results_layout.addWidget(results_heading_2, 0, 1)

        self.haar_results_label = QLabel("No Detections")
        self.haar_results_label.setFont(QFont("Manrope", 14, 400))
        
        self.haar_results_prob = QLabel("0.00")
        self.haar_results_prob.setFont(QFont("Manrope", 14, 400))

        results_layout.addWidget(self.haar_results_label, 1, 0)
        results_layout.addWidget(self.haar_results_prob, 1, 1)

        return results_layout
    

    def setup_classifier_switches_view(self):
        """
        This initializes Radio buttons for switching between the
        Haar Cascade Classifier and the Ensemble Classifier.
        """
        switches_layout = QHBoxLayout()

        self.toggle_ensemble = QPushButton("Ensemble Classifier (SqueezeNet + Haar)")
        self.toggle_ensemble.setCheckable(True)
        self.toggle_ensemble.setChecked(True)
        self.toggle_ensemble.setStyleSheet("font-size: 16px; font-family: 'Manrope'; font-weight: 600;")
        self.toggle_ensemble.clicked.connect(lambda: self.switch_classifier("ensemble"))

        self.toggle_haar = QPushButton("Haar Cascade Classifier")
        self.toggle_haar.setCheckable(True)
        self.toggle_haar.setStyleSheet("font-size: 16px; font-family: 'Manrope'; font-weight: 600;")
        self.toggle_haar.clicked.connect(lambda: self.switch_classifier("haar"))

        switches_layout.addWidget(self.toggle_ensemble)
        switches_layout.addWidget(self.toggle_haar)

        return switches_layout
    

    def switch_classifier(self, target_classifier):
        """
        Method which updates all properties related to classifier switching.
        """
        # update local property
        self.active_classifier = target_classifier

        # update the target classifier used by VideoThread
        self.vthread.switch_classifier(self.active_classifier)

        # change view
        if self.active_classifier == "ensemble":
            self.results_stack.setCurrentIndex(0)
            self.toggle_ensemble.setChecked(True)
            self.toggle_haar.setChecked(False)
        if self.active_classifier == "haar":
            self.results_stack.setCurrentIndex(1)
            self.toggle_ensemble.setChecked(False)
            self.toggle_haar.setChecked(True)


    def create_video_thread(self, camera_index):
        """
        Initializes a VideoThread instance.
        """
        vthread = VideoThread(camera_index)
        vthread.update_frame_signal.connect(self.update_frame)
        vthread.update_results_signal.connect(self.update_results)
        vthread.done_averaging.connect(self.done_averaging)
        vthread.averaging_results.connect(self.display_averaging_results)
        
        return vthread

    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        self.current_frame = frame
        self.update_active_frame(frame)


    @pyqtSlot(bool)
    def done_averaging(self, status):
        if status:
            self.averaging_mode = False
            self.toggle_averaging_btn.setChecked(False)

    
    @pyqtSlot(dict)
    def display_averaging_results(self, results):
        if results:
            result_class = list(results.keys())[0]
            self.alert = QMessageBox()
            self.alert.setText(f"HIGHEST-SCORING CLASS: {result_class} ({results[result_class]})")
            self.alert.exec()
    
    
    @pyqtSlot(object)
    def update_results(self, results):
        label = "No Detections"
        probability = "0.00"

        if results is not None:
            if self.active_classifier == "ensemble" and self.vthread.target_classifier == "ensemble":
                if 'label' in results:
                    label = results['label']
                if 'probability' in results:
                    probability = results['probability']

                self.ensemble_results_label.setText(f"{label}")
                self.ensemble_results_prob.setText(f"{probability}")
            
            if self.active_classifier == "haar" and self.vthread.target_classifier == "haar":
                if 'label' in results:
                    label = results['label']
                if 'score' in results:
                    probability = results['score']
                
                self.haar_results_label.setText(f"{label}")
                self.haar_results_prob.setText(f"{probability}")
        else:
            self.ensemble_results_label.setText(label)
            self.haar_results_label.setText(label)

            self.ensemble_results_prob.setText(probability)
            self.haar_results_prob.setText(probability)


    def update_active_frame(self, frame):
        """
        Updates the Pixmap/Image displayed on the UI.
        """
        pixmap = self.cv2pixmap(frame)
        self.active_frame.setPixmap(pixmap)


    def cv2pixmap(self, cv2_frame):
        """
        Converts OpenCV/MatLike inputs into Pixmap.
        """
        frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        image_widget = QImage(frame.data, width, height, channel * width, QImage.Format_RGB888)
        return QPixmap.fromImage(image_widget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", '-c', nargs='?', type=int, default=0, help="Index of the camera that will be used by cv2.VideoCapture()")

    qdarktheme.enable_hi_dpi()
    args = parser.parse_args()

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    a = App(camera_index=args.camera)
    a.show()
    sys.exit(app.exec_())
