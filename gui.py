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
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox
)
from pathlib import Path
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
from src.colors import retrieve_dominant_colors, get_color_luminance
from ultralytics import YOLO

YOLOV8_WEIGHTS_PATH = "./pretrained/best.pt"


class YOLOv8Thread(QObject):
    result = pyqtSignal(object)
    annotated = pyqtSignal(np.ndarray)
    done = pyqtSignal()

    def __init__(self, image_path, model_path=YOLOV8_WEIGHTS_PATH, parent=None):
        QThread.__init__(self, parent)
        self.image_path = image_path
        self.model_path = model_path
        self.model = YOLO(self.model_path)

    
    def run(self):
        image = cv2.imread(self.image_path)
        results = self.model.predict(image)[0]
        annotated = results.plot()
        dominant_colors = []

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            cropped = image[y1:y2, x1:x2]
            colors = retrieve_dominant_colors(cropped)
            dominant_colors.append(colors)
        
        self.result.emit(dominant_colors)
        self.annotated.emit(annotated)
        self.done.emit()


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

        # initialize classifiers/models
        self.clf = EnsembleClassifier()
        self.haar_clf = HaarCascadeClassifier()
        
        if is_rpi:
            # for RPi, use Picamera2 for camera inputs instead of OpenCV
            from picamera2 import Picamera2
            from libcamera import Transform
            
            picam2 = Picamera2()
            
            config = {
                "format": "RGB888",
                "size": (640, 480)
            }

            picam2.configure(picam2.create_preview_configuration(main=config, transform=Transform(vflip=1, hflip=1)))
            picam2.start()

            while True:
                if self.averaging_mode:
                    self.run_averaging_mode(picam2, is_rpi)
                else:
                    frame = picam2.capture_array()
                    ret = frame.any()

                    self.detect(self.clf, self.haar_clf, frame, ret)
        else:
            # for x86 and non-RPi devices
            cap = cv2.VideoCapture(self.camera_index)

            try:
                while True:
                    if self.averaging_mode:
                        self.run_averaging_mode(cap, is_rpi)
                    else:
                        ret, frame = cap.read()

                        if ret:
                            self.detect(self.clf, self.haar_clf, frame, ret)
            finally:
                cap.release()


    def run_averaging_mode(self, cap, is_rpi: bool):
        labels = []
        probs = []
        ctr = 0

        while ctr < 60:
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

            results[label] = sum(probs) / 60

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

        tabs = QTabWidget()
        tabs.addTab(self.detector_ui(), "Detector")
        tabs.addTab(self.color_density_analyzer_ui(), "Color Density")
        main.addWidget(tabs)

        self.show()


    def detector_ui(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # camera view
        self.active_frame = QLabel(self)
        self.active_frame.resize(256, 256)
        layout.addWidget(self.active_frame, alignment=Qt.AlignCenter)

        # classifier switches
        switches_layout = self.setup_classifier_switches_view()
        layout.addLayout(switches_layout)

        # results view
        self.results_stack = QStackedLayout()

        ensemble_results_layout = self.setup_ensemble_results_view()
        ensemble_results_widget = self.convert_layout_to_widget(ensemble_results_layout)
        self.results_stack.addWidget(ensemble_results_widget)

        haar_results_layout = self.setup_haar_results_view()
        haar_results_widget = self.convert_layout_to_widget(haar_results_layout)
        self.results_stack.addWidget(haar_results_widget)

        layout.addLayout(self.results_stack)
        tab.setLayout(layout)

        return tab


    def color_density_analyzer_ui(self):
        tab = QWidget()
        vbox = QVBoxLayout()
        io_row = QHBoxLayout()
        input_row = QHBoxLayout()

        # input image preview
        self.preview_box = QLabel(self)
        self.preview_box.resize(256, 256)

        # output results box
        self.output_box = QVBoxLayout()

        io_row.addWidget(self.preview_box, alignment=Qt.AlignCenter)
        io_row.addLayout(self.output_box)

        # filepath input element
        self.filepath_input = QLineEdit()
        self.filepath_input.setReadOnly(True)
        
        # select file button
        input_btn = QPushButton("Select an Image")
        input_btn.clicked.connect(lambda: self.select_file())

        input_row.addWidget(self.filepath_input)
        input_row.addWidget(input_btn)
        
        vbox.addLayout(io_row)
        vbox.addLayout(input_row)

        tab.setLayout(vbox)

        return tab


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

        self.ensemble_results_label = QLabel("Unidentified Cockatiel Mutation")
        self.ensemble_results_label.setFont(QFont("Manrope", 14, 400))
        self.ensemble_results_prob = QLabel("0.00")
        self.ensemble_results_prob.setFont(QFont("Manrope", 14, 400))

        results_layout.addWidget(self.ensemble_results_label, 1, 0)
        results_layout.addWidget(self.ensemble_results_prob, 1, 1)

        view_layout.addLayout(results_layout)

        self.toggle_averaging_btn = QPushButton("Cockatiel Mutation Classifier Averaging Mode (60 frames)")
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

        self.haar_results_label = QLabel("Unidentified Bird")
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


    def run_yolov8(self, input_image):
        """
        Spins up a thread for performing object detection with YOLOv8
        """
        self.yolo_thread = QThread()
        self.yolo = YOLOv8Thread(input_image)

        results = {
            "results": None,
            "image_path": None,
            "annotated": None
        }

        def update_results(value):
            nonlocal results
            results["results"] = value
            results["image_path"] = input_image

        def attach_annotated(value):
            nonlocal results
            results["annotated"] = value

        self.yolo_thread.started.connect(self.yolo.run)
        self.yolo_thread.finished.connect(self.yolo_thread.deleteLater)
        
        self.yolo.result.connect(update_results)
        self.yolo.annotated.connect(attach_annotated)
        self.yolo.done.connect(self.yolo_thread.quit)
        self.yolo.done.connect(self.yolo.deleteLater)
        
        self.yolo_thread.start()
        self.yolo_thread.finished.connect(lambda: self.pass_to_output_box(results))

    
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
            self.alert.setText(f"The cockatiel mutation is a {result_class} cockatiel with a probability of ({results[result_class]:.2%})")
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


    def select_file(self):
        """
        File selection method
        """
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd(), "Images (*.jpg *.jpeg *.png)")

        if not filename:
            return

        # parse filepath
        image_path = Path(filename)

        # set filepath indicator value
        self.filepath_input.setText(str(image_path))

        if not image_path or image_path is None:
            self.alert = QMessageBox()
            self.alert.setText("Please provide a valid input image")
            self.alert.exec()
            return

        # clear output box
        self.clear_layout(self.output_box)

        self.preview_box.setText("Running YOLOv8...")
        self.show()
        self.run_yolov8(image_path)


    def clear_layout(self, layout):
        """
        Method for clearing a layout of widgets
        """
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()


    def pass_to_output_box(self, results):
        """
        Display color analyzer results to output box
        """
        if results["annotated"] is not None:
            pixmap = self.cv2pixmap(results["annotated"]).scaledToHeight(500)
            self.preview_box.setPixmap(pixmap)

        if results["results"] is not None and len(results["results"]) > 0:
            for color in results["results"][0]:
                hex_code, percentage = color
                color_block = QLabel()

                # adjust text color according to the detected color's luminance
                luminance = get_color_luminance(hex_code)
                text_color = "black" if luminance > 186 else "white"

                color_block.setStyleSheet(f"background-color: {hex_code}; width: 50px; height: 50px; font-weight: 600; font-size: 14px; color: {text_color}")
                color_block.setText("{} ({:.2f}%)".format(hex_code, percentage))
                color_block.setAlignment(Qt.AlignCenter)
                self.output_box.addWidget(color_block)


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
