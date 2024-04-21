from .haar import HaarCascadeClassifier
from .squeezenet import SqueezeNet
import cv2
import os

DEFAULT_SQUEEZENET_CLF_PATH = "pretrained/model_10-epochs_1713362879.588241.pt"

class EnsembleClassifier(HaarCascadeClassifier):

    def __init__(self, squeezenet_clf_path=DEFAULT_SQUEEZENET_CLF_PATH):
        super().__init__()
        self.squeezenet_clf = SqueezeNet()
        self.squeezenet_clf.load_custom_model(self.parse_model_path(squeezenet_clf_path))

        self.bbox_colors = {"bird": (0, 255, 0),
                            "cinnamon": (255, 0, 0),
                            "lutino": (0, 0, 255),
                            "pearl": (0, 255, 255),
                            "pied": (255, 255, 0),
                            "whiteface": (255, 0, 255)}

    def parse_model_path(self, model_path):
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, '..', model_path)
        return filepath

    # override
    def classify(self, image, display=True, as_matlike=False, print_results=True):
        """
        Runs the entire object detection + classification process.
        """
        frame = image
        result_label = ""

        if not as_matlike:
            frame = cv2.imread(image)

        roi_frame = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Haar Cascade
        birds = self.detect(self.bird_clf, gray)
        cockatiels = self.detect(self.cockatiel_clf, gray)

        bird_det = self.get_detections("Bird", birds)
        cockatiel_det = self.get_detections("Cockatiel", cockatiels)

        haar_result = self.parse_result(bird_det, cockatiel_det)

        snet_result = None

        # if the Haar classifier detects a cockatiel, run the SqueezeNet classifier
        if haar_result["label"] == "cockatiel":
            # if the input image is not MatLike, pass the path to the image instead
            # of the processed 'frame'
            if not as_matlike:
                snet_result = self.squeezenet_clf.test(image, print_results=print_results)
            else:
                snet_result = self.squeezenet_clf.test(frame, as_matlike=True, print_results=print_results)

            if print_results:
                print("[SqueezeNet Result] Label: {}, Probability: {}, Time: {}".format(snet_result["label"], snet_result["probability"], snet_result["inference_time"]))
            
            # use bounding boxes from Haar, and use label from SqueezeNet
            frame = self.attach_bounding_boxes(frame, haar_result["bboxes"], snet_result["label"])

            result_label = snet_result["label"]
        else:
            frame = self.attach_bounding_boxes(frame, haar_result["bboxes"], "bird")

            result_label = "bird"

        if display:
            self.display_cv_image(frame)

        return {
            "frame": frame,
            "result": result_label,
            "full_result": snet_result
        }

    # override
    def attach_bounding_boxes(self, frame, bboxes, label):
        """
        Overrides the attach_bounding_boxes() method from HaarCascadeClassifier
        by colorizing bounding boxes for different cockatiel species.
        """
        for (x, y, w, h) in bboxes:
            bbox_color = self.bbox_colors[label]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), bbox_color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.68, bbox_color, 2)

        return frame

    
    # override
    def classify_live(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            self.classify(frame, display=False, as_matlike=True)
            cv2.imshow("Cockatiel Variety Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
