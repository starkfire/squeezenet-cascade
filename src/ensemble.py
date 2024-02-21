from .haar import HaarCascadeClassifier
from .squeezenet import SqueezeNet
import cv2

DEFAULT_SQUEEZENET_CLF_PATH = "./pretrained/model.pt"

class EnsembleClassifier(HaarCascadeClassifier):

    def __init__(self, squeezenet_clf_path=DEFAULT_SQUEEZENET_CLF_PATH):
        super().__init__()
        self.squeezenet_clf = SqueezeNet()
        self.squeezenet_clf.load_custom_model(squeezenet_clf_path)

    # override
    def classify(self, image, display=True, as_matlike=False):
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

        # if the Haar classifier detects a cockatiel, run the SqueezeNet classifier
        if haar_result["label"] == "cockatiel":
            snet_result = None

            # if the input image is not MatLike, pass the path to the image instead
            # of the processed 'frame'
            if not as_matlike:
                snet_result = self.squeezenet_clf.test(image)
            else:
                snet_result = self.squeezenet_clf.test(frame, as_matlike=True)

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
            "result": result_label
        }

    
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
