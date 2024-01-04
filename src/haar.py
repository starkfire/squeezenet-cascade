import cv2
import os

DEFAULT_BIRD_CLF_PATH = "../pretrained/bird15stages.xml"
DEFAULT_COCKATIEL_CLF_PATH = "../pretrained/cockatiel15stages.xml"

class HaarCascadeClassifier:

    def __init__(self, bird_clf_path=DEFAULT_BIRD_CLF_PATH, cockatiel_clf_path=DEFAULT_COCKATIEL_CLF_PATH):
        self.bird_clf = self.import_model(bird_clf_path)
        self.cockatiel_clf = self.import_model(cockatiel_clf_path)

        # threshold
        # self.bird_threshold = 1.351643598615917e-05
        # self.cockatiel_threshold = 1.3819598955238318e-05
      
        # threshold (optimizer)
        self.bird_threshold = 3.0046271257736914e-05
        self.cockatiel_threshold = 1.814486863115111e-05

        self.roi_x = 100
        self.roi_y = 100
        self.roi_w = 500
        self.roi_h = 500

        # detectMultiScale parameters
        self.scaleFactor = 1.124
        self.minNeighbors = 10
        self.minSize = (224, 224)

        # bounding box parameters
        self.bird_bbox_color = (0, 255, 0)
        self.cockatiel_bbox_color = (0, 0, 255)

    
    def import_model(self, model_path):
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, model_path)
        return cv2.CascadeClassifier(filepath)

    
    def get_roi(self, image, x, y, w, h):
        return image[y:y+h, x:x+w]


    def detect(self, classifier, frame):
        return classifier.detectMultiScale(frame,
                                           scaleFactor=self.scaleFactor,
                                           minNeighbors=self.minNeighbors,
                                           minSize=self.minSize)


    def get_detections(self, label, frame, outputs):
        confidence_scores = []

        for (x, y, w, h) in outputs:
            # roi = frame[y:y+h, x:x+w]
            confidence = len(outputs) / (w * h)

            # bbox_color = self.cockatiel_bbox_color if label.lower() == "cockatiel" else self.bird_bbox_color
            # threshold = self.cockatiel_threshold if label.lower() == "cockatiel" else self.bird_threshold

            # if confidence >= threshold:
            # cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(roi, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.68, bbox_color, 2)
            confidence_scores.append(confidence)

        return {
            "scores": confidence_scores
        }


    def mean(self, numbers):
        return sum(numbers) / len(numbers)


    def parse_result(self, bird_scores, cockatiel_scores): 
        if len(bird_scores) > len(cockatiel_scores):
            return "bird"
        elif len(cockatiel_scores) > len(bird_scores):
            return "cockatiel"
        else:
            return "cockatiel" if self.mean(cockatiel_scores) < self.mean(bird_scores) else "bird"


    def classify(self, image, display=True):
        frame = cv2.imread(image)
        roi_frame = frame[self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        birds = self.detect(self.bird_clf, gray)
        cockatiels = self.detect(self.cockatiel_clf, gray)

        # print(birds)
        # print(cockatiels)

        bird_det = self.get_detections("Bird", gray, birds)
        cockatiel_det = self.get_detections("Cockatiel", gray, cockatiels)

        if display:
            self.display_cv_image(frame)

        results = {
            "frame": frame,
            "bird_scores": bird_det["scores"],
            "cockatiel_scores": cockatiel_det["scores"],
            "result": self.parse_result(bird_det["scores"], cockatiel_det["scores"])
        }

        return results


    def display_cv_image(self, image_mat):
        """
        Method to display resulting image matrices using OpenCV
        """
        cv2.imshow('image', image_mat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
