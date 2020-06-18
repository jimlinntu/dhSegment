from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import binarization, hysteresis_thresholding, cleaning_probs, line_vectorization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from skimage import io
import argparse
import numpy as np
import cv2
import tensorflow as tf
import time

class TextLineDetector():
    def __init__(self, model_dir, debug=True):
        self.sess = tf.Session()
        with self.sess.as_default():
            self.model = LoadedModel(model_dir, predict_mode="image")
        self.debug = debug
        self.debug_dir = Path("./dhSegment_debug/")
        self.debug_dir.mkdir(exist_ok=True)

    def __exit__():
        self.sess.close()

    def detect(self, img, model_predict_h_w=None):
        '''
            Input:
                img: a BGR image (np.ndarray)
            Return:
                cv2 style contours

            Reference:
                https://github.com/dhlab-epfl/fdh-tutorials/blob/master/computer-vision-deep-learning/3-applications/dl-document-processing-textlines/fdh_document_processing.ipynb
        '''
        assert isinstance(img, np.ndarray) and len(img.shape) == 3
        assert model_predict_h_w is None or isinstance(model_predict_h_w, tuple)

        with self.sess.as_default():
            # Deep Learning based textline detection
            start = time.time()
            # Note: the model takes RGB image as input
            output_textline = self.model.predict(img[:,:,[2,1,0]])
            if self.debug:
                print("[!] The model took {} to predict this image".format(time.time()-start))
            textline_probs = output_textline['probs'][0,:,:,1]
            if self.debug:
                plt.imshow(textline_probs)
                plt.savefig(str(self.debug_dir / "textline_probability.png"))

            textline_probs2 = cleaning_probs(textline_probs, sigma=2)
            textline_mask = hysteresis_thresholding(textline_probs2, low_threshold=0.3, high_threshold=0.6, candidates_mask=None)
            if self.debug:
                plt.imshow(textline_mask)
                plt.savefig(str(self.debug_dir / "textline_mask.png"))

            start = time.time()
            line_contours = line_vectorization.find_lines(resize(textline_mask, img.shape[0:2]))
            if self.debug:
                print("[!] Find lines took {} secs".format(time.time() - start))
            
            if self.debug:
                drawn_line_img = cv2.drawContours(img.copy(), line_contours, -1, (0, 255, 0), thickness=3)
                cv2.imwrite(str(self.debug_dir / "drawn_line_img.png"), drawn_line_img)
        
        return line_contours

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=Path)
    args = parser.parse_args()
    assert args.img_path.exists(), "{} is not exisiting".format(img_path)

    img = io.imread(args.img_path)
    img = img[:, :, [2,1,0]] # RGB -> BGR
    img = np.ascontiguousarray(img)
    textline_detector = TextLineDetector("demo/polylines", debug=True)
    
    start = time.time()
    line_contours = textline_detector.detect(img)
    print("[*] First prediction took {} secs".format(time.time() - start))
    start = time.time()
    # The second time will be faster
    line_contours = textline_detector.detect(img)
    print("[*] Second prediction took {} secs".format(time.time() - start))
