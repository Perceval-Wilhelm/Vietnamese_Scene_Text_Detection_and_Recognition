import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import sys
import json
import time
import copy

# Append paths for PaddleOCR imports
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, './tools')))

# Import PaddleOCR modules
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger

logger = get_logger()

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def load_image(image_file):
    img, flag_gif, flag_pdf = check_and_read(image_file)
    if not flag_gif and not flag_pdf:
        img = cv2.imread(image_file)
    return img


def generate_html_with_positions(image_path, dt_boxes, rec_res):
    img = Image.open(image_path)
    width, height = img.size
    html_content = f'<div style="position: relative; width: {width}px; height: {height}px;">\n'
    
    for box, (text, score) in zip(dt_boxes, rec_res):
        left = min(box[:, 0])
        top = min(box[:, 1])
        box_width = max(box[:, 0]) - left
        box_height = max(box[:, 1]) - top
        html_content += f'<div style="position: absolute; left: {left}px; top: {top}px; font-size: {box_height}px;">{text}</div>\n'
    
    html_content += '</div>'
    return html_content


def main(args):
    text_sys = TextSystem(args)

    st.title("Vietnamese Scene Text Detection and Recognition")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Save uploaded file
        image_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img = load_image(image_path)

        if img is not None:
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Detecting and recognizing text...")

            dt_boxes, rec_res, time_dict = text_sys(img)
            
            # Generate HTML with text positions
            html_content = generate_html_with_positions(image_path, dt_boxes, rec_res)
            
            # Visualize text on image
            result_img = draw_ocr_box_txt(
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                dt_boxes,
                [rec_res[i][0] for i in range(len(rec_res))],
                [rec_res[i][1] for i in range(len(rec_res))],
                drop_score=args.drop_score,
                font_path=args.vis_font_path)
            
            st.image(result_img, caption='Processed Image.', use_column_width=True)
            st.write("Detection and recognition completed.")
            st.write(f"Total time: {time_dict['all']:.3f} seconds")

            # Display the recognized text in the original positions
            st.markdown(html_content, unsafe_allow_html=True)

if __name__ == "__main__":
    args = utility.parse_args()  # Corrected to avoid TypeError
    args.use_gpu = False
    args.det_algorithm = "SAST"
    args.det_model_dir = "./inference/SAST/"
    args.rec_algorithm = "SVTR"
    args.rec_model_dir = "./inference/SVTRtiny/"
    args.rec_image_shape = "3,64,256"
    args.drop_score = 0.5
    args.vis_font_path = "font-times-new-roman.ttf"
    args.image_dir = "./images/"
    args.rec_char_dict_path = "./ppocr/utils/dict/vn_dictionary.txt"
    args.show_log = True  # or False, based on your preference
    
    main(args)
