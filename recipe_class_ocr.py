import os
import traceback
import pdb
import subprocess
import shutil
import json
import base64
import requests

import pandas as pd
import numpy as np

class OCR_local():
    def __init__(self, ):
        self.max_x_dist = 800
        self.min_y_overlap_ratio = 0.5

    def is_on_same_line(self, box_a, box_b, min_y_overlap_ratio=0.8):
        """Check if two boxes are on the same line by their y-axis coordinates.
        Two boxes are on the same line if they overlap vertically, and the length
        of the overlapping line segment is greater than min_y_overlap_ratio * the
        height of either of the boxes.
        Args:
            box_a (list), box_b (list): Two bounding boxes to be checked
            min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                        allowed for boxes in the same line
        Returns:
            The bool flag indicating if they are on the same line
        """
        a_y_min = np.min(box_a[1::2])
        b_y_min = np.min(box_b[1::2])
        a_y_max = np.max(box_a[1::2])
        b_y_max = np.max(box_b[1::2])

        # Make sure that box a is always the box above another
        if a_y_min > b_y_min:
            a_y_min, b_y_min = b_y_min, a_y_min
            a_y_max, b_y_max = b_y_max, a_y_max

        if b_y_min <= a_y_max:
            if min_y_overlap_ratio is not None:
                sorted_y = sorted([b_y_min, b_y_max, a_y_max])
                overlap = sorted_y[1] - sorted_y[0]
                min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
                min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
                return overlap >= min_a_overlap or \
                    overlap >= min_b_overlap
            else:
                return True
        return False

    def stitch_boxes_into_lines(self, boxes, max_x_dist=20, min_y_overlap_ratio=0.5):

        """Stitch fragmented boxes of words into lines.
        Note: part of its logic is inspired by @Johndirr
        (https://github.com/faustomorales/keras-ocr/issues/22)
        Args:
            boxes (list): List of ocr results to be stitched
            max_x_dist (int): The maximum horizontal distance between the closest
                        edges of neighboring boxes in the same line
            min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                        allowed for any pairs of neighboring boxes in the same line
        Returns:
            merged_boxes(list[dict]): List of merged boxes and texts
        """


        if len(boxes) <= 1:
            return boxes

        merged_boxes = []

        # sort groups based on the x_min coordinate of boxes
        x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
        # store indexes of boxes which are already parts of other lines
        skip_idxs = set()

        i = 0
        # locate lines of boxes starting from the leftmost one
        for i in range(len(x_sorted_boxes)):
            if i in skip_idxs:
                continue
            # the rightmost box in the current line
            rightmost_box_idx = i
            line = [rightmost_box_idx]
            for j in range(i + 1, len(x_sorted_boxes)):
                if j in skip_idxs:
                    continue
                if self.is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                                x_sorted_boxes[j]['box'],
                                min_y_overlap_ratio):
                    line.append(j)
                    skip_idxs.add(j)
                    rightmost_box_idx = j
            
            # # ------------------------------------------------------------------------------------------------
            # print()
            # print(f"x_sorted_boxes[{line[0]}]['text'] = {x_sorted_boxes[line[0]]['text']}")
            # print(f"line = {line}")
            # for line_elem in line:
            #     print(f"\tx_sorted_boxes[{line_elem}]['text'] = {x_sorted_boxes[line_elem]['text']}")
            # print()
            # # ------------------------------------------------------------------------------------------------

            # split line into lines if the distance between two neighboring
            # sub-lines' is greater than max_x_dist
            lines = []
            line_idx = 0
            lines.append([line[0]])
            for k in range(1, len(line)):
                prev_box = x_sorted_boxes[line[k - 1]]
                curr_box = x_sorted_boxes[line[k]]
                dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
                # print(f"dist={dist} > {max_x_dist}=max_x_dist")
                if dist > max_x_dist:
                    line_idx += 1
                    lines.append([])
                lines[line_idx].append(line[k])

            # # ------------------------------------------------------------------------------------------------
            # print()
            # print(f"x_sorted_boxes[{line[0]}]['text'] = {x_sorted_boxes[line[0]]['text']}")
            # print(f"lines = {line}")
            # for line_elem in line:
            #     print(f"\tx_sorted_boxes[{line_elem}]['text'] = {x_sorted_boxes[line_elem]['text']}")
            # print()
            # # ------------------------------------------------------------------------------------------------
            
            # Get merged boxes
            for box_group in lines:
                merged_box = {}
                merged_box['text'] = ' '.join(
                    [x_sorted_boxes[idx]['text'] for idx in box_group])
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')
                for idx in box_group:
                    x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                    x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                    y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                    y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
                merged_box['box'] = [
                    x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
                ]
                merged_boxes.append(merged_box)

            # # ------------------------------------------------------------------------------------------------
            # print()
            # print(f"len(merged_boxes) = {len(merged_boxes)}")
            # for merged_boxes_elem in merged_boxes:
            #     print(f"\tmerged_boxes_elem['text'] = {merged_boxes_elem['text']}")
            # print()
            # # ------------------------------------------------------------------------------------------------

            # # ------------------------------------------------------------------------------------------------
            # pdb.set_trace()
            # # ------------------------------------------------------------------------------------------------
        return merged_boxes

    def prediction_to_text(self, ocr_prediction):
        # convert to df
        ocr_prediction_df = pd.DataFrame(ocr_prediction['predictions'][0])

        # merge
        text_boxes_lst = []
        for idx, row in ocr_prediction_df.iterrows():
            raw_text = row['rec_texts']
            ocr_texts = raw_text.replace('<UKN>', '').strip()
            if ocr_texts == '':
                continue
            text_boxes_lst.append({
                'text': ocr_texts,
                'box': row['det_polygons'],
            })

        # stitch
        merged_boxes = self.stitch_boxes_into_lines(text_boxes_lst,
                                                    max_x_dist = self.max_x_dist,
                                                    min_y_overlap_ratio = self.min_y_overlap_ratio)

        # extract the text
        ocr_text_lst = []
        for merged_details in merged_boxes:
            current_text = merged_details['text']
            ocr_text_lst.append(current_text)

        # merge to texts
        ocr_text = '\n'.join(ocr_text_lst)

        return ocr_text

    def run_ocr(self, image_path, DONT_SKIP=True):
        destination_image, ocr_prediction = image_path, "failed, try again"
        print(f"image_path = {image_path}")

        # Define source and destination paths
        destination_path = "/mnt/c/Local_LL/Coding/demo/v4/forRAGcipe/input.png"
        # Copy the file
        shutil.copy(image_path, destination_path)
        print(f'copied\n\t{image_path}\n\t{destination_path}')

        # Run OCR
        python_cmd = ""
        python_cmd += "export KMP_DUPLICATE_LIB_OK=TRUE"
        python_cmd += " && cd /c/Local_LL/Coding/demo/v4/forRAGcipe"
        python_cmd += " && conda activate demo_v4"
        python_cmd += " && python run_ocr.py"
        command = f'"/mnt/c/Program Files/Git/git-bash.exe" -c "{python_cmd}"'
        print(f"\tRunning {python_cmd} ...")
        if DONT_SKIP:
            subprocess.run(command, shell=True)
        print(f"\tdone")

        # Copy the prediction
        source_dir = "/mnt/c/Local_LL/Coding/demo/v4/forRAGcipe/results"
        destination_dir = "./data/ocr/tmp"
        if DONT_SKIP:
            print(f'remove {destination_dir}')
            shutil.rmtree(destination_dir, ignore_errors=True)
        destination_path = f"{destination_dir}/ocr_prediction.json"
        destination_image = f"{destination_dir}/vis/input.jpg"

        # Copy the file
        if DONT_SKIP:
            shutil.copytree(source_dir, destination_dir)
        print(f'copied\n\t{source_dir}\n\t{destination_path}')

        # read the file
        ocr_prediction = json.load(open(destination_path, 'r'))

        # post process it
        ocr_texts = self.prediction_to_text(ocr_prediction)

        return destination_image, ocr_texts


class OCR():
    def __init__(self, ):
        self.openai_chat_completions_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        self.get_recipe_prompt = "Describe what’s the recipe in this image?"

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def prediction_to_text(self, response):
        response_dict = response.json()
        message_dict = response_dict['choices'][0]['message']
        return message_dict['content']

    def run_ocr(self, image_path, DONT_SKIP=True):
        print(f"run_ocr(self, {image_path})")
        # Getting the base64 string
        base64_image = self.encode_image(image_path)

        # Construct payload
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": self.get_recipe_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 300
        }

        # Send to openai
        response = requests.post(self.openai_chat_completions_url,
                                 headers=self.headers,
                                 json=payload)

        # Extract the texts
        ocr_texts = self.prediction_to_text(response)
        print(f"ocr_texts = {ocr_texts}")

        return image_path, ocr_texts
