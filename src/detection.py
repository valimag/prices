import onnxruntime
import cv2
import numpy as np


class NumbersDetection():

    def __init__(self):
        self.session_det = onnxruntime.InferenceSession('models/detector_model.onnx',providers=['CUDAExecutionProvider'])
        model_inputs_det = self.session_det.get_inputs()
        self.input_names_det = [model_inputs_det[i].name for i in range(len(model_inputs_det))] 
        model_outputs_det = self.session_det.get_outputs()
        self.output_names_det = [model_outputs_det[i].name for i in range(len(model_outputs_det))]
        self.detection_width, self.detection_height = 128, 128

        
    def pred(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128)) 
        image = image.astype(np.float32) / 255.0
        input_image = np.expand_dims(image, axis=0)
        return input_image
    
    def post(self, y_pred, detection_threshold=0.7,text_threshold=0.4,link_threshold=0.4,size_threshold=10):
        box_groups = []
        for y_pred_cur in y_pred:
            textmap = y_pred_cur[..., 0].copy()
            linkmap = y_pred_cur[..., 1].copy()
            img_h, img_w = textmap.shape
            _, text_score = cv2.threshold(
                textmap, thresh=text_threshold, maxval=1, type=cv2.THRESH_BINARY
            )
            _, link_score = cv2.threshold(
                linkmap, thresh=link_threshold, maxval=1, type=cv2.THRESH_BINARY
            )
            n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
                np.clip(text_score + link_score, 0, 1).astype("uint8"), connectivity=4
            )
            boxes = []
            for component_id in range(1, n_components):
                size = stats[component_id, cv2.CC_STAT_AREA]

                if size < size_threshold:
                    continue

                if np.max(textmap[labels == component_id]) < detection_threshold:
                    continue
                segmap = np.zeros_like(textmap)
                segmap[labels == component_id] = 255
                segmap[np.logical_and(link_score, text_score)] = 0
                x, y, w, h = [
                    stats[component_id, key]
                    for key in [
                        cv2.CC_STAT_LEFT,
                        cv2.CC_STAT_TOP,
                        cv2.CC_STAT_WIDTH,
                        cv2.CC_STAT_HEIGHT,
                    ]
                ]

                niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
                sx, sy = max(x - niter, 0), max(y - niter, 0)
                ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
                segmap[sy:ey, sx:ex] = cv2.dilate(
                    segmap[sy:ey, sx:ex],
                    cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)),)

                contours = cv2.findContours(
                    segmap.astype("uint8"),
                    mode=cv2.RETR_TREE,
                    method=cv2.CHAIN_APPROX_SIMPLE)[-2]
                contour = contours[0]
                box = cv2.boxPoints(cv2.minAreaRect(contour))
                w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
                box_ratio = max(w, h) / (min(w, h) + 1e-5)
                if abs(1 - box_ratio) <= 0.1:
                    l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                    t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                    box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
                else:
                    box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
                boxes.append(2 * box)
                box_groups.append(np.array(boxes))
            return box_groups
        
    def __call__(self,image):
        input_image = self.pred(image)
        detections = self.session_det.run(self.output_names_det, {self.input_names_det[0]: input_image})
        output = detections[0]  
        boxes = self.post(output)

        image_to_cut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_width = image_to_cut.shape[1]
        original_height = image_to_cut.shape[0]
        scale_x = original_width / self.detection_width
        scale_y = original_height / self.detection_height
        scaled_boxes = []
        for box_group in boxes:
            scaled_group = []
            for box in box_group:
                scaled_box = np.zeros_like(box)
                for i, (x, y) in enumerate(box):
                    scaled_box[i] = [x * scale_x, y * scale_y]
                scaled_group.append(scaled_box)
            scaled_boxes.append(scaled_group)

        cropped_images = []
        for box_group in scaled_boxes:
            for box in box_group:
                rect = cv2.boundingRect(box.astype(int))
                x, y, w, h = rect
                cropped_image = image_to_cut[y:y+h, x:x+w]
                cropped_images.append(cropped_image)
        return cropped_images