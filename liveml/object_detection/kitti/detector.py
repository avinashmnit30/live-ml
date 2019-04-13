# 3rd-party imports
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
import numpy as np


# local imports
from ...base import BaseDetector


class Detector(BaseDetector):

    def __init__(
            self,
            model_path,
            labels_path,
            min_score_threshold: int = 0.5
    ):
        super().__init__()
        self._model_path = model_path
        self._category_index = label_map_util.create_category_index_from_labelmap(labels_path,
                                                                                  use_display_name=True)
        self._detection_graph = None
        self._session = None
        self._tensor_dict = None
        self._image_tensor = None
        self._min_score_threshold = min_score_threshold

    def _init_model(self):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._detection_graph = detection_graph

    def init_session(self):
        self._init_model()
        with self._detection_graph.as_default():
            self._session = tf.Session()
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                    # Reframe is required to translate mask from box coordinates to image coordinates
                    # and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                self._tensor_dict = tensor_dict
                self._image_tensor = image_tensor

    def close_session(self):
        self._session.close()

    def _run_inference(self, image):
        # Run inference
        output_dict = self._session.run(self._tensor_dict,
                                        feed_dict = {self._image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def _visualize(self, image, detections):
        # Visualization of the results of a detection.
        result_image = image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            result_image,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self._category_index,
            instance_masks = detections.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=self._min_score_threshold)
        return result_image
