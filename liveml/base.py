import abc
from typing import List, Tuple, Optional, Generator
import numpy as np
import cv2


class _BaseDetector(abc.ABC):

    @abc.abstractmethod
    def _resize_image(self, image: np.ndarray):
        pass

    @abc.abstractmethod
    def init_session(self):
        pass

    @abc.abstractmethod
    def close_session(self):
        pass

    @abc.abstractmethod
    def _run_inference(self, image: np.ndarray):
        pass

    @abc.abstractmethod
    def _detect_on_image(self, image: np.ndarray):
        pass

    @abc.abstractmethod
    def detect_on_images(self, *images: List[np.ndarray]):
        pass

    @abc.abstractmethod
    def _visualize(self, image: np.ndarray, detections: dict):
        pass

    @abc.abstractmethod
    def visualize_detection_on_images(self, *images: List[np.ndarray]):
        pass


class BaseDetector(_BaseDetector):

    def __init__(self, model_image_size: Optional[Tuple[int, int]] = None) -> None:
        self._model_image_size = model_image_size

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self._model_image_size is not None:
            image = cv2.resize(image, self._model_image_size, cv2.INTER_AREA)
        return image

    def init_session(self):
        pass

    def close_session(self):
        pass

    def _run_inference(self, image: np.ndarray) -> dict:
        return {}

    def _detect_on_image(self, image: np.ndarray) -> dict:
        resized_image = self._resize_image(image)
        return self._run_inference(resized_image)

    def detect_on_images(self, *images: List[np.ndarray]) -> Generator:
        for image in images:
            yield self._detect_on_image(image)

    def _visualize(self, image: np.ndarray, detections: dict) -> np.ndarray:
        return image

    def visualize_detection_on_images(self, *images: List[np.ndarray]) -> Generator:
        for image in images:
            detection = self._detect_on_image(image)
            yield self._visualize(image, detection)
