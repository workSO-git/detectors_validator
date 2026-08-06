from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def predict(self, image_path_or_array, conf_threshold=0.25):
        """
        Takes an image path or numpy array and returns a standardized dictionary:
        {
            'masks': [...], # List of numpy binary masks (height, width)
            'boxes': [...], # List of bounding boxes [x1, y1, x2, y2]
            'classes': [...] # List of class IDs
        }
        """
        pass
    
    @abstractmethod
    def predict_and_save(self, image_path, save_dir):
        """
        Run prediction on a single instance and save the visualization to save_dir.
        """
        pass
