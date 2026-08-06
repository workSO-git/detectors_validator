from abc import ABC, abstractmethod
import numpy as np

class BaseVideoModel(ABC):
    """
    Базовий клас для моделей, що обробляють відео (збереження стану).
    """
    
    @abstractmethod
    def reset_state(self):
        """
        Скидає внутрішній стан моделі (наприклад, історію трекінгу).
        Викликається перед початком обробки нового відео.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Обробляє один кадр відео і повертає результати.
        Модель має сама зберігати необхідний стан між викликами (напр. попередній кадр).
        
        Args:
            frame: поточний кадр відео (numpy array у форматі BGR).
            
        Returns:
            dict: словник з результатами:
            {
                'masks': [...], # Список бінарних масок
                'boxes': [...], # Список боксів [x1, y1, x2, y2]
                'classes': [...], # Класи об'єктів
                'points': [...] # (Опціонально) Список точок (x,y) для трекера
            }
        """
        pass
