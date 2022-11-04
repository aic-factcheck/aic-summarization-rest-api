from typing import List, Set, Dict, Tuple, Optional
from abc import ABC, abstractmethod

class AbstractSummarizer(ABC):
    @abstractmethod
    def __init__(self, model_name, lang):
        pass
    
    @abstractmethod
    def prepare_input_batch(self, batch: List[str]):
        pass