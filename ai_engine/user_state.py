from dataclasses import dataclass
from ai_engine.config import READING_SPEED_WPS, IMG_EXTRA_FIXED_TIME


@dataclass
class UserStateConfig:
    reading_speed_wps: int = READING_SPEED_WPS
    img_extra_fixed_time: float = IMG_EXTRA_FIXED_TIME


class UserState:

    def __init__(self, config: UserStateConfig = UserStateConfig()):
        self.config = config

    def is_interaction_successful(self, dwell_time: float, estimated_reading_time: float) -> bool:
        """
        Determine if user interaction is successful based on dwell time
        compared to estimated reading time.
        """
        return dwell_time >= estimated_reading_time
    
    def compute_reading_time(self, content_length_words: int, has_image: bool = False) -> float:
        """
        Estimate reading time for given content length in words.
        If content has an image, add extra fixed time.
        """
        base_time = content_length_words / self.config.reading_speed_wps
        if has_image:
            base_time += self.config.img_extra_fixed_time
        return base_time
    
