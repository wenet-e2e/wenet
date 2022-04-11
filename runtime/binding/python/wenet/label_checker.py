import wave
from typing import Generator, List

from wenet.lib._pywrap_wenet import (LabelCheckerWrapper, Params,
                                     SimpleAsrModelWrapper)
from wenet.model import WenetWrapper


class LabelChecker:
    """LabelChecker is a class for check labels
    """
    def __init__(self,
                 model : WenetWrapper,
                 is_penalty :float = 3.0,
                 del_penalty :float = 4.0):

        self.checker = LabelCheckerWrapper(model.model)

        self.is_penalty = is_penalty
        self.del_penaltu = del_penalty

    def check(
            self,
            filepath: str,
            labels: List[str]) -> str:

        """

        Args:
            filepath (path-like object or file-like object):
        Returns:
            str:
        """
        if not isinstance(labels, list):
            raise ValueError("labels must be tokens list")
        with wave.open(filepath, "rb") as f:
            assert f.getnchannels() == 1
            # TODO: sample_rate from params
            assert f.getframerate() == 16000
            length = int(f.getnframes())
            wav_bytes = f.readframes(length)

        return self.checker.Check(
            wav_bytes,
            length,
            labels,
            self.is_penalty,
            self.del_penaltu)
