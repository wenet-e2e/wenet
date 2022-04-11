import threading
from typing import Generator, Iterator, Tuple

from wenet.lib._pywrap_wenet import (Params, SimpleAsrModelWrapper,
                                     StreammingAsrWrapper)
from wenet.model import WenetWrapper


class StreammingAsrDecoder:
    '''StreammingAsrDecoder is a class for
    streamming decoding given a model
    '''
    def __init__(self,
                 model : WenetWrapper,
                 nbest :int = 1,
                 continuous_decoding :bool= False):
        self.streamming_decoder = StreammingAsrWrapper(model.model, nbest, continuous_decoding)
        self.continuous_decoding = continuous_decoding
        self.nbest = nbest
        self.final = False

        self.has_data = False

    def streamming_recognize(self, pcm_generator: Generator):
        """

        Args:
            pcm_generator (Generator): (bytes:chunk_bytes, bool:is_last_chunk)
            # TODO(Mddct): make this function async not thread
        Returns:
            None: nothing
        """
        def send():
            while True:
                nbytes, is_final = pcm_generator
                self._acceptWaveform(nbytes,is_final)

                if is_final:
                     break

        t = threading.Thread(target=send)
        t.start()
        for res in self.streamming_decoder.GetInstanceResult():
            yield res

        t.join()

    def _acceptWaveform(self,
                        pcm :bytes,
                        final :bool=True):
        if len(pcm) == 0:
            return
        # when GetInstanceResult get final result, then must call Reset and then call AcceptWaveform
        if self.final:
            return
        self.has_data = True
        self.streamming_decoder.AcceptWaveform(pcm, int(len(pcm)/2), final)

    def _getInstanceResult(self) -> Generator:
        """
        one thread: decoder.AcceptWaveform
        another thread: get result
        for res in decoder.GetInstanceResult:
            print(res)
        Returns:
            Generator:
        """
        assert (self.has_data is True), "no data received"
        result = ""
        final = False
        while not final:
            result, final = self.streamming_decoder.GetInstanceResult()
            yield result

        self.final = True
        self._reset()

    def _reset(self, nbest :int =1 , continuous_decoding :bool = False):
        self.final = False
        self.has_data = False
        self.streamming_decoder.Reset(nbest, continuous_decoding)
        self.nbest = nbest
        self.continuous_decoding = continuous_decoding
