import os
from typing import List

import jieba
import torch
from wenet.cli.hub import Hub
from wenet.paraformer.search import _isAllAlpha
from wenet.text.char_tokenizer import CharTokenizer


class PuncModel:

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')

        self.model = torch.jit.load(model_path)
        self.tokenizer = CharTokenizer(units_path)
        self.device = torch.device("cpu")
        self.use_jieba = False

        self.punc_table = ['<unk>', '', '，', '。', '？', '、']

    def split_words(self, text: str):
        if not self.use_jieba:
            self.use_jieba = True
            import logging

            # Disable jieba's logger
            logging.getLogger('jieba').disabled = True
            jieba.load_userdict(os.path.join(self.model_dir, 'jieba_usr_dict'))

        result_list = []
        tokens = text.split()
        current_language = None
        buffer = []

        for token in tokens:
            is_english = token.isascii()
            if is_english:
                language = "English"
            else:
                language = "Chinese"

            if current_language and language != current_language:
                if current_language == "Chinese":
                    result_list.extend(jieba.cut(''.join(buffer), HMM=False))
                else:
                    result_list.extend(buffer)
                buffer = []

            buffer.append(token)
            current_language = language

        if buffer:
            if current_language == "Chinese":
                result_list.extend(jieba.cut(''.join(buffer), HMM=False))
            else:
                result_list.extend(buffer)

        return result_list

    def add_punc_batch(self, texts: List[str]):
        batch_text_words = []
        batch_text_ids = []
        batch_text_lens = []

        for text in texts:
            words = self.split_words(text)
            ids = self.tokenizer.tokens2ids(words)
            batch_text_words.append(words)
            batch_text_ids.append(ids)
            batch_text_lens.append(len(ids))

        texts_tensor = torch.tensor(batch_text_ids,
                                    device=self.device,
                                    dtype=torch.int64)
        texts_lens_tensor = torch.tensor(batch_text_lens,
                                         device=self.device,
                                         dtype=torch.int64)

        log_probs, _ = self.model(texts_tensor, texts_lens_tensor)
        result = []
        outs = log_probs.argmax(-1).cpu().numpy()
        for i, out in enumerate(outs):
            punc_id = out[:batch_text_lens[i]]
            sentence = ''
            for j, word in enumerate(batch_text_words[i]):
                if _isAllAlpha(word):
                    word = '▁' + word
                word += self.punc_table[punc_id[j]]
                sentence += word
            result.append(sentence.replace('▁', ' '))
        return result

    def __call__(self, text: str):
        if text != '':
            r = self.add_punc_batch([text])[0]
            return r
        return ''


def load_model(model_dir: str = None,
               gpu: int = -1,
               device: str = "cpu") -> PuncModel:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang('punc')
    if gpu != -1:
        # remain the original usage of gpu
        device = "cuda"
    punc = PuncModel(model_dir)
    punc.device = torch.device(device)
    punc.model.to(device)
    return punc
