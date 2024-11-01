# Copyright (c) 2022  Mddct(hamddct@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
import tqdm


def download(url: str, dest: str, only_child=True):
    """ download from url to dest
    """
    assert os.path.exists(dest)
    print('Downloading {} to {}'.format(url, dest))

    def progress_hook(t):
        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            if tsize not in (None, -1):
                t.total = tsize
            displayed = t.update((b - last_b[0]) * bsize)
            last_b[0] = b
            return displayed

        return update_to

    # *.tar.gz
    name = url.split('?')[0].split('/')[-1]
    tar_path = os.path.join(dest, name)
    with tqdm.tqdm(unit='B',
                   unit_scale=True,
                   unit_divisor=1024,
                   miniters=1,
                   desc=(name)) as t:
        urlretrieve(url,
                    filename=tar_path,
                    reporthook=progress_hook(t),
                    data=None)
        t.total = t.n

    with tarfile.open(tar_path) as f:
        if not only_child:
            f.extractall(dest)
        else:
            for tarinfo in f:
                if "/" not in tarinfo.name:
                    continue
                name = os.path.basename(tarinfo.name)
                fileobj = f.extractfile(tarinfo)
                with open(os.path.join(dest, name), "wb") as writer:
                    writer.write(fileobj.read())


class Hub(object):
    """Hub for wenet pretrain runtime model
    """
    # TODO(Mddct): make assets class to support other language
    Assets = {
        # wenetspeech
        "chinese": "wenetspeech_u2pp_conformer_libtorch.tar.gz",
        # gigaspeech
        "english": "gigaspeech_u2pp_conformer_libtorch.tar.gz",
        # paraformer
        "paraformer": "paraformer.tar.gz",
        # punc
        "punc": "punc.tar.gz"
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_model_by_lang(lang: str) -> str:
        if lang not in Hub.Assets.keys():
            print('ERROR: Unsupported language {} !!!'.format(lang))
            sys.exit(1)

        # NOTE(Mddct): model_dir structure
        # Path.Home()/.wenet
        # - chs
        #    - units.txt
        #    - final.zip
        # - en
        #    - units.txt
        #    - final.zip
        model = Hub.Assets[lang]
        model_dir = os.path.join(Path.home(), ".wenet", lang)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # TODO(Mddct): model metadata
        if set(["final.zip",
                "units.txt"]).issubset(set(os.listdir(model_dir))):
            return model_dir
        # If not exist, download
        response = requests.get(
            "https://modelscope.cn/api/v1/datasets/wenet/wenet_pretrained_models/oss/tree"  # noqa
        )
        model_info = next(data for data in response.json()["Data"]
                          if data["Key"] == model)
        model_url = model_info['Url']
        download(model_url, model_dir, only_child=True)
        return model_dir
