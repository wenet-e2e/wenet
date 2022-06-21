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
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

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
    name = url.split("/")[-1]
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
        "chs":
        "https://github.com/wenet-e2e/wenet/releases/download/v2.0.1/chs.tar.gz",
        # gigaspeech
        "en":
        "https://github.com/wenet-e2e/wenet/releases/download/v2.0.1/en.tar.gz"
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_model_by_lang(lang: str) -> str:
        assert lang in Hub.Assets.keys()
        # NOTE(Mddct): model_dir structure
        # Path.Home()/.went
        # - chs
        #    - units.txt
        #    - final.zip
        # - en
        #    - units.txt
        #    - final.zip
        model_url = Hub.Assets[lang]
        model_dir = os.path.join(Path.home(), ".wenet", lang)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # TODO(Mddct): model metadata
        if set(["final.zip",
                "units.txt"]).issubset(set(os.listdir(model_dir))):
            return model_dir
        download(model_url, model_dir, only_child=True)
        return model_dir
