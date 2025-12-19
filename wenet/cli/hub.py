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
import shutil
import sys
import tarfile
import tempfile
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

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=temp_dir)
            contents = os.listdir(temp_dir)
            extracted_dir = os.path.join(temp_dir, contents[0])
            for item in os.listdir(extracted_dir):
                source_item = os.path.join(extracted_dir, item)
                dest_item = os.path.join(dest, item)
                if os.path.exists(dest_item):
                    if os.path.isdir(dest_item):
                        shutil.rmtree(dest_item)
                    else:
                        os.remove(dest_item)
                shutil.move(source_item, dest)
                print(f"Extract {source_item} to {dest}")

        except tarfile.TarError as e:
            print(f"Error during tar file extraction: {e}")
        except OSError as e:
            print(f"Error during file operation: {e}")


class Hub(object):
    """Hub for wenet pretrain model
    """
    # TODO(Binbin Zhang): make assets class to support more models
    assets = {
        "wenetspeech": "wenetspeech_u2pp_conformer_exp.tar.gz",
        "whiper-tiny": "whisper-tiny.tar.gz",
        "whiper-base": "whisper-base.tar.gz",
        "whiper-small": "whisper-small.tar.gz",
        "whiper-medium": "whisper-medium.tar.gz",
        "whisper-large-v3": "whisper-large-v3.tar.gz",
        "whisper-large-v3-turbo": "whisper-large-v3-turbo.tar.gz",
        "paraformer": "paraformer.tar.gz",
        "firered": "firered.tar.gz",
        "sensevoice_small": "sensevoice_small.tar.gz",
        "punc": "punc.tar.gz"
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def download_model(model_name: str) -> str:
        if model_name not in Hub.assets.keys():
            print('ERROR: Unsupported model {} !!!'.format(model_name))
            sys.exit(1)
        model = Hub.assets[model_name]
        model_dir = os.path.join(Path.home(), ".wenet", model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if set(["final.pt",
                "train.yaml"]).issubset(set(os.listdir(model_dir))):
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
