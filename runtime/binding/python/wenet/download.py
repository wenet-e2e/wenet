import io
import os
import tarfile

import requests
import tqdm


def download(url=None, parent_dir=''):

    subdir = os.path.join(parent_dir, os.path.basename(url).split(".")[0])
    if os.path.exists(subdir):
        return subdir

    os.makedirs(subdir)

    response = requests.get(url, stream=True)
    response_length = int(response.headers['Content-Length']) / 1024 / 1024# Don't trust response.length!
    model_bytes = bytes()
    for data in tqdm.tqdm(iterable=response.iter_content(1024*1024),unit="MB",total=response_length,desc="model"):
        model_bytes += data

    with io.BytesIO(model_bytes) as model_fobj:
        with tarfile.open(fileobj=model_fobj,  mode="r|*") as model_stream:
            for tarinfo in model_stream:
                name = tarinfo.name
                if name.endswith(".zip") or name.endswith("words.txt"):
                    with model_stream.extractfile(tarinfo) as file_obj:
                        with open(os.path.join(subdir, os.path.basename(name)), "wb") as f:
                            f.write(file_obj.read())

    return subdir
