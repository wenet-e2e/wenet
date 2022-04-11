# wenet-binding
binding wrapper for runtime

## Highlights:

* go binding
* python binding
* streamming and nonstreamming decoding
* c interface you can binding for other language
* some instresting tools eg: label_check

## example:
- [go example](go/README.md)
- [python example](python/README.md)

## build
``` bash
# docker image: quay.io/pypa/manylinux2014_x86_64
# in runtime/binding

PLAT=manylinux1_x86_64
for PYTHON in `ls /opt/python | xargs`; do
    mkdir build${PYTHON} && cd build${PYTHON} \
        && cmake -D PYTHON_EXECUTABLE=/opt/python/${PYTHON}/bin/python \
                 -D PYTHON_TAG=`echo ${PYTHON} | cut -d "-" -f 1` ../ \
        && cmake --build . \
        && cd -
done;


function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w wheelhouse/
    fi
}

mkdir python/wheelhouse &&  mv python/wenet/dist/*  python/wheelhouse

for whl in python/wheelhouse/*.whl; do
    repair_wheel "$whl"
done

twine upload wheelhouse/*

```
