# python flask 非流式部署

## 使用方法
```
将该文件目录放到wenet的根目录

需要先安装wenet的依赖
pip install -r requirements.txt

导入python的目录环境
. ./pypath.sh

gunicorn http启动
gunicorn -c config/gunicorn.py server:server

测试脚本
python3 bin/qps.py --concurrent=1 --sample-rate 8000 ${wav_dir} ${times} http://${ip}:${port}/
```