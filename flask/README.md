# python flask 非流式部署

## 使用方法
```
将该文件目录放到wenet的根目录

需要先安装wenet的依赖
pip install -r requirements.txt

导入python的目录环境
. ./pypath.sh

需要修改config下的配置文件相应选项
chrome录音需要用https服务，所以需要ssl证书
python3 server.py

gunicorn http启动
gunicorn -c config/gunicorn.py server:server
```