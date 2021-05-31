# python flask 非流式部署

## 使用方法
```
将该文件目录放到wenet的根目录

需要先安装wenet的依赖
pip install -r requirements.txt

导入python的目录环境
. ./pypath.sh

配置文件
1.首先需要修改模型配置文件train.yaml中global_cmvn的path
2.修改config/config.py中的模型路径checkpoint,yaml_path,vocab_path
3.修改config/config.py中的port，用python3 server.py测试是否能正常启动
4.修改config/gunicorn.py中bind的ip和port为自己需要的地址，并且修改对应参数为自己需要的配置参数，比如workers为启动的服务实例数


gunicorn http启动
gunicorn -c config/gunicorn.py server:server

测试脚本
python3 bin/qps.py --concurrent=1 --sample-rate ${sample_rate} ${wav_dir} ${times} http://${ip}:${port}/
```
