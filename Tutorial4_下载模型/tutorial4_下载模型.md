# Tutorial4: 下载模型

* 集群类型：超算平台
* 所需镜像：无
* 所需模型：无
* 所需数据集：无
* 所需资源：
* 目标：本节旨在使用超算平台展示如何下载大模型 [Qwen3-4B] (https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507) 。

## 1、使用超算平台下载模型

1.1.1 登录SCOW平台，选取超算平台，进入HPC集群
![alt text](assets/image.png)

1.1.2 点击 Shell > 选择集群（这里Ascend-CranSched是HPC集群的名称） > 选择节点（这里login01是节点的名称） ，连接到集群的login节点，等待十几秒左右，能看到
![alt text](assets/image-17.png)

1.1.3 拷贝命令 pwd 粘贴到界面，并按 回车键，确保在正确的目录下：/data/home/你的用户名
![alt text](assets/image-18.png)

1.1.4 拷贝命令 mkdir model 粘贴到界面，并按 回车键，这样就在当前目录下新创建了一个名为 model 的目录，下载的模型都可以统一放在这个目录下面

1.1.4 拷贝命令 cd model 粘贴到界面，并按 回车键，这样就进入到刚新创建的名为 model 的目录里
![alt text](assets/image-19.png)

1.1.5 拷贝命令 pip install modelscope 粘贴到界面，并按 回车键。
这里是安装了modelscope工具，此工具由模型下载的镜像网站提供

1.1.6 拷贝命令 modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ./Qwen/Qwen/Qwen3-4B-Instruct-2507 粘贴到界面，并按 回车键。
这里是通过刚安装的modelscope这个工具去镜像网站下载模型 Qwen3-4B-Instruct-2507

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ./Qwen/Qwen/Qwen3-4B-Instruct-2507
```
