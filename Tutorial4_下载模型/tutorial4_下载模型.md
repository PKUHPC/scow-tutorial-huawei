# Tutorial4: 下载模型

* 集群类型：超算平台
* 所需镜像：无
* 所需模型：无
* 所需数据集：无
* 所需资源：
* 目标：本节旨在使用超算平台展示如何下载大模型 [Qwen3-4B] (https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507) 。

使用本教程前请确保超算集群与智算集群的文件管理系统是互通的，如果不是互通的那么`章节2`中智算集群创建模型时就无法用到在`章节1`中超算集群的shell下载的模型，此时你请参考`章节0`的方法在智算集群中下载模型并跳过`章节1`，如果是互通的则跳过`章节0`即可。

## 0、使用智算平台下载模型

登录SCOW平台，选取智算平台

![alt text](assets/assets/image.png)

点击作业->应用进入创建应用界面，接着点击VSCode应用

![alt text](assets/assets/image-1.png)

进到创建VSCode中，分别填写以下内容：
* 镜像源选择远程镜像
* 远程镜像地址填写 `app-store-images.pku.edu.cn/ascend/cann:8.1.rc1-910b-openeuler22.03-py3.10`
* 修改默认命令填写 `${SCOW_AI_ALGORITHM_PATH}/bin/code-server`
* 添加算法，选择公共算法->code-server->4.95.3
* 最长运行时间按需填写，需要大于模型预估下载时间

其余维持不变，点击提交

![alt text](assets/assets/image-2.png)

在跳转后的页面点击进入进入vscode应用并打开终端

![alt text](assets/assets/image-3.png)
![alt text](assets/assets/image-4.png)

在终端中依次运行以下命令
```shell
pip install modelscope
tmux new -s tutorial4  # 建立tmux会话
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir $WORK_DIR/Qwen3-4B-Instruct-2507
# 按ctrl+d再单按b退出
tmux kill-session -t tutorial4  # 删除tmux会话
```

运行命令 `echo $WORK_DIR/Qwen3-4B-Instruct-2507` 查看安装模型的路径，其中红框框起来的部分就是模型的绝对路径

![alt text](assets/assets/image-5.png)

回到集群主页，点击文件进入文件系统

![alt text](assets/assets/image-6.png)

在地址栏中填写模型的绝对路径按回车，就能够看到模型文件

![alt text](assets/assets/image-7.png)

请记住模型的绝对路径，在章节2中选择模型文件时会使用到

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

## 2、使用智算平台管理模型
2.1 选取智算平台，进入AI集群

2.2 点击 模型 > 我的模型，进入提交作业页面
![alt text](assets/assets/image-9.png)

2.3 点击添加模型
![alt text](assets/assets/image-10.png)

名称 QWen；集群 下拉菜单 ascend-k8s。
因为刚下载模型的是 Qwen3-4B-Instruct-2507，这里取的名称为QWen。可以根据下载的模型的名称进行命名。
![alt text](assets/assets/image-11.png)

2.4 给模型添加版本
点击 + 号，给刚下载的模型创建版本号
![alt text](assets/assets/image-12.png)

因为刚下载模型的是 Qwen3-4B-Instruct-2507，这里的版本名称取为 Qwen3-4B，在选择模型这里点击右边的图标
![alt text](assets/assets/image-13.png)

进入文件选择页面，在左侧找到刚创建的 model 目录，点击打开 QWen，在右边文件名中找到并选中 Qwen3-4B-Instruct-2507，点击右下角的 确认
![alt text](assets/assets/image-14.png)

回到新建版本页面，此时可以看到 选择模型 中已经选好 Qwen3-4B-Instruct-2507模型，点击右下角的 确认
![alt text](assets/assets/image-15.png)

回到我的模型页面，点击 QWen 前面的 +号，可以看到模型的新版本 Qwen3-4B 已经创建成功
![alt text](assets/assets/image-16.png)

---
> 作者：孔德硕；龙汀汀*
>
> 联系方式：l.tingting@pku.edu.cn