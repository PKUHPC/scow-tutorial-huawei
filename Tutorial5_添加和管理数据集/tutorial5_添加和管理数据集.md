# Tutorial5: 添加和管理数据集

* 集群类型：智算平台
* 所需镜像：无
* 所需模型：无
* 所需数据集：无
* 所需资源：
* 目标：本节旨在使用智算平台展示如何添加和管理数据集。


## 1、添加和管理数据集

1.1 准备数据集

1.1.1 登录SCOW平台，选取智算平台，进入AI集群
![alt text](assets/image-52.png)

1.1.2 准备数据集
点击文件 -> 选择集群（这里ascend-k8s是AI集群的名称） 
![alt text](assets/image-53.png)

进入AI集群的文件系统
![alt text](assets/image-54.png)

点击 新目录，用来创建数据集及相关文件所在的目录
![alt text](assets/image-55.png)

创建目录时，将目录名定为 data ，点击 确定 按钮
![alt text](assets/image-56.png)

进入新建的 data 目录，看到目录里什么文件都没有

点击 新文件，来创建新文件，文件命名为 identity.json 
![alt text](assets/image-57.png)

点击 确定 按钮，data 目录里面，已经创建了一个名字为 identity.json 的文件
![alt text](assets/image-58.png)

此时文件没有内容，点击文件名 identity.json，打开文件，文件为空白，点击右下角的 编辑 按钮，对文件进行编辑
![alt text](assets/image-59.png)

把[数据集内容](dataset/identity-pku-assistant.json)进行 复制，然后粘贴到 文件 中，点击 保存 按钮，identity.json 就有了内容，在后续的步骤中将作为数据集使用
![alt text](assets/image-60.png)

1.1.3 创建数据集相关的文件

点击 新文件，来创建新文件，文件名命名为 dataset_info.json
![alt text](assets/image-61.png)

将下面 代码 复制后，粘贴到文件

```json
{
  "identity": {
    "file_name": "identity.json"
  }
}  
```

点击 保存 按钮，可以看到 dataset_info.json 文件创建成功，data目录下面已创建两个文件：indentiy.json作为数据集，dataset_info.json作为数据相关信息
![alt text](assets/image-62.png)

1.1.4 为数据集设置版本，方便管理

点击 数据集 > 我的数据集
![alt text](assets/image-63.png)

进入 我的数据集 页面进行管理，点击 添加 按钮
![alt text](assets/image-64.png)

将数据集名称命名为 identity，数据类型中选择 文本，点击 确定 按钮
![alt text](assets/image-65.png)

点击刚添加的数据集 identity 后的 + 加号，为它创建新版本
![alt text](assets/image-66.png)

版本名称可以用日期，例如 2025922，也可以使用自己好理解的名称，点击 选择数据集 最右边图标
![alt text](assets/image-67.png)

选择刚创建的目录 data, 点击 确认 按钮
![alt text](assets/image-68.png)

点击数据集名称前的 + 加号，+ 加号变成 - 减号后，展开查看数据集的版本已经添加成功：
![alt text](assets/image-69.png)
