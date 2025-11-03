# Tutorial2: CLIP图像文本分类

* 集群类型：SCOW超算平台
* 所需镜像：无
* 所需模型：教程内提供
* 所需数据集：教程内提供
* 所需资源：建议使用1张910B NPU运行本教程。
* 目标：本节旨在展示使用‌CLIP模型进行图像文本分类的简单案例，使用OPENAI提供的CLIP库以及给出的示例图片。

‌CLIP模型（Contrastive Language-Image Pre-training）是一种由OpenAI在2021年发布的多模态预训练模型，旨在通过大量文本-图像对进行训练，以理解和匹配图像内容与相应的自然语言描述‌。‌

此教程运行在SCOW超算平台中，请确保运行过tutorial0中安装conda的步骤，再来尝试运行本教程

## 1、环境准备
切换到超算平台中
![alt text](assets/image.png)

点击交互式应用->创建应用进入创建界面，选择vscode应用
![alt text](assets/image-1.png)
![alt text](assets/image-2.png)

节点数填写1，单节点加速卡卡数填写1，最长运行时间适当填写，最后点击提交
![alt text](assets/image-4.png)

在跳转到的页面中点击进入
![alt text](assets/image-5.png)

进到vscode应用中打开terminal
![alt text](assets/image-14.png)

运行下面的命令创建文件夹、配置环境
```shell
mkdir tutorial2
cd tutorial2
conda create -n tutorial2 python==3.10
conda activate tutorial2
pip install ftfy==6.3.1 regex==2024.11.6 tqdm==4.67.1 pyyaml==6.0.2 traitlets==5.14.3 decorator==5.2.1 attrs==25.4.0 psutil==7.1.2
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install .
pip install torch==2.3.1 torch-npu==2.3.1 torchvision==0.18.1 numpy==1.26.4 pandas==2.2.2 scipy==1.13.1
cd ..
```

## 2、数据准备
供模型调用的图像存储在教程同路径下的[CLIP.png](https://github.com/PKUHPC/scow-tutorial-huawei/blob/main/Tutorial2_classification/CLIP.png)文件中，图像如下：
![alt text](assets/image-8.png)

能够看出是CLIP模型的预训练和预测流程图，后面将调用CLIP对本图像进行分类

点击[图像链接](https://github.com/PKUHPC/scow-tutorial-huawei/blob/main/Tutorial2_classification/CLIP.png)进入，点击下载
![alt text](assets/image-9.png)

记住图片下载的路径，通过拖动的方式将图片传到tutorial2文件夹下
![alt text](assets/image-10.png)

最后得到的文件夹结构如下
![alt text](assets/image-11.png)

## 3、模型推理
在tutorial2下创建Python脚本
```shell
echo "" > tutorial2.py
```
在tutorial2.py中放入下面的代码
```python
import torch
import torch_npu
import clip
from PIL import Image

# 设置设备为 NPU
device = "npu" if torch.npu.is_available() else "cpu"

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图像和文本
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 推理
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```

运行下面的命令开始推理
```shell
python tutorial2.py
```

## 4、推理结果
推理结果如下：
![alt text](assets/image-12.png)

可以看到a diagram对应的百分比最高，可知该图像最符合diagram的描述，与事实相符。
![alt text](assets/image-13.png)