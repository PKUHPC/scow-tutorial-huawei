# scow-tutorial-huawei
本教程介绍如何在基于 [SCOW](https://www.pkuscow.com/) 的华为鲲鹏昇腾集群上申请计算资源并运行各类计算任务。本教程已纳入 [华为官方在线课程](https://www.hiascend.com/developer/courses/detail/1909399063897702401)。

[教程入口](https://github.com/PKUHPC/scow-tutorial-huawei/blob/main/tutorial_scow_huawei.md)

## 贡献指南
欢迎贡献新教程或改进现有教程。请参考现有教程的格式，编写教程后提交pr，仓库管理员会不定期进行合并。

## issue
如果遇到问题，欢迎提交issue。

## 生成web页面

```bash
pip install notebook beautifulsoup4
conda install pandoc
conda install -c conda-forge parallel 
sudo apt-get update
sudo apt-get install texlive-xetex google-chrome-stable pdftk

bash release.sh
```
