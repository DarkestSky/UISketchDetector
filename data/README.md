# 相关数据集

> 本目录保存使用到的数据集，以及生成的中间结果

数据集包含三部分内容

- 输入数据
    - 草图数据
    - 文本描述数据
- 输出数据
    - 页面元素，作为 ground-truth

## 草图数据

草图数据使用的是 [SynZ](https://dl.acm.org/doi/10.1145/3397482.3450725) 提供的方案，参照 [Rico 数据集](https://interactionmining.org/rico) 中的页面标注信息，使用 [UISketch 数据集]() 提供的页面元素草图进行拼接得到整个页面的草图图片。

参考内容如下：

| Content  | Links          |
|----------|----------------|
| SynZ     | [Paper](https://dl.acm.org/doi/10.1145/3397482.3450725), [Dataset](https://www.kaggle.com/datasets/vinothpandian/synz-dataset) |
| Rico     | [Paper](https://dl.acm.org/doi/10.1145/3126594.3126651#:~:text=The%20Rico%20dataset%20contains%20design,than%2072k%20unique%20UI%20screens.), [Dataset](https://interactionmining.org/rico) |
| UISketch | [Paper](https://dl.acm.org/doi/fullHtml/10.1145/3411764.3445784), [Dataset](https://www.kaggle.com/datasets/vinothpandian/uisketch) |

其中，SynZ 提供了生成草图数据集的代码，也可以直接下载生成结果

## 文本描述数据

文本描述数据使用的是 [Screen2Words](https://arxiv.org/abs/2108.03353) 提供的页面描述文本，并对其中部分明显的错误数据进行了修正

数据保存在 `./description` 文件夹下

| Content      | Links          |
|--------------|----------------|
| Screen2Words | [Paper](https://arxiv.org/abs/2108.03353), [Dataset](https://github.com/google-research-datasets/screen2words) |

## 页面元素

使用的是 Rico 数据集，上述两部分全部是基于 Rico 数据集进行扩充，参考链接见上

## 草图识别结果

根据模型设计，首先使用 RetinaNet 对草图进行识别，结果作为后续网络的输入，并且两部分独立执行

草图的识别结果以 `.pth` 文件的形式保存在 `./synz_prediction` 目录下

