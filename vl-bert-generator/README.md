# VL-BERT Generator

本目录下的内容在 [VL-BERT](https://arxiv.org/abs/1908.08530) 项目的基础上，针对课题的数据集进行了调整，环境要求等请参考 [VL-BERT 官方 GitHub 仓库](https://github.com/jackroos/VL-BERT) 中的说明。本课题主要参考的是其中 `refcoco` 目录下的内容

> 原仓库的 README 中的 Prepare 一节描述了环境的准备工作，请参考

基于原始代码，主要修改了 dataset 与 dataloader，以适配本课题的数据；将生成部分替换为了基于 GPT-1 的设计，参考的是 [LayoutTransformer](https://arxiv.org/abs/2006.14615) 中的设计，在 [原代码仓库](https://github.com/kampta/DeepLayout) 基础上修改

> LayoutTransformer 提供的代码难以复现得到论文中描述的效果

代码中添加了一些必要的注释，运行方式参考 `.sh` 文件，配置文件在 `./cfgs/refcoco/base_detected_regions_4x16G.yaml`，同样在原文件上进行了一些修改，主要集中在 DATASET 部分