# 与异常理解相关的检测工作：2024–2026

本轮按用户澄清后的范围选篇：像 PI-VAD 一样，通过语义、上下文或事件内容理解帮助异常检测的顶会工作。主要输出可以是检测分数与时间定位，但与异常理解的联系必须来自论文的具体机制。

## 新增五篇

| 工作 | 正式发表 | 与异常理解的联系 | 线路位置 |
| --- | --- | --- | --- |
| [PE-MIL](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html) | CVPR 2024 | 异常类别语义提示与正常上下文提示，区分异常内容和背景并改善事件边界 | VadCLIP → PE-MIL → OVVAD |
| [FedVAD](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6981_ECCV_2024_paper.php) | ECCV 2024 | GPT 生成／校准公共视频描述，多模态教师将语义知识蒸馏进联邦检测模型 | PEL → FedVAD → DSRL |
| [PI-VAD / π-VAD](https://openaccess.thecvf.com/content/CVPR2025/html/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.html) | CVPR 2025 | 姿态、深度、全景分割、光流和视觉语言语义补充 RGB，辨别外观相似但含义不同的事件 | Anomize → PI-VAD → Ex-VAD |
| [LEC-VAD](https://proceedings.mlr.press/v267/wang25l.html) | ICML 2025 | 类别感知／类别无关语义、原型记忆与边界约束，形成更完整的异常事件定位 | Ex-VAD → LEC-VAD → MP-GDFL |
| [D²MIL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_CVPR_2026_paper.html) | CVPR 2026 | 视觉语言模型复核被当作噪声丢弃的片段，找回有价值的困难异常 | Alert-CLIP → D²MIL → TD-VAD |

上述联系与线路安排是依据论文贡献作出的编辑归类，连线表示阅读顺序，不声明方法继承。五篇均接入现有视频异常检测线，保留八条方法线路；不因使用视觉语言模型就增加理解或推理线路的换乘归属。

## 阅读边界与来源

- PI-VAD 的五个额外模态骨干只在训练时使用；推断时由 RGB 及诱导模块完成检测。其语义丰富性支持异常辨别，论文主要任务并非自然语言解释。
- PE-MIL 使用视频级异常类别标注和可学习提示，不能概括成只需要正常／异常二值标签。PE-MIL 与已有 PEL 是不同工作。
- FedVAD 分别评测无监督与弱监督协议；语言模型用于语义知识构建与蒸馏，不代表在线逐帧调用 GPT。作者仓库暂记为项目入口，不据论文的发布承诺推断代码已开放。页码、卷号和 DOI 另据 [Springer 正式章节](https://link.springer.com/chapter/10.1007/978-3-031-73668-1_14)；会议月份据 [ECCV 主会安排](https://eccv.ecva.net/Conferences/2024/Registration)，使用 10 月。
- LEC-VAD 正式发表于 **ICML 2025**，事件完整性指时间定位，不等同于故事生成或因果解释。
- D²MIL 的视觉语言复核参与训练样本筛选，是与异常内容理解相关的检测训练方法，不单独标为异常问答。

题名、作者、正式发表、页码与方法摘要均据上表的一手论文页；训练设置、数据集和原图据其链接的论文正文。五篇均采用作者 Figure 2，逐图页码、PDF 来源与署名保存在 `data/catalog.json`，由生成器同步至图片索引。

## 撤回范围偏离的地图扩展

上一批新增的正常性建模线路已撤下。Future Frame Prediction（2022）、MGFN、UR-DMU、STG-NF、FPDM（2023），以及 SSMCTB、Self-Distilled MAE、BN-WVAD、ADSM 保留为背景文献与引用，通过 `mapExclusion` 移出地图。没有删除此前已有的理解相关早期站点。

本轮更新后目录 126 篇，地图 104 个独立站点、8 条线路，102 篇有作者原图。未将纯重构、预测或密度估计方面的检测改进作为此次新增入图的充分理由。
