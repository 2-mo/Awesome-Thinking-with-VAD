# 视频异常理解 · 数据集与评测

[论文年表](../llm4vad.md) · [方法比较](comparison.md) · [阅读路线](reading-guide.md)

[引用导出](citations.md) · [更新记录](../CHANGELOG.md)

16 个数据资源记录。登记依据论文与作者来源，不以缩略图为前提，也不等同于已发布可下载数据。关联论文只包含已核验关系；没有关联不表示没有使用。

| 数据集 | 年份／发表 | 任务 | 标注 |
| --- | --- | --- | --- |
| [UCF-Crime](#dataset-ucf-crime) | 2018 · CVPR | 异常检测、异常定位 | 视频级异常标签、测试时序标注 |
| [XD-Violence](#dataset-xd-violence) | 2020 · ECCV | 异常检测、异常定位 | 视频级多标签、测试时序标注 |
| [CUVA](#dataset-cuva-dataset) | 2024 · CVPR | 异常解释、异常推理、视频问答 | 异常类别与时间边界、事件描述、原因解释、后果描述 |
| [HIVAU-70k](#dataset-hivau-70k) | 2025 · CVPR | 异常检测、异常解释、视频问答 | 片段级字幕、事件级判断、描述与分析、视频级总结 |
| [HAWK](#dataset-hawk-dataset) | 2024 · NeurIPS | 异常解释、视频问答 | 异常视频语言描述、异常相关问答 |
| [FineW3](#dataset-finew3) | 2026 · AAAI | 异常解释 | 异常事件、参与实体、位置 |
| [UCA](#dataset-uca) | 2024 · CVPR | 异常检测、异常定位、异常解释 | 事件级自然语言描述、句子起止时间 |
| [ECVA](#dataset-ecva) | 2024 · arXiv | 异常解释、异常推理、视频问答 | 异常类别与时间边界、事件描述、原因解释、后果描述 |
| [Vad-Reasoning](#dataset-vad-reasoning) | 2025 · NeurIPS | 异常检测、异常推理、异常解释 | 结构化推理过程、最终异常判断、异常类型与时间边界、RL 子集视频级弱标签 |
| [CueBench](#dataset-cuebench-data) | 2026 · AAAI | 异常检测、异常定位、异常预判、基准评测 | 条件性／绝对异常类别、场景与属性层级、任务相关标注 |
| [VAGU](#dataset-vagu-data) | 2026 · AAAI | 异常定位、异常解释、视频问答 | 异常类别、语义解释、时间边界、视频问答／选择问答 |
| [VALU](#dataset-valu-data) | 2026 · ACL | 异常定位、异常解释、基准评测 | 五级异常语义、各语义级时间边界、细粒度文本描述 |
| [A2Seek](#dataset-a2seek-data) | 2025 · NeurIPS Datasets and Benchmarks | 异常检测、异常定位、空间定位、异常解释 | 异常类别、帧级时间戳、区域边界框、自然语言解释 |
| [TAU-Bench](#dataset-tau-bench-data) | 2026 · arXiv | 空间定位、异常解释、异常推理、基准评测 | 异常实例轨迹、像素级掩码、实例／事件／场景描述 |
| [Pistachio](#dataset-pistachio-data) | 2026 · ECCV | 异常检测、异常解释、异常推理、基准评测 | 帧级标注、事件级描述、视频级描述 |
| [VANE-Bench](#dataset-vane-bench-data) | 2025 · NAACL Findings | 异常检测、异常定位、视频问答、基准评测 | 异常类别、异常问答 |

<a id="dataset-ucf-crime"></a>

## UCF-Crime

真实监控长视频异常检测基准，也是多项语言异常理解工作的视觉来源。

- 模态：视频
- 评测协议／阅读关注：采用官方训练／测试划分；弱监督训练与帧级定位评估须区分。
- 来源入口：[作者／论文](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>) · [论文](<https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html>)
- 已关联论文：[VadCLIP](<catalog.md#paper-vadclip>) · [LAVAD](<catalog.md#paper-lavad>) · [Ex-VAD](<catalog.md#paper-ex-vad>) · [EventVAD](<catalog.md#paper-eventvad>) · [MoniTor](<catalog.md#paper-monitor>) · [OVVAD](<catalog.md#paper-ovvad>) · [TPWNG](<catalog.md#paper-tpwng>) · [UCA](<catalog.md#paper-uca-paper>) · [Anomize](<catalog.md#paper-anomize>) · [VA-GPT](<catalog.md#paper-va-gpt>) · [LAVIDA](<catalog.md#paper-lavida>) · [LAS-VAD](<catalog.md#paper-las-vad>) · [VADTree](<catalog.md#paper-vadtree>) · [MemoVAD](<catalog.md#paper-memovad>) · [Flashback](<catalog.md#paper-flashback>) · [LRPO](<catalog.md#paper-lrpo>) · [TD-VAD](<catalog.md#paper-td-vad>) · [URF-ZS-HVAA](<catalog.md#paper-urf-zs-hvaa>) · [CLUE-VAD](<catalog.md#paper-clue-vad>) · [HeadHunt-VAD](<catalog.md#paper-headhunt-vad>) · [Probe-VAD](<catalog.md#paper-probe-vad>)
- 核验依据：[来源](<https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html>) — 资源首发论文或作者发布页：核实任务、标注与出版信息。；[来源](<https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/>) — 作者数据发布页与使用说明。

<a id="dataset-xd-violence"></a>

## XD-Violence

包含音视频与多场景暴力事件的弱监督检测基准。

- 模态：视频、音频
- 评测协议／阅读关注：报告所用模态与官方划分；不可将音视频方法和纯视觉方法混为同一设置。
- 来源入口：[作者／论文](<https://roc-ng.github.io/XD-Violence/>)
- 已关联论文：[VadCLIP](<catalog.md#paper-vadclip>) · [LAVAD](<catalog.md#paper-lavad>) · [Ex-VAD](<catalog.md#paper-ex-vad>) · [EventVAD](<catalog.md#paper-eventvad>) · [MoniTor](<catalog.md#paper-monitor>) · [OVVAD](<catalog.md#paper-ovvad>) · [TPWNG](<catalog.md#paper-tpwng>) · [Anomize](<catalog.md#paper-anomize>) · [VA-GPT](<catalog.md#paper-va-gpt>) · [LAVIDA](<catalog.md#paper-lavida>) · [LAS-VAD](<catalog.md#paper-las-vad>) · [VADTree](<catalog.md#paper-vadtree>) · [MemoVAD](<catalog.md#paper-memovad>) · [Flashback](<catalog.md#paper-flashback>) · [LRPO](<catalog.md#paper-lrpo>) · [TD-VAD](<catalog.md#paper-td-vad>) · [URF-ZS-HVAA](<catalog.md#paper-urf-zs-hvaa>) · [CLUE-VAD](<catalog.md#paper-clue-vad>) · [HeadHunt-VAD](<catalog.md#paper-headhunt-vad>) · [Probe-VAD](<catalog.md#paper-probe-vad>)
- 核验依据：[来源](<https://roc-ng.github.io/XD-Violence/>) — 资源首发论文或作者发布页：核实任务、标注与出版信息。；[来源](<https://roc-ng.github.io/XD-Violence/>) — 作者数据发布页与使用说明。

<a id="dataset-cuva-dataset"></a>

## CUVA

以异常事件的内容、原因与后果为核心的因果理解基准。

- 模态：视频、文本
- 评测协议／阅读关注：按原论文任务与 MMEval 协议评估；定位、描述和因果解释分别报告。
- 来源入口：[作者／论文](<https://github.com/fesvhtr/CUVA>) · [论文](<https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html>)
- 已关联论文：[CUVA](<catalog.md#paper-cuva>)
- 核验依据：[来源](<https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html>) — 资源首发论文或作者发布页：核实任务、标注与出版信息。；[来源](<https://github.com/fesvhtr/CUVA>) — 作者数据发布页与使用说明。

<a id="dataset-hivau-70k"></a>

## HIVAU-70k

Holmes-VAU 提出的片段、事件和视频三级异常指令数据。

- 模态：视频、文本
- 评测协议／阅读关注：区分时间粒度与任务类型；数据构建包含模型生成与人工复核。
- 来源入口：[作者／论文](<https://github.com/pipixin321/HolmesVAU>) · [论文](<https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html>)
- 已关联论文：[Holmes-VAU](<catalog.md#paper-holmes-vau>) · [TargetVAU](<catalog.md#paper-targetvau>)
- 核验依据：[来源](<https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html>) — 资源首发论文或作者发布页：核实任务、标注与出版信息。；[来源](<https://github.com/pipixin321/HolmesVAU>) — 作者数据发布页与使用说明。

<a id="dataset-hawk-dataset"></a>

## HAWK

面向开放场景异常视频描述与交互问答的数据资源。

- 模态：视频、文本
- 评测协议／阅读关注：遵循作者数据划分，分别检查描述生成与问答表现。
- 来源入口：[作者／论文](<https://github.com/jqtangust/hawk>) · [论文](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html>)
- 已关联论文：[HAWK](<catalog.md#paper-hawk>)
- 核验依据：[来源](<https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html>) — 资源首发论文或作者发布页：核实任务、标注与出版信息。；[来源](<https://github.com/jqtangust/hawk>) — 作者数据发布页与使用说明。

<a id="dataset-finew3"></a>

## FineW3

围绕 What、Who、Where 补充细粒度视觉事实的异常理解数据与评估。

- 模态：视频、文本
- 评测协议／阅读关注：使用 FVScore 检查关键视觉元素，结合人类一致性分析；不能只比较语言流畅度。
- 来源入口：[作者／论文](<https://finevau.github.io/>) · [论文](<https://ojs.aaai.org/index.php/AAAI/article/download/37790/41752>)
- 已关联论文：[FineVAU](<catalog.md#paper-finevau>)
- 核验依据：[来源](<https://ojs.aaai.org/index.php/AAAI/article/download/37790/41752>) — 资源首发论文或作者发布页：核实任务、标注与出版信息。；[来源](<https://finevau.github.io/>) — 作者数据发布页与使用说明。

<a id="dataset-uca"></a>

## UCA

在 UCF-Crime 监控视频上增加细粒度事件语句和时间边界，连接异常检测、语言定位与密集描述。

- 模态：视频、文本
- 评测协议／阅读关注：使用作者训练、验证、测试划分；分别报告语言时序定位、视频描述、密集描述与多模态异常检测。
- 来源入口：[作者／论文](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>) · [论文](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>)
- 已关联论文：[UCA](<catalog.md#paper-uca-paper>)
- 核验依据：[来源](<https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html>) — 一手论文：定义数据任务与标注。；[来源](<https://github.com/jia-wang-11/Surveillance-Video-Understanding-dataSet>) — 作者仓库：数据已发布，提供数据获取与评估说明。

<a id="dataset-ecva"></a>

## ECVA

CUVA 的扩展因果理解基准，围绕异常经过、发生原因与事件后果提供人工语言标注。

- 模态：视频、文本
- 评测协议／阅读关注：按作者发布版本分别评估描述、原因和后果；AnomEval 检查推理、回答一致性与幻觉，避免与 CUVA 的 MMEval 混用。
- 来源入口：[作者／论文](<https://github.com/Dulpy/ECVA>) · [论文](<https://arxiv.org/abs/2412.07183>)
- 已关联论文：待核验
- 核验依据：[来源](<https://arxiv.org/abs/2412.07183>) — 一手论文：定义数据任务与标注。；[来源](<https://github.com/Dulpy/ECVA>) — 作者仓库：数据已发布，提供数据获取与评估说明。；[来源](<https://www.modelscope.cn/datasets/gouchenyi/ECVA/files>) — 作者仓库链接的数据与标注发布位置。

<a id="dataset-vad-reasoning"></a>

## Vad-Reasoning

Vad-R1 提出的异常推理数据，使用感知到认知的结构化推理标注，并区分监督微调和强化学习子集。

- 模态：视频、文本
- 评测协议／阅读关注：分别使用 Vad-Reasoning-SFT 的训练／测试划分与 Vad-Reasoning-RL；SFT 含推理文本，RL 仅有视频级弱标签。
- 来源入口：[作者／论文](<https://github.com/wbfwonderful/Vad-R1>) · [论文](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>)
- 已关联论文：[Vad-R1](<catalog.md#paper-vad-r1>)
- 核验依据：[来源](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/abccc325c84dedf23dbe8de3f686c733-Abstract-Conference.html>) — 一手论文：定义数据任务与标注。；[来源](<https://github.com/wbfwonderful/Vad-R1>) — 作者仓库：数据已发布，提供数据获取与评估说明。；[来源](<https://huggingface.co/datasets/wbfwonderful/Vad-R1>) — 作者发布的 Vad-Reasoning 数据文件。

<a id="dataset-cuebench-data"></a>

## CueBench

以场景和属性组织条件性与绝对异常，检验异常判断的上下文依赖。

- 模态：视频、文本
- 评测协议／阅读关注：分别报告识别、时序定位、检测和预判；按场景／属性分析条件性异常。当前登记依据论文，不据此宣称数据已开放下载。
- 来源入口：[作者／论文](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>)
- 已关联论文：[CueBench / Cue-R1](<catalog.md#paper-cuebench>)
- 核验依据：[来源](<https://ojs.aaai.org/index.php/AAAI/article/view/38209>) — 一手论文确认数据集名称、标注和任务；未据论文存在推定资源已经开放下载。

<a id="dataset-vagu-data"></a>

## VAGU

联合视频异常时间定位与语义理解的基准。

- 模态：视频、文本
- 评测协议／阅读关注：使用原版 VAGU 的问答与 JeAUG 联合评价，联合分数之外保留定位和理解单项结果；不混入扩展稿 VAGU-T。当前未核验数据下载状态。
- 来源入口：[作者／论文](<https://ojs.aaai.org/index.php/AAAI/article/view/42412>)
- 已关联论文：[VAGU & GtS](<catalog.md#paper-vagu-gts>)
- 核验依据：[来源](<https://arxiv.org/abs/2507.21507>) — 原稿摘要明确异常类别、解释、时间定位、视频问答与 JeAUG；正式发表身份见论文记录。

<a id="dataset-valu-data"></a>

## VALU

以五个语义层级组织异常事件边界与文本描述。

- 模态：视频、文本
- 评测协议／阅读关注：按语义层级分别评估 temporal grounding、anomaly localization 和 detail discrimination。论文页写明将公开基准，本次未确认下载状态。
- 来源入口：[作者／论文](<https://aclanthology.org/2026.acl-long.56/>)
- 已关联论文：[VALU](<catalog.md#paper-valu>)
- 核验依据：[来源](<https://aclanthology.org/2026.acl-long.56/>) — 一手论文确认数据集名称、标注和任务；未据论文存在推定资源已经开放下载。

<a id="dataset-a2seek-data"></a>

## A2Seek

面向动态航拍视角，连接异常事件、时空证据与因果解释。

- 模态：视频、文本
- 评测协议／阅读关注：区分异常判断、区域定位和解释；遵循作者场景与分布外设置，不与固定监控结果直接混排。当前未核验下载状态。
- 来源入口：[作者／论文](<https://2-mo.github.io/A2Seek/>) · [论文](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>)
- 已关联论文：[A2Seek / A2Seek-R1](<catalog.md#paper-a2seek>)
- 核验依据：[来源](<https://proceedings.neurips.cc/paper_files/paper/2025/hash/de02de513503962e1d21035ab50ce661-Abstract-Datasets_and_Benchmarks_Track.html>) — 一手论文确认数据集名称、标注和任务；未据论文存在推定资源已经开放下载。

<a id="dataset-tau-bench-data"></a>

## TAU-Bench

以异常实例轨迹绑定分层语义，评估解释是否对应正确对象。

- 模态：视频、文本
- 评测协议／阅读关注：联合检查实例跟踪和细粒度语义，不能以描述流畅度替代轨迹正确性；本次未确认数据／模型有效下载入口。
- 来源入口：[作者／论文](<https://yarkupa.github.io/tau-bench.github.io/>) · [论文](<https://arxiv.org/abs/2608.05699>)
- 已关联论文：[TAU-Bench](<catalog.md#paper-tau-bench>)
- 核验依据：[来源](<https://arxiv.org/abs/2608.05699>) — 一手论文确认数据集名称、标注和任务；未据论文存在推定资源已经开放下载。

<a id="dataset-pistachio-data"></a>

## Pistachio

包含 Pistachio-VAD 和 Pistachio-VAU 两部分的可控合成视频基准，覆盖检测与事件语义理解。

- 模态：视频、文本
- 评测协议／阅读关注：分别报告 VAD 与 VAU 协议；关注合成到真实的域差异及多事件子集。数据下载未在本次逐文件验证。
- 来源入口：[作者／论文](<https://pistachio-video.github.io>) · [论文](<https://arxiv.org/abs/2511.19474>)
- 已关联论文：[Pistachio](<catalog.md#paper-pistachio>)
- 核验依据：[来源](<https://arxiv.org/abs/2511.19474>) — 一手摘要核验题名、作者和研究内容；Comments 注明 Accepted to ECCV 2026；正式书目未核验，引用导出保留 arXiv 版本。；[来源](<https://arxiv.org/html/2511.19474v6>) — 第 3 节区分 Pistachio-VAD 与 Pistachio-VAU；事件级、视频级描述与多异常视频由正文确认。

<a id="dataset-vane-bench-data"></a>

## VANE-Bench

通过问答评测合成视频不一致性和真实视频异常，覆盖检测与定位。

- 模态：视频、文本
- 评测协议／阅读关注：分别检查合成与真实视频子集；问答正确率不直接等价于逐帧检测 AUC。作者提供代码和数据入口，本次未逐文件验证下载。
- 来源入口：[作者／论文](<https://github.com/rohit901/VANE-Bench>) · [论文](<https://aclanthology.org/2025.findings-naacl.171/>)
- 已关联论文：[VANE-Bench](<catalog.md#paper-vane-bench>)
- 核验依据：[来源](<https://aclanthology.org/2025.findings-naacl.171/>) — 一手摘要核验题名、作者和研究内容；ACL Anthology 正式发表页及其 BibTeX 核验作者顺序、年份、页码和 DOI。
