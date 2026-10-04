# 视觉异常理解：数据维护

`catalog.json` 是网页、[创新目录](../literature/catalog.md)、[论文年表](../llm4vad.md)、[发表索引](../literature/venues.md)、[方法比较](../literature/comparison.md)、[数据集与评测](../literature/benchmarks.md)、[阅读路线](../literature/reading-guide.md)和[引用导出](../literature/citations.md)的共同数据源，类型定义见 `src/types.ts`。当前数量与更新时间由生成的年表展示。

## 收录范围

项目名称中的 VAD 指 Visual Anomaly Detection。保持异常理解主线，当前内容以视频相关工作为主，围绕异常描述、问答、解释、推理、证据定位及评测整理文献。

选择性收录与主线密切相关的工业图像异常理解、道路异常理解／预测工作，按其具体问题与方法归入现有文献体系，不为场景覆盖而扩充论文或另建分类。检测、表征与多模态融合论文应说明对异常理解的具体支持作用。检测方向新增入图优先选择 2024–2026 年顶会工作，并明确其与异常理解的联系，例如语义提示、语言知识蒸馏、多模态语义诱导、事件完整性和视觉语言复核。纯预测、重构、概率建模等通用检测基础可保留为背景文献，不因此扩展线路；不新增 2022 年工作。所有纳入结构化目录的论文沿用 `scope: "core"`。

WACV、Findings 与 workshop 论文在 README 的“其他论文”分类展示；WACV 与 workshop 的补充笔记见[其他论文](../literature/other-papers.md)，已有结构化记录的 Findings 条目继续由目录生成。`research/` 中的历史排除说明只记录当时批次的范围，后续选篇以本节为准。

本轮 TPAMI 的 VAD／VAU 与直接基础文献系统补录，正式卷期优先于 DOI 年份；2021 年会议论文不补入，期刊论文不受此限制。TPAMI 已核验条目见[期刊索引](../archive/journals/tpami.md)。

### 线路图独立规划

文献收录与线路图选篇分别规划。文献可以先入库，线路图按异常理解中的研究问题和方法机制精选代表性工作，保留现有主线；入图采用 `isMapPaper` 规则，单篇编辑取舍通过 `mapExclusion: { note }` 记录，文献索引和引用继续保留。直接继承或扩展关系仍需一手来源。

检测线同时保留不同方法范式的代表性工作：PI-VAD 代表多模态信息融合，DSRL 代表欧氏／双曲空间表征，FedVAD 代表联邦学习与语言语义蒸馏。是否直接生成语言解释不是唯一入图条件；精简优先处理同方向的重复站点。PE-MIL、PEL 作为提示学习补充阅读留在目录与引用中，移出线路图；其他已有站点保持当前选篇。

### 来源与记录

- 题名和会议以正式论文集、作者论文或官方会议页面核实；未确认录用时保留 `arXiv`。
- `year` 采用正式发表年；预印本使用首次提交年。NeurIPS Datasets and Benchmarks 在数据中保留准确分轨，展示时合并为 NeurIPS。
- 同一工作的预印本、会议版和扩展版先核对版本关系，避免重复统计。新增扩展贡献不能倒写进原会议论文的摘要。
- `sources` 记录一手 URL 与其支持的具体事实；`verifiedAt` 记录本次核验日期。代码仓库仍仅有 README 或写明即将发布时用 `links.project`，不标为已发布代码。
- `mechanism` 用不超过 24 字概括创新；`summary` 概括贡献，`takeaway` 与 `limitation` 是编辑阅读提示，不冒充复现实验结论。
- `datasetIds` 仅列来源明确支持、且已存在于目录的数据集。空数组表示本轮未建立关联。

## 任务与比较维度

`tasks` 只使用：异常检测、异常定位、空间定位、异常解释、异常推理、异常预判、异常检索、视频问答、基准评测。方法、训练与场景标签放入可选 `tags`，参与搜索但不混入任务筛选。异常检索用于文本／音频查询与视频匹配，按检索候选库和召回指标评测。

道路事故预测／提前预警按论文实际任务使用 `异常预判`，并核对预测时点、可见帧范围与预测目标。工业图像问答按实际贡献使用异常解释、异常推理等任务及具体场景标签，不标为 `视频问答`；扩展受控任务词表时同步修改类型、校验器和生成器。

可选 `comparison` 分为 `outputs`（输出／评测对象）、`training`（训练与适配）、`inference`（运行设置）、`futureFrames`（未来帧访问）、`evaluation`（验证方式）。每项格式为 `{ values: ["内容"], evidence: { url, note } }`，须有针对该维度的一手依据。未核验就省略该项，生成表显示“待核验”。基准论文可记录其评测对象，不假装它是统一输出模型；冻结主模型不等于全流程无需训练，流式标签不自动证明严格无未来帧访问。

`limitation` 暂保留兼容字段名，统一展示为“阅读关注”，不冒充实验确认的缺陷。

## 方法主线与支线

`clusters[].name` 用于中文文献索引；`nameEn` 用于网页分类、线路图和 SVG。SVG 的标题、图例与换乘说明统一使用英文。发表信息统一使用无边框文字，会议稍粗，期刊使用常规字重并配按出版商区分的浅底色（IEEE 期刊为相近冷色，IJCV 为暖色），arXiv 使用灰色斜体；图例只解释三种贡献类型，不解释发表样式或换乘；这些样式不改变节点形状、研究方向颜色或时间位置。

可选 `layoutNear` 指定希望相邻排布的分类，不改变论文归属，也不生成换乘、分叉或继承关系。当前异常判据与结构化推理通过 AnomalyRuler、LRPO 的双重方法归属自然相邻，不再设置额外的 `layoutNear`。

评测支线使用 `evaluation.layoutNear: "reasoning"`，优先靠近自己的父线：浅绿支线排在深绿推理主线上方，异常判据红线在下方。稀疏支线从分叉论文先作短斜向分离，接近下一站时再完成主要高度变化。

| ID | 方法方向 | 主要关注 |
| --- | --- | --- |
| `detection` | 视频异常检测 | 语义提示、语言蒸馏、多模态语义、完整事件与语义复核 |
| `alignment` | 多模态理解与对齐 | 语义检索、异常描述、问答与解释 |
| `explanation` | 异常判据与提示优化 | 可读属性、正常规则、语言反馈与异常判断标准 |
| `understanding` | 时序建模与记忆 | 姿态与轨迹序列、事件边界和在线记忆 |
| `evidence` | 主动观察与工具决策 | 疑点搜索、补充采样、检索和工具使用 |
| `reasoning` | 结构化推理与验证 | 因果与关系推理、反思纠错 |
| `synthesis` | 异常构造与监督 | 表征对齐的支线：伪标签、伪异常和可控视频生成 |
| `evaluation` | 评测与任务拓展 | 推理的支线：异常问答、关键事实与专题评测 |

`clusters[].branchOf` 指向父线路，父线路可以是主线，也可以是已有支线；同一线路可以在不同论文处多次分支。必须同时提供 `branchAt: { paperId, evidence: { url, note } }`，指定有一手内容依据的论文作为分叉节点。该论文通过主归属与 `secondaryMethods` 同时连接两条线路，并位于所接支线论文之前。更早的独立阅读段须在 `routes` 中从另一个父／子方向共享论文起始，整段早于此分叉；索引中的补充论文不参与地图分叉的时间约束。校验器拒绝缺少节点、依据或形成循环的分支。地图中的分叉站只有一个站点、一个标签，轨道从同一中心分开。

当前类别分叉为 Vad-R1（感知认知推理、自验证强化学习与 Vad-Reasoning 数据）；OVVAD 与 TPWNG 以普通衔接站组成蓝色主干内的监督构造色段。TPWNG 依据可学习文本提示、正常视觉提示与 CLIP 适配添加表征对齐次级归属，不单独悬出一站。浅绿评测支线从 Vad-R1 接向 Cue-R1，便于对照 R1 式推理训练与任务评测；CUVA 保留在推理主线，并以因果理解基准与 MMEval 的贡献兼属评测方向，接出 CUVA → MMAD → Phys-AD 的早期评测阅读段。这是基于论文贡献的阅读结构，不把后续同方向论文一律认定为直接继承。名称、颜色、节点与依据统一维护在数据中。

WACV、Findings、workshop 及编辑移出主图的论文保留在文献索引；网页的站点、筛选、详情入口、计数与 SVG 导出统一依据 `isMapPaper` 选取。可选 `mapExclusion: { note }` 记录主图编辑取舍，不改变发表信息或方法分类，也不作为论文质量的事实断言。STEP、TrajVAD、PE-MIL、PEL 等论文使用此标记；EWAD、SEEK-VAU、CA-Judge、ROAD 保留入图。结构化目录 127 篇，地图 103 篇、103 个独立站点，共 8 条线路。Vad-R1 与 Vad-R1-Plus 各自保留节点和日期，扩展关系由带来源的 `extends` 单独记录。

每篇论文选一个主要方法方向，允许存在其他贡献。线路是编辑分组，不是引用或继承关系；没有论文节点的交叉不代表方法融合。


`mapIcon: { kind: "industry" | "road" | "video", evidence }` 为可选的地图小图标提示。工业图像使用齿轮，道路／交通使用道路，主要输出为视频异常判断、评分、定位或预判的检测工作使用视频图标；视频异常理解不标。场景优先，每篇最多一个图标；是否使用推理组件不能单独决定任务归类。图标放在发表信息旁，悬停显示含义，不增加图例、文字标签或新的线路分类。

地图横轴是时间，纵向由方法拓扑决定，会议随论文名称显示。`cluster` 是主要方法归属；可选 `secondaryMethods: [{ cluster, evidence: { url, note } }]` 记录有一手来源依据的兼属方法；多路交汇时组成换乘站，一进一出的跨色衔接与支线回接使用单个普通站；不得重复主方法或引用未知分类。兼属是编辑归类，不是引用、继承关系，不能仅凭使用某个通用组件就添加。当前换乘包含 Ex-VAD、PANDA、MemoVAD、A2Seek-R1、Anom-π 和 AnomalyRuler；具体形态由各站实际连接决定。LAVIDA 保留两个贡献方向的来源记录，图面以单个普通站衔接棕色检测线与粉色构造线。FineVAU 作为评测支线普通站；Vad-R1-Plus 以推理模型及分阶段问答基准的共同贡献连接推理与评测，主归属仍为推理，图面以单个普通节点承接支线回流。分叉起点同样使用普通节点，不额外显示 Fork 字样或 Y 形符号。CG-CoE 以异常事件抽取与类别约束匹配组成的评价链，连接结构化验证和任务评测，主归属仍为评测；图面使用单个普通分叉节点，两种方法归属与依据保留。后两篇以规则归纳与演绎、推理轨迹与语言经验优化连接异常判据和结构化推理。

可选 `timeline: { month, basis, source }` 在同一年内提供隐藏排序：`conference` 表示主会月份，`journal` 表示有来源的期刊上线或正式发表月份（来源说明采用哪种日期），`preprint` 表示 arXiv 首次提交月份，也可用于期刊条目可核验的首次公开时间（如 CRCL）；正式发表身份与引文独立保留。没有可靠来源就不猜月份，未知项置于该年已知月份之后。2025 年起显示 Q1–Q4，季度边界按已核验月份确定；空季度保留窄空档，未知月份另列待定区。间距用于排版，不表示精确时间间隔；线路不得向左折返。`clusters[].position` 仅是保留的旧布局元数据，当前拓扑算法不使用它。

## 数据集与关系

论文可用 `contribution: { kind, evidence: { url, note } }` 标注贡献类型：`resource` 表示以数据集、benchmark 或评测协议为主，`hybrid` 表示同时提出方法与数据／基准；省略时按方法显示。此标注依据论文实际贡献，与研究线路、发表分轨分别维护，不能由 `datasetIds` 非空、使用了某个数据集或题名含 Bench 自动推断。基准中用于比较的常规基线不单独算作方法贡献。线路图以圆形、空心方形、方框内圆点分别表示三类，详情保留判定来源。

数据集的性质与开放情况可分别记录，旧条目允许暂时省略；未确认开放时，卡片只提供“项目入口”：

- `composition: { kind, note, evidence, baseDatasetIds? }`：`kind` 为 `original`（原始数据）、`annotation`（扩展标注）、`resplit`（重新划分）或 `mixed`（混合数据）。`note` 说明组成与新增内容，`evidence: { url, note }` 提供依据；可选 `baseDatasetIds` 只引用目录内其他数据集，不得引用自身。
- `availability: { status, note, evidence, verifiedAt }`：`status` 为 `available`（已开放）、`partial`（部分开放）、`pending`（待发布）或 `unverified`（待核验）。说明具体可用资源与限制，附作者来源及有效 `YYYY-MM-DD` 核验日期；论文发表或项目页面存在不自动等于数据已开放。

生成的数据集索引沿用论文卡片的标题、徽章、摘要和原图结构，按异常检测、理解与推理、异常检索分组。正文只展示简介、精简标注、可用原图与必要的 `usageNote`；该可选短句用于说明实际影响获取或复现的限制。协议、基础数据、相关论文与来源入口收在折叠区，完整核验记录仍维护在 JSON 中。期刊徽章只显示刊名，年份另列。

数据集可先登记文字记录，再补图片。可选 `thumbnail` 一旦提供，就需要本地 `/datasets/` 或已有 `assets/papers/` 路径、替代文本、原图来源和作者署名；使用作者的数据样例、标注示意或基准概览，避免纯方法框架图。卡片使用 `../public/datasets/` 或 `../assets/papers/` 相对路径，兼容 GitHub Markdown；可选 `caption` 简短说明图意。新增图像遵循 [图片来源记录](../public/datasets/README.md)。

关系 `source → target` 只使用 `uses`、`introduces`、`extends`，每条均需明确来源。`extends` 只用于有证据的方法继承，不能以主题相似替代。关系保留在详情中，不决定主地图走线。

## 更新流程

1. 按收录范围核对主来源、发表状态和已有版本，补齐论文记录与来源说明；独立决定是否入图，暂不入图的条目填写 `mapExclusion` 说明。
2. 运行 `npm run build`，同步生成阅读索引、配图来源清单、`literature/references.bib` 与 `docs/` 静态网页。
3. 运行 `npm run check`，检查数据引用、生成文件一致性、地图几何与 TypeScript。
4. 在开发分支提交，检查通过后合并 main。`research/` 保存检索记录，不作为另一份运行时数据。

结构校验不能代替学术事实核验，也不保证外部链接永久有效。

## 论文卡片与配图

`llm4vad.md` 复用旧版“标题 → 会议／代码徽章 → 摘要 → 论文配图”的逐篇卡片结构，并保留年份、会议和单篇锚点。新增或修改内容应维护 `catalog.json`，不要只改生成后的 Markdown。

可选 `figure` 记录 `src`、`alt`、`caption`、`sourceUrl`、`sourcePageUrl`、`credit`、`verifiedAt`；图片统一存放在根目录 `assets/papers/`，使用作者原图或原论文 PDF 图区渲染，配图版本与正式发表身份分别记录。点击图可查看本地大图；图下只显示简短图注、署名与一个“来源”入口，链接至 `assets/papers/README.md#figure-<paper-id>` 的对应行。来源清单由同一记录生成，保留完整原图／PDF、作者页面、图注及核验日期。图片不纳入仓库文字和代码许可，版权归原作者／出版方。

没有取得可靠原图时保留论文，用 `figurePending: { note, sources, verifiedAt }` 记录具体障碍与已查来源；不能同时填写 `figure`。不以生成图或其他论文插图填空。`npm run validate` 检查图片路径、来源字段、本地文件与实际文件格式；更新图片时还须人工核对论文、图号和裁切完整性。

## 引用元数据

已确认录用但作者尚未核齐时，可以先入图与索引。此时 `citation.version: "pending"` 只记录 `title`、`year`、`url`、`note`、`sources`、`verifiedAt`，引用页显示“书目待补”，不导出 BibTeX。方法可依据题名先作编辑判断，并设置 `classification: { basis: "title", evidence: { url, note } }`；详情与生成目录明确标为“按题名暂定归类”。作者、结果、训练与实验设置仍需逐项核验。

完整引用的 `citation` 必须包含 `key`、`type`、`version`、`title`、完整有序 `authors`、引用 `year`、`url`、`sources` 与 `verifiedAt`。正式版本（`published`）提供 `publication`；已录用但尚无正式出版书目的记录（`accepted`）使用 `misc` 与 `publication`，BibTeX 导出为 `note`；arXiv 预印本（`preprint`）使用 `misc` 与 `arxivId`。`doi` 仅填写已核验标识符，不填写 URL；没有证据就省略，不从题名推测 DOI。卷、期、页码分别为可选 `volume`、`number`、`pages`。

官方 BibTeX 优先决定引用年份；网页上线日期不一定等于会议年份。不要混合不同版本的作者、年份、DOI 和论文集。`literature/citations.md` 与 `literature/references.bib` 由同一记录生成，保留来源；生成器保护缩写大小写，并转义 BibTeX 特殊字符与常见姓名重音。

### 局部线路与节点分叉

可选 `clusters[].routes: [{ paperIds, evidence }]` 显式列出同色主线、局部支线和终点的论文顺序，每段需提供编辑依据。列表覆盖该分类全部入图论文，只引用真实同方向论文并遵守跨月时间顺序；共享起点仍只画一个论文站。它不创建新分类或直接继承声明。当前早期监督段从 OVVAD 至 TPWNG，两端在普通节点与表征主干衔接；LAVIDA 从 Alert-CLIP 节点的短支线接入网络，再接后期生成段；COPRA 从 UPR-VAD 节点分出；早期评测段由 CUVA 经 MMAD、Black Swan 至 Phys-AD；后续评测段由 Vad-R1 经 Cue-R1、FineVAU，在 Vad-R1-Plus 回接推理与验证主线；同一区域的深绿主线经 URF-ZS-HVAA、TargetVAU，下方同色支线经 VAD-DPO，也在 Plus 汇合，后期评测段从 CG-CoE 的评测／验证共享节点接出，避免横穿 Anom-π 两侧。所有分叉都从论文站台出发，过线用底色间隙区分。`layoutSide: "above" | "below"` 与 `layoutNear` 配合约束上下排布。 同色密集区设置回接阅读段：TTHF 经 PEL、DSRL、VarCMP、Anomize 回到 Ex-VAD；Holmes-VAU 经 EventVAD、VADTree 回到 PANDA；TD-VAD 经 EWAD 至 DEAL、MemoVAD 至 S2MGraph-VAD 均独立收尾，不强制回接主线。工业推理则由 Vad-R1-Plus 经 IAD-R1、JUDO 至 O-VAD 独立收尾，不再回接 LAS-VAD。分支只调整阅读拓扑，不新增方法归属或继承关系。

`timeline.year` 可选，用于线路图采用在线发表年份、而 `paper.year` 与 `citation.year` 保留正式卷期年份的情况；须提供日期来源，且不能晚于正式发表年份。PromptVAD 线路图采用用户提供的 2025-10-30，正式 TNNLS 37(3) 引用仍为 2026。完整日保存在 `timeline.source.note`，图上按月定位。Pistachio 与 ReactVAU 分别按用户提供的 2026-07-07、2026-09-09 版本日期定位，不将其误称为会议月份。

本轮检测线补充 PE-MIL（CVPR 2024）、FedVAD（ECCV 2024）、PI-VAD（CVPR 2025）、LEC-VAD（ICML 2025）与 D²MIL（CVPR 2026），保持八条方法线路。各站说明语义知识如何改善异常判断或完整事件定位，不把使用语言模型等同于输出解释。选篇与线路位置见[本轮记录](../research/semantic-detection-2026-10-04.md)。

粉色后期合成段位于检测主干上方 432 单位；EWAD／DEAL 位于中间的上侧分支。COPRA 与 TD-VAD 仅归属检测，不再连接蓝线或紫线；紫线由 PANDA 经 VALU 继续到 MemoVAD，保留平直主干。蓝线按 VA-GPT → HiProbe-VAD → HeadHunt-VAD → SteerVAD 组织有效词元、隐藏状态、注意力头与表示校正的阅读段。MLLM-EVAD 按编辑要求移出目录及地图。当前排版见更新记录；前一版布局见[检测区扩展记录](../research/detection-space-2026-10-04.md)。
