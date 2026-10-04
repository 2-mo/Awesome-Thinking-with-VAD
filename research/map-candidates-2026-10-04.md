# 线路图候选与入图记录 · 2026-10-04

六篇候选经用户确认，现已全部加入论文目录与线路图，并补齐原图和引用。以下连接建议是编辑阅读顺序，不代表引用、继承或正式扩展关系。优先补异常理解的具体能力，避免把工业检测论文全面铺开。

| 候选 | 正式发表 | 与当前主线的关系 | 建议 |
| --- | --- | --- | --- |
| Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events | CVPR 2025 | 从部分观察推断意外事件，并在新证据出现后修正解释；适合和 CUVA、Phys-AD 对读 | 优先，异常理解评测方向 |
| AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models | AAAI 2024 | 工业异常定位、语言描述与多轮交互；可作为 MMAD、JUDO 之前的阅读入口 | 优先，早期工业异常理解锚点 |
| ADSeeker: A Knowledge-Grounded Reasoning Framework for Industry Anomaly Detection and Reasoning | CVPR 2026 | 图像查询检索领域知识，再用于异常检测和推理；知识证据来源比单纯输出解释更明确 | 优先，知识检索与异常推理 |
| Where and What: Contextual Dynamics-Aware Anomaly Detection in Surveillance Videos | TIP 2025 | 将场景位置和原子动作结合为上下文异常判断；主要输出仍是检测结果 | 可选，情境判据与表征方向 |
| IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection | AAAI 2026 | PA-SFT 与 SC-GRPO 连接感知、推理及答案的一致性 | 可选，与视频 R1 类方法对读，避免同类堆叠 |
| Towards Training-free Anomaly Detection with Vision and Language Foundation Models（LogSAD） | CVPR 2025 | 使用视觉语言模型提出匹配方案与组合规则，处理逻辑和结构异常 | 可选，若需要显式逻辑异常判据；不是专门的异常问答基准 |

## 入图位置

- AnomalyGPT → 早期表征对齐线；Black Swan → MMAD 与 Phys-AD 之间的评测段。
- Where and What → 时序建模；IAD-R1 → 2026 年一月的结构化推理段。
- ADSeeker → 主动观察与工具／知识获取；LogSAD → 异常判据。
- 工业图像用齿轮，道路用道路图标，检测为主的视频工作用视频图标；视频异常理解不标。根据主要场景和输出手工归类，每项证据保存在 `mapIcon.evidence`，不由题名中的 VAD/VAU 自动推断。
- 新增 BlackSwanSuite、MulA、SEEK-M&V、Expert-AD 四项数据／知识资源。Where and What 正式来源只有发表年，未推断具体出版月份。

## 一手来源

- **Black Swan**：[CVF 正式论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Chinchure_Black_Swan_Abductive_and_Defeasible_Video_Reasoning_in_Unpredictable_Events_CVPR_2025_paper.html)，pp. 24201–24210。这里指 CVPR 2025 主会论文。
- **AnomalyGPT**：[AAAI 正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/27963)，38(3):1932–1940，DOI 10.1609/aaai.v38i3.27963。正式发表为 2024，不能沿用预印本首发年 2023 作为会议年份。
- **ADSeeker**：[CVF 正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ADSeeker_A_Knowledge-Grounded_Reasoning_Framework_for_Industry_Anomaly_Detection_and_CVPR_2026_paper.html)，pp. 21379–21388；[作者稿](https://arxiv.org/abs/2508.03088)。使用正式题名 Knowledge-Grounded，早期预印本题名为 Knowledge-Infused。
- **Where and What**：[作者机构论文记录及摘要](https://scholar.korea.ac.kr/item/26547082-0865-4289-9181-69f54af7cece)，TIP 34:6993–7007，[DOI](https://doi.org/10.1109/TIP.2025.3623392)。
- **IAD-R1**：[AAAI 正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/37588)，40(8):6583–6591，DOI 10.1609/aaai.v40i8.37588。
- **LogSAD**：[CVF 正式论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.html)，pp. 15204–15213。

## 道路方向待进一步核验的线索

**Structured Graph Reasoning for Traffic Anomaly Detection**：作者的[西安交通大学主页](https://faculty.xjtu.edu.cn/spzhou/zh_CN/zhym/1014186/list/index.htm)列为 IEEE TMM 2026，DOI 10.1109/TMM.2026.3703345。目前已确认题名与发表记录，但本轮未取得可直接核查的方法正文，不据二手摘要确定线路位置。
