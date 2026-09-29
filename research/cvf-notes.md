# CVF / ECCV 补充核验（2026-09-29）

本轮新增 12 篇正式主会论文（CVPR 8、ICCV 1、WACV 3），所有论文都已读取 CVF 论文集页面的正式标题、作者、年份和方法摘要。与原有 17 条无重复。年份取正式发表年：AnyAnomaly、ASK-HINT 是 WACV 2026，不能沿用 2025 预印本年份；MissionGNN 是 WACV 2025，不能沿用 2024 预印本年份。

- 直接理解/解释：EVAL、UCA、VA-GPT、AnyAnomaly、ASK-HINT。EVAL 是属性解释，不是 LLM；UCA 是 UCF-Crime 的句子与时间标注扩展，不等同于 2026 ACL 的 VALU。
- 支撑语义理解的检测路线：OVVAD、TPWNG、Anomize、Alert-CLIP、LAVIDA、LAS-VAD、MissionGNN。保留其准确任务，不把所有语义检测都标为自由文本异常解释。
- 代码保守标记：AnyAnomaly 与 MissionGNN 通过作者论文指向代码仓库并有实现；LAVIDA 虽然论文写了 code available，但仓库的 train/inference 仍是 Coming Soon，故只填 project。其余没有确认官方实现则不补猜测的 GitHub 链接。
- datasetIds 只关联原 catalog 已有且明确验证的数据。UCA 新数据集尚不在原 catalog，因此目前关联视觉来源 ucf-crime；主代理合并新数据集后可以补 introduces uca 的关系。追加阅读 CVF 正文实验设置后确认 OVVAD、LAVIDA、VA-GPT 均使用 UCF-Crime 与 XD-Violence，已补齐关联；未依靠引用列表推断。

筛掉或暂未加入：

- Phys-AD、AdaCLIP、工业/医学异常、OOD road objects、Beyond Walking 的图文行人搜索不属本轮视频异常理解核心范围。
- PrismVAU 是 WACV 2026 Workshops，SmartHome-Bench 是 CVPR 2025 Workshops；本轮已有足够主会补充，未混入主会。
- PE-MIL（CVPR 2024）有可靠官方出处，但与 TPWNG 同属早期提示弱监督支线，本轮控制前史比例未再加入。官方页：https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html
- FedVAD（ECCV 2024）是可进一步扩充的语言知识蒸馏支线，此轮只发现官方论文，没有纳入最终 12 条：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06981.pdf
- Streaming Video Crime Anticipation with Spatio-Temporal Causal Reasoning（CVPR 2026）是犯罪预判邻接任务，未为数量扩张纳入。官方页：https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Streaming_Video_Crime_Anticipation_with_Spatio-Temporal_Causal_Reasoning_CVPR_2026_paper.html
- 已有 VADER 可补正式来源：https://openaccess.thecvf.com/content/WACV2026/html/Cheng_VADER_Towards_Causal_Video_Anomaly_Understanding_with_Relation-Aware_Large_Language_WACV_2026_paper.html

注意旧 llm4vad.md 有 “ICCV 2024” 标题，这不是有效 ICCV 主会年份；未沿用其会议归属。
