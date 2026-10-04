# 异常理解 · 阅读路线

[论文年表](../llm4vad.md) · [方法比较](comparison.md) · [数据集与评测](benchmarks.md) · [背景综述](../research/README.md)

[引用导出](citations.md) · [更新记录](../CHANGELOG.md)

按研究问题选择路线。以下顺序是编辑阅读建议，不表示论文之间存在引用或继承关系。

- [从语义表征到异常判据](#guide-from-detection)
- [从时间组织到主动观察](#guide-long-active)
- [从结构化解释到可检验结论](#guide-understand-anomaly)
- [如何评估异常理解](#guide-evaluate-understanding)

<a id="guide-from-detection"></a>

## 从语义表征到异常判据

问题：语言如何参与异常判断？依次比较表征对齐、运动监督、语言中介、规则归纳与提示优化。

1. [VadCLIP](<catalog.md#paper-vadclip>)：创新：视觉语言双分支对齐。先问异常类别语义如何改变检测表征。
2. [HAWK](<catalog.md#paper-hawk>)：创新：运动信息与语言监督。再问仅有静态外观会遗漏什么。
3. [LAVAD](<catalog.md#paper-lavad>)：创新：字幕、时序聚合与语言评分。观察语义如何转化为异常判断。
4. [AnomalyRuler](<catalog.md#paper-anomalyruler>)：创新：从正常参考归纳规则。把通用判断改成有场景依据的判据。
5. [VERA](<catalog.md#paper-vera>)：创新：用弱标签优化引导问题。检查显式判据是否可以被数据改进。

<a id="guide-long-active"></a>

## 从时间组织到主动观察

问题：有限上下文里怎样保留并补充有效证据？比较时间分层、事件划分、在线记忆与观察策略。

1. [Holmes-VAU](<catalog.md#paper-holmes-vau>)：创新：多粒度指令与异常聚焦采样。先确定片段、事件和视频如何组织。
2. [EventVAD](<catalog.md#paper-eventvad>)：创新：动态时空图与事件边界。研究依据事件调整时间单元的方式。
3. [MoniTor](<catalog.md#paper-monitor>)：创新：历史预测与分数队列。加入在线约束，检查哪些信息可被保留。
4. [PANDA](<catalog.md#paper-panda>)：创新：场景规划、工具反思与经验记忆。研究如何根据疑点改变处理流程。
5. [Anom-π](<catalog.md#paper-anom-pi>)：创新：交替推理和时间操作的策略学习。把下一次观察纳入优化目标。

<a id="guide-understand-anomaly"></a>

## 从结构化解释到可检验结论

问题：怎样区分合理叙述与有依据的理解？连接问题分解、反思修正和事实／边界评估。

1. [CUVA](<catalog.md#paper-cuva>)：创新：事件、原因与后果的任务分解及评估。先明确解释到底要回答什么。
2. [SRVAU-R1](<catalog.md#paper-srvau-r1>)：创新：初始推理、反思与修正的训练序列。检验纠错是否改变最终判断。
3. [FineVAU](<catalog.md#paper-finevau>)：创新：关键视觉事实分解与 FVScore。判断回答是否抓住事件、实体和位置。
4. [VALU](<catalog.md#paper-valu>)：创新：语义层级下的异常边界及评估协议。检验定位是否覆盖完整事件语义。

<a id="guide-evaluate-understanding"></a>

## 如何评估异常理解

问题：怎样区分回答流畅、视觉事实正确和推理可靠？依次比较任务定义、事实评估、反例诊断与指标元评测。

1. [CUVA](<catalog.md#paper-cuva>)：任务定义：事件、原因与后果是否分别评估？定位正确能否代表解释正确？
2. [ECVA / AnomShield](<catalog.md#paper-ecva-anomshield>)：因果理解扩展：对照 CUVA 与 ECVA 的事件、原因、后果和关键证据标注，区分 MMEval 与 AnomEval。
3. [FineVAU](<catalog.md#paper-finevau>)：事实评估：回答是否覆盖关键事件、参与者与位置？指标与人工判断是否一致？
4. [VAD-DPO](<catalog.md#paper-vad-dpo>)：反例诊断：视觉相似而异常语义相反时，模型能否摆脱物体与异常词语的共现捷径？
5. [CG-CoE](<catalog.md#paper-cg-coe>)：指标元评测：保持事件语义但改变措辞，评分是否稳定？类别引导是否引入额外偏差？
