# Google Map Review Data Analysis and Machine Learning: From Data Wrangling to Multimodal AI （谷歌地图评论数据分析与机器学习：从数据规整到多模态 AI）

[English](./README.md) | **中文**

> **项目状态:** 🟢 进行中 (App 1, App 5 已完成, App 2-4 计划中)

## 🏆 荣誉与认可 (Achievement)

本项目的前身（Task 2 EDA 探索性数据分析报告）获得了 **Monash University FIT5196 (Data Wrangling)** 教学团队的高度评价，并被请求作为 **“优秀范例 (Reference Template)”** 和学习资源分享给未来的学生。

> *"We were particularly impressed by the quality of your Task 3 EDA report (Note: labeled as Task 2 in this repo)... It demonstrated clear insight, strong structure, and thoughtful presentation—an excellent example of applied data analysis."*
>
> — **Kiara, FIT5196 Teaching Team, Monash University**

📄 **[点击此处查看完整邮件凭证 (PDF)](./Monash%20University%20Mail%20-%20Willingness%20to%20Share%20Your%20Excellent%20EDA%20Report%20as%20a%20Learning%20Resource.pdf)**

---

## 1. 数据集背景与项目演变 (Dataset & Project Evolution)

本项目经历了一个从“小规模课程实验”到“大规模数据工程与机器学习应用”的完整演变过程。

### 1.1 数据集概览 (Dataset Overview)
本项目的底层数据源自 **Google Maps**，重点聚焦于**美国加利福尼亚州 (California, USA)** 范围内的商铺及其相关的用户交互记录。
* **商铺元数据 (Metadata):** 包含近 51 万家加州商铺的地理位置、名称、所属类别（如餐饮、零售、美容服务等）、平均评分及官方网站等结构化信息。
* **用户交互数据 (Review Data):** 包含真实用户的非结构化文本评价、1-5 星打分、时间戳以及用户上传的实景图片。这些高度真实的业务数据为后续的自然语言处理 (NLP)、计算机视觉 (CV) 及检索增强生成 (RAG) 提供了极具挑战性与价值的多模态语料库。

### 1.2 原始版本 (Original Version)
* **定位:** Monash FIT5196 课程作业 (`02_original_version`)。
* **数据范围:** 针对单一学生组分配的数据 (**Group 020**)，数据量相对较小。
* **核心任务:** 完成了对 Group 020 脏数据的清洗，文本预处理与分析，并产出了 EDA 报告。

### 1.3 拓展版本 (Extended Version)
* **定位:** 课程结束后的自主探索与工程化升级 (`03_extended_version`)。
* **数据范围:** 通过编写自动化脚本，将原本分散的所有学生组数据进行了合并与去重，构建了 **GroupALL** 数据集。数据规模从几万条扩展至 **570万+ 行** 评论数据及对应的 **50万+ 张** 图片，店家数量从不到 200 家扩展至近 30000 家。这使得项目从简单的分析转变为对“大数据”处理能力的挑战。
* **核心任务 (Tasks):**
    * **[Task 1: Data Wrangling (Big Data Pipeline)](./03_extended_version/Task1_Data_Wrangling.ipynb)**：针对 500万+ 行级别的脏数据重构了清洗管道，处理了更复杂的边缘情况与数据一致性问题。
    * **[Task 2: Original EDA Report (Award-Winning)](./03_extended_version/Task2_EDA_Report_Original_VR.ipynb)**：这是基于 **Group 020 (小数据集)** 的原始分析报告。正是这份报告的分析逻辑与呈现方式获得了教学团队的表彰。
    * **[Task 3: Validation EDA (GroupALL)](./03_extended_version/Task3_EDA_Report_Extended_VR.ipynb)**：基于 **GroupALL (全量大数据集)** 的验证性分析。目的在于验证在 Task 2 小数据集中发现的规律与模式（如评分分布、关键词趋势），在数据量扩大 200 倍后是否依然成立，从而检验分析结论的鲁棒性。

---

## 2. 机器学习应用组合 (Machine Learning Portfolio)

基于清洗后的 GroupALL 全量多模态数据，目前规划了 5 个核心应用模块 (`04_extended_app`)：

| 模块 | 应用名称 | 核心技术栈 | 状态 | 技术文档 |
| :--- | :--- | :--- | :--- | :--- |
| **App 1** | **多模态情感分类器** | **RoBERTa + Swin Transformer + LoRA** | ✅ 已完成 | **[点击查看详情](./05_ml_workflow_description/App1_Multimodal_Text_Classification.md)** |
| **App 2** | 个性化推荐系统 | DeepFM / NeuralCF | 📅 计划中 | - |
| **App 3** | 商家图网络分析 | Graph Neural Networks (GNN) | 📅 计划中 | - |
| **App 4** | 跨语言评论翻译 | Seq2Seq / Transformer | 📅 计划中 | - |
| **App 5** | **RAG 商业问答系统** | **Llama-3 + ChromaDB + QLoRA** | ✅ 已完成 | **[点击查看详情](./05_ml_workflow_description/App5_Fine_Tuning_LLM.md)** |

### 🌟 App 1: 多模态评分预测 (Multimodal Sentiment Classifier)
App 1 是一个基于 **双塔架构 (Two-Tower Architecture)** 的模型，它能够同时处理评论文本和用户上传的图片，预测用户的 1-5 星评分。
* **挑战:** 真实数据极其稀疏（仅 8% 含图）且存在严重的长尾分布。
* **解决方案:** 在 A100 环境下设计了“条件计算 (Conditional Computation)”与“全内存预加载 (In-Memory Caching)”策略，使用 LoRA 技术对 RoBERTa 与 Swin Transformer 进行参数高效微调。
* **性能:** 在独立测试集上 F1-Macro 达到 **0.5859**，且成功识别出文本与图片情感不一致的复杂案例（如反讽）。

### 🌟 App 5: 双轨制 RAG 商业分析系统 (Dual-Track RAG Business Analysis System)
App 5 是一个结合了**指令微调 (Instruction Fine-Tuning)** 与**检索增强生成 (RAG)** 的端到端商业分析问答系统，旨在通过自然语言交互，为用户提供准确、客观的本地商铺洞察。
* **挑战:** 基础大模型缺乏海量本地商铺的长尾与动态信息，且容易在垂直领域的商业分析中捏造事实（幻觉）。
* **解决方案:** 
  1. **模型微调:** 基于 Llama-3-8B-Instruct 模型进行 4-bit QLoRA 微调。在 SFT 数据集构建阶段，刻意引入了 20% 的“拒绝回答”抗幻觉负样本，强制模型对齐严谨的商业分析客观性与格式规范。
  2. **双轨检索架构:** Track 1 基于 Parquet 查找表提供 O(1) 复杂度的客观静态属性兜底；Track 2 基于 ChromaDB 和 1024 维 BGE Embedding 执行非结构化评论文本的 Top-K 语义召回，彻底剥离模型对内部静态记忆的依赖。
* **性能:** 系统在百万级数据规模下实现了跨词汇的隐式语义匹配。在端到端系统评估中，展现了出色的指令遵从度、极高的引用溯源精准度，并在面对跨领域提问及非法查询等边界测试中表现出高度的工程鲁棒性。

---

## 3. 项目文件结构 (Project Structure)

> **注:** 由于数据规模庞大（包含数百万行文本及数十万张图片），部分大文件（如 `.ftr` 数据集、`.pth` 模型权重、原始 `.txt` 数据）在 Git 仓库中通过 `.gitignore` 排除，但它们是项目运行的核心组成部分，以下展示了主要的本地工作区结构。

```text
2024-07-Google-Map-Review-Data-Analysis-and-Machine-Learning
├─ 01_data                                # 数据中心
│  ├─ 01_data_raw                         # 原始脏数据 (Excel/TXT)
│  │  └─ Student Data
│  │     ├─ student_group001              # 示例：第1组数据
│  │     │  ├─ group001.xlsx
│  │     │  ├─ group001_0.txt
│  │     │  └─ ... (更多分片文件)
│  │     └─ ... (student_group002 ~ 017)
│  ├─ 02_data_wrangled                    # 清洗后的中间数据 (JSON/CSV)
│  │  ├─ Task1_for_Group020.csv           # 原始版本使用的单组数据
│  │  └─ Task1_for_GroupALL.csv           # 拓展版本使用的全量数据 (5.7M+ rows)
│  ├─ 03_images_raw                       # 全局图像资产库 (从 Google CDN 下载)
│  │  ├─ 0x14e1..._..._0.jpg
│  │  └─ ... (~50万张图片)
│  ├─ 04_data_for_app1                    # App 1 专用数据集 (Feather格式, 高效读取)
│  │  ├─ App1_Data_Train.ftr              # 训练集 (80%, ~1.8M rows)
│  │  ├─ App1_Data_Val.ftr                # 验证集 (10%)
│  │  └─ App1_Data_Test.ftr               # 测试集 (10%)
│  │
│  ├─ 08_data_for_app5                    # App 5 专用数据集 (JSONL格式, 用于SFT)
│  │  ├─ App5_Data_Train.jsonl            # SFT 训练集 (1000条)
│  │  ├─ App5_Data_Val.jsonl              # SFT 验证集 (200条)
│  │  ├─ App5_Data_Test.jsonl             # SFT 测试集 (10条)
│  │  └─ App5_Data_Test_Mini.jsonl        # 早期 API 测试验证集 (5条)
│  └─ ... (App 2-4 数据预留目录)
│
├─ 02_original_version                    # 原始课程作业代码 (基于 Group 020 数据)
│  ├─ task1_020.ipynb                     # 数据清洗脚本
│  ├─ task2_020.ipynb                     # 文本预处理与分析
│  ├─ task3_020.ipynb                     # 获奖的 EDA 报告 (Legacy)
│  └─ ... (PDFs, Scripts)
│
├─ 03_extended_version                    # 拓展版核心分析 (基于 GroupALL 全量数据)
│  ├─ Task1_Data_Wrangling.ipynb          # [Task 1] 大数据清洗管道
│  ├─ Task2_EDA_Report_Original_VR.ipynb  # [Task 2] 基于小数据的原始 EDA (获奖版本)
│  └─ Task3_EDA_Report_Extended_VR.ipynb  # [Task 3] 基于大数据的验证性 EDA
│
├─ 04_extended_app                        # 深度学习应用模块
│  ├─ App1_Multimodal_Text_Classification
│  │  ├─ 01_Data_Preprocessing.ipynb      # App 1 ETL 管道 (构建 .ftr)
│  │  ├─ 02_Multimodal_Text_Classification.ipynb # 双塔模型训练 (A100)
│  │  ├─ 03_Model_Query.ipynb             # 推理演示与案例测试 (Demo)
│  │  └─ test_images                      # 手动收集的测试用例图片 (用于 Demo)
│  │
│  ├─ App5_Fine_Tuning_LLM                # App 5: 双轨制 RAG 与 LLM 微调
│  │  ├─ 01_Data_Preprocessing.ipynb      # SFT 数据合成与清洗管道
│  │  ├─ 02_Fine_Tuning_LLM.ipynb         # Llama-3 QLoRA 微调训练
│  │  ├─ 03_LLM_Validation.ipynb          # 模型验证与最优 Checkpoint 选择
│  │  ├─ 04_Vector_Database_Build.ipynb   # ChromaDB 向量库与静态表构建
│  │  ├─ 05_Model_Query.ipynb             # 端到端 RAG 推理与系统评估
│  │  ├─ ChromaDB                         # 本地向量数据库持久化实例
│  │  └─ Llama3_8B_App5_LoRA              # 保存的最佳 LoRA 权重目录
│  └─ App2...App4                         # 后续 App 目录结构规划
│
├─ 05_ml_workflow_description             # 技术文档与资源
│  ├─ App1_Multimodal_Text_Classification.md # App 1 详细技术架构文档
│  ├─ App5_Fine_Tuning_LLM.md             # App 5 详细技术架构文档
│  └─ images                              # 文档插图
│
├─ Monash University Mail...pdf           # 教学团队表彰邮件
├─ README.md                              # 英文说明
└─ README.zh.md                           # 中文说明
```

---

## 4. 关于作者 (About)

* **作者:** Zihan Yin
* **教育背景:**
    * **Master of Data Science**, Monash University
    * **Bachelor of Science**, University of Melbourne (Major in Data Science, Minor in Statistics & Stochastic Processes)
* **联系方式:** [2002317yzh@gmail.com](mailto:2002317yzh@gmail.com)
* **Github:** [https://github.com/ZihanYin12138](https://github.com/ZihanYin12138)

如果你对这个项目感兴趣，欢迎 Star ⭐️ 或 Fork！

---
