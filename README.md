# Google Map Review Data Analysis and Machine Learning: From Data Wrangling to Multimodal AI

**English** | [中文](./README.zh.md)

> **Project Status:** 🟢 In Progress (App 1, App 5 Completed, App 2-4 Planned)

## 🏆 Achievement

The predecessor of this project (Task 2 EDA Exploratory Data Analysis Report) received high praise from the **Monash University FIT5196 (Data Wrangling)** teaching team and was requested to be shared as a **"Reference Template"** and learning resource for future students.

> *"We were particularly impressed by the quality of your Task 3 EDA report (Note: labeled as Task 2 in this repo)... It demonstrated clear insight, strong structure, and thoughtful presentation—an excellent example of applied data analysis."*
>
> — **Kiara, FIT5196 Teaching Team, Monash University**

📄 **[Click here to view the full email credential (PDF)](./Monash%20University%20Mail%20-%20Willingness%20to%20Share%20Your%20Excellent%20EDA%20Report%20as%20a%20Learning%20Resource.pdf)**

---

## 1. Dataset Background & Project Evolution

This project has evolved from a "small-scale coursework experiment" into a "large-scale data engineering and machine learning application" portfolio.

### 1.1 Dataset Overview
The underlying data for this project originates from **Google Maps**, focusing specifically on local merchants and their related user interaction records within **California, USA**.
* **Merchant Metadata:** Contains structural information of nearly 510,000 Californian merchants, including geographic locations, names, categories (e.g., dining, retail, beauty services), average ratings, and official websites.
* **User Interaction Data (Review Data):** Contains unstructured text reviews from real users, 1-5 star ratings, timestamps, and user-uploaded real-world images. These highly authentic business data provide a challenging and valuable multimodal corpus for subsequent Natural Language Processing (NLP), Computer Vision (CV), and Retrieval-Augmented Generation (RAG) tasks.

### 1.2 Original Version
* **Scope:** Monash FIT5196 Coursework (`02_original_version`).
* **Data Scale:** Specific to data assigned to a single student group (**Group 020**), relatively small scale.
* **Core Tasks:** Completed data cleaning for Group 020's dirty data, text preprocessing & analysis, and produced an EDA report.

### 1.3 Extended Version
* **Scope:** Independent exploration and engineering upgrade after the course (`03_extended_version`).
* **Data Scale:** By writing automated scripts, the data from all dispersed student groups was merged and deduplicated to build the **GroupALL** dataset. The data scale expanded from tens of thousands to **5.7 million+ rows** of review data and corresponding **500,000+ images**, with the number of merchants expanding from less than 200 to nearly 30,000. This transformed the project from simple analysis into a "Big Data" processing challenge.
* **Core Tasks:**
    * **[Task 1: Data Wrangling (Big Data Pipeline)](./03_extended_version/Task1_Data_Wrangling.ipynb)**: Reconstructed the cleaning pipeline for 5-million+ row level dirty data, handling more complex edge cases and data consistency issues.
    * **[Task 2: Original EDA Report (Award-Winning)](./03_extended_version/Task2_EDA_Report_Original_VR.ipynb)**: The original analysis report based on **Group 020 (Small Dataset)**. It was the analytical logic and presentation of this report that won the commendation from the teaching team.
    * **[Task 3: Validation EDA (GroupALL)](./03_extended_version/Task3_EDA_Report_Extended_VR.ipynb)**: Validation analysis based on **GroupALL (Full Big Dataset)**. The purpose is to verify whether the laws and patterns found in the Task 2 small dataset (e.g., rating distribution, keyword trends) still hold true after the data volume is expanded by 200 times, thereby testing the robustness of the analytical conclusions.

---

## 2. Machine Learning Portfolio

Based on the cleaned GroupALL full multimodal data, 5 core application modules are currently planned (`04_extended_app`):

| Module | Application Name | Core Stack | Status | Tech Docs |
| :--- | :--- | :--- | :--- | :--- |
| **App 1** | **Multimodal Sentiment Classifier** | **RoBERTa + Swin Transformer + LoRA** | ✅ Completed | **[View Details](./05_ml_workflow_description/App1_Multimodal_Text_Classification.md)** |
| **App 2** | Personalized Recommendation | DeepFM / NeuralCF | 📅 Planned | - |
| **App 3** | Merchant Graph Analysis | Graph Neural Networks (GNN) | 📅 Planned | - |
| **App 4** | Cross-Lingual Translation | Seq2Seq / Transformer | 📅 Planned | - |
| **App 5** | **RAG Business Q&A System** | **Llama-3 + ChromaDB + QLoRA** | ✅ Completed | **[View Details](./05_ml_workflow_description/App5_Fine_Tuning_LLM.md)** |

### 🌟 App 1: Multimodal Rating Prediction (Multimodal Sentiment Classifier)
App 1 is a model based on **Two-Tower Architecture**, capable of processing review text and user-uploaded images simultaneously to predict user ratings (1-5 stars).
* **Challenge:** Real-world data is extremely sparse (only 8% contains images) and has a severe long-tail distribution.
* **Solution:** Designed "Conditional Computation" and "In-Memory Caching" strategies in an A100 environment, and used LoRA technology for parameter-efficient fine-tuning of RoBERTa and Swin Transformer.
* **Performance:** Achieved an F1-Macro of **0.5859** on an independent test set, successfully identifying complex cases where text and image sentiments are inconsistent (e.g., sarcasm).

### 🌟 App 5: Dual-Track RAG Business Analysis System
App 5 is an end-to-end business analysis Q&A system combining **Instruction Fine-Tuning** and **Retrieval-Augmented Generation (RAG)**, aimed at providing users with accurate and objective local merchant insights through natural language interaction.
* **Challenge:** Base large models lack long-tail and dynamic information about massive local merchants, and are highly prone to fabricating facts (hallucination) in vertical domain business analysis.
* **Solution:** 1. **Model Fine-Tuning:** Conducted 4-bit QLoRA fine-tuning based on the Llama-3-8B-Instruct model. During the SFT dataset construction phase, 20% "refusal to answer" anti-hallucination negative samples were deliberately introduced to force the model to align with rigorous business analysis objectivity and formatting standards.
  2. **Dual-Track Retrieval Architecture:** Track 1 provides O(1) complexity objective static attribute fallback based on a Parquet lookup table; Track 2 performs Top-K semantic recall of unstructured review text based on ChromaDB and 1024-dimensional BGE Embedding, completely stripping the model's reliance on internal static memory.
* **Performance:** The system achieved cross-vocabulary implicit semantic matching at a million-level data scale. In the end-to-end system evaluation, it demonstrated excellent instruction adherence, extremely high citation traceability accuracy, and showed high engineering robustness when facing boundary tests such as cross-domain questions and invalid queries.

---

## 3. Project Structure

> **Note:** Due to the massive scale of data (containing millions of text rows and hundreds of thousands of images), large files (such as `.ftr` datasets, `.pth` model weights, original `.txt` data) are excluded from the Git repository via `.gitignore`, but they are integral parts of the project runtime. The following displays the complete local workspace structure.

```text
2024-07-Google-Map-Review-Analysis-and-Machine-Learning
├─ 01_data                                # Data Center
│  ├─ 01_data_raw                         # Raw dirty data (Excel/TXT)
│  │  └─ Student Data
│  │     ├─ student_group001              # Example: Group 1 Data
│  │     │  ├─ group001.xlsx
│  │     │  ├─ group001_0.txt
│  │     │  └─ ... (More split files)
│  │     └─ ... (student_group002 ~ 017)
│  ├─ 02_data_wrangled                    # Cleaned intermediate data (JSON/CSV)
│  │  ├─ Task1_for_Group020.csv           # Used in Original Version
│  │  └─ Task1_for_GroupALL.csv           # Used in Extended Version (5.7M+ rows)
│  ├─ 03_images_raw                       # Global Image Asset Library (From Google CDN)
│  │  ├─ 0x14e1..._..._0.jpg
│  │  └─ ... (~500k images)
│  ├─ 04_data_for_app1                    # App 1 Specific Dataset (Feather format)
│  │  ├─ App1_Data_Train.ftr              # Train Set (80%, ~1.8M rows)
│  │  ├─ App1_Data_Val.ftr                # Val Set (10%)
│  │  └─ App1_Data_Test.ftr               # Test Set (10%)
│  │
│  ├─ 08_data_for_app5                    # App 5 Specific Dataset (JSONL format, for SFT)
│  │  ├─ App5_Data_Train.jsonl            # SFT Train Set (1000 samples)
│  │  ├─ App5_Data_Val.jsonl              # SFT Validation Set (200 samples)
│  │  ├─ App5_Data_Test.jsonl             # SFT Test Set (10 samples)
│  │  └─ App5_Data_Test_Mini.jsonl        # Early API testing validation set (5 samples)
│  └─ ... (Reserved for App 2-4 Data)
│
├─ 02_original_version                    # Original Coursework Code (Based on Group 020)
│  ├─ task1_020.ipynb                     # Data Cleaning Script
│  ├─ task2_020.ipynb                     # Text Preprocessing & Analysis
│  ├─ task3_020.ipynb                     # Award-Winning EDA Report (Legacy)
│  └─ ... (PDFs, Scripts)
│
├─ 03_extended_version                    # Extended Core Analysis (Based on GroupALL)
│  ├─ Task1_Data_Wrangling.ipynb          # [Task 1] Big Data Cleaning Pipeline
│  ├─ Task2_EDA_Report_Original_VR.ipynb  # [Task 2] Original EDA (Awarded Version)
│  └─ Task3_EDA_Report_Extended_VR.ipynb  # [Task 3] Validation EDA on Big Data
│
├─ 04_extended_app                        # Deep Learning Applications
│  ├─ App1_Multimodal_Text_Classification
│  │  ├─ 01_Data_Preprocessing.ipynb      # App 1 ETL Pipeline (Build .ftr)
│  │  ├─ 02_Multimodal_Text_Classification.ipynb # Two-Tower Model Training (A100)
│  │  ├─ 03_Model_Query.ipynb             # Inference Demo & Case Testing
│  │  └─ test_images                      # Manually collected test images (For Demo)
│  │
│  ├─ App5_Fine_Tuning_LLM                # App 5: Dual-Track RAG & LLM Fine-Tuning
│  │  ├─ 01_Data_Preprocessing.ipynb      # SFT Data Synthesis & Cleaning Pipeline
│  │  ├─ 02_Fine_Tuning_LLM.ipynb         # Llama-3 QLoRA Fine-Tuning Training
│  │  ├─ 03_LLM_Validation.ipynb          # Model Validation & Checkpoint Selection
│  │  ├─ 04_Vector_Database_Build.ipynb   # ChromaDB Vector DB & Static Table Build
│  │  ├─ 05_Model_Query.ipynb             # End-to-End RAG Inference & System Evaluation
│  │  ├─ ChromaDB                         # Local Vector DB Persistence Instance
│  │  └─ Llama3_8B_App5_LoRA              # Saved Optimal LoRA Weights Directory
│  └─ App2...App4                         # Future Apps Structure
│
├─ 05_ml_workflow_description             # Technical Documentation & Resources
│  ├─ App1_Multimodal_Text_Classification.md # App 1 Detailed Tech Architecture
│  ├─ App5_Fine_Tuning_LLM.md             # App 5 Detailed Tech Architecture
│  └─ images                              # Documentation Illustrations
│
├─ Monash University Mail...pdf           # Teaching Team Commendation Email
├─ README.md                              # English Documentation
└─ README.zh.md                           # Chinese Documentation
```

## 4. About

* **Author:** Zihan Yin
* **Education:**
    * **Master of Data Science**, Monash University
    * **Bachelor of Science**, University of Melbourne (Major in Data Science, Minor in Statistics & Stochastic Processes)
* **Contact:** [2002317yzh@gmail.com](mailto:2002317yzh@gmail.com)
* **Github:** [https://github.com/ZihanYin12138](https://github.com/ZihanYin12138)

If you are interested in this project, feel free to Star ⭐️ or Fork!
