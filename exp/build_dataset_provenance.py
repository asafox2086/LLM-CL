"""Generate the reviewed provenance catalog; performs no network access or raw-data edits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = {}

def paper(key, title, authors, year, venue, url, status='published', doi=None):
    P[key] = dict(key=key, title=title, authors=authors.split(';'), year=year,
                  venue=venue, url=url, publication_status=status, doi=doi)

paper('goyal2017vqav2','Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering','Yash Goyal;Tejas Khot;Douglas Summers-Stay;Dhruv Batra;Devi Parikh',2017,'CVPR','https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_v_CVPR_2017_paper.pdf')
paper('hudson2019gqa','GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering','Drew A. Hudson;Christopher D. Manning',2019,'CVPR','https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html')
paper('singh2019textvqa','Towards VQA Models That Can Read','Amanpreet Singh;Vivek Natarajan;Meet Shah;Yu Jiang;Xinlei Chen;Dhruv Batra;Devi Parikh;Marcus Rohrbach',2019,'CVPR','https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html')
paper('rajpurkar2016squad','SQuAD: 100,000+ Questions for Machine Comprehension of Text','Pranav Rajpurkar;Jian Zhang;Konstantin Lopyrev;Percy Liang',2016,'EMNLP','https://aclanthology.org/D16-1264/',doi='10.18653/v1/D16-1264')
paper('benabacha2019medquad','A Question-Entailment Approach to Question Answering','Asma Ben Abacha;Dina Demner-Fushman',2019,'BMC Bioinformatics','https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/')
paper('lau2018vqarad','A dataset of clinically generated visual questions and answers about radiology images','Jason J. Lau;Soumya Gayen;Asma Ben Abacha;Dina Demner-Fushman',2018,'Scientific Data','https://pmc.ncbi.nlm.nih.gov/articles/PMC6244189/',doi='10.1038/sdata.2018.251')
paper('he2020pathvqa','PathVQA: 30000+ Questions for Medical Visual Question Answering','Xuehai He;Yichen Zhang;Luntian Mou;Eric Xing;Pengtao Xie',2020,'arXiv:2003.10286','https://arxiv.org/abs/2003.10286',status='preprint; dataset introduction, not asserted to be a published conference paper')
paper('liu2021slake','SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering','Bo Liu;Li-Ming Zhan;Li Xu;Lin Ma;Yan Yang;Xiao-Ming Wu',2021,'ISBI','https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/')
paper('hendrycks2021cuad','CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review','Dan Hendrycks;Collin Burns;Anya Chen;Spencer Ball',2021,'NeurIPS Datasets and Benchmarks','https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html')
paper('chen2021finqa','FinQA: A Dataset of Numerical Reasoning over Financial Data','Zhiyu Chen;Wenhu Chen;Charese Smiley;Sameena Shah;Iana Borova;Dylan Langdon;Reema Moussa;Matt Beane;Ting-Hao Huang;Bryan Routledge;William Yang Wang',2021,'EMNLP','https://aclanthology.org/2021.emnlp-main.300/',doi='10.18653/v1/2021.emnlp-main.300')
paper('zhu2022tatdqa','Towards Complex Document Understanding By Discrete Reasoning','Fengbin Zhu;Wenqiang Lei;Fuli Feng;Chao Wang;Haozhou Zhang;Tat-Seng Chua',2022,'ACM Multimedia','https://arxiv.org/abs/2207.11871')
paper('garcia2020aqua','A Dataset and Baselines for Visual Question Answering on Art','Noa Garcia;Chentao Ye;Zihua Liu;Qingtao Hu;Mayu Otani;Chenhui Chu;Yuta Nakashima;Teruko Mitamura',2020,'ECCV VISART workshop','https://github.com/noagarcia/ArtVQA')
paper('garcia2018semart','How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval','Noa Garcia;George Vogiatzis',2018,'ECCV workshop','https://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Garcia_How_to_Read_Paintings_Semantic_Art_Understanding_with_Multi-Modal_Retrieval_ECCVW_2018_paper.pdf')
paper('masry2022chartqa','ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning','Ahmed Masry;Do Xuan Long;Jia Qing Tan;Shafiq Joty;Enamul Hoque',2022,'Findings of ACL','https://aclanthology.org/2022.findings-acl.177/',doi='10.18653/v1/2022.findings-acl.177')
paper('mathew2021docvqa','DocVQA: A Dataset for VQA on Document Images','Minesh Mathew;Dimosthenis Karatzas;C. V. Jawahar',2021,'WACV','https://openaccess.thecvf.com/content/WACV2021/html/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.html')
paper('kuckreja2024geochat','GeoChat: Grounded Large Vision-Language Model for Remote Sensing','Kartik Kuckreja;Muhammad Sohail Danish;Muzammal Naseer;Abhijit Das;Salman Khan;Fahad Shahbaz Khan',2024,'CVPR','https://github.com/mbzuai-oryx/GeoChat')

paper('wang2025smolora','SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning','Ziqi Wang;Chang Che;Qi Wang;Yangyang Li;Zenglin Shi;Meng Wang',2025,'ICCV','https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html')
paper('wang2026cope','CoPE: Continual Probe-guided Expansion for Large Vision-Language Models','Ziqin Wang;Hengyuan Zhao;Qixin Sun;Kaiyou Song;Yilin Li;Xiaolin Hu;Qingpei Guo;Linjiang Huang;Si Liu',2026,'ICML','https://proceedings.mlr.press/v306/wang26le.html')
paper('devlin2019bert','BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding','Jacob Devlin;Ming-Wei Chang;Kenton Lee;Kristina Toutanova',2019,'NAACL','https://aclanthology.org/N19-1423/',doi='10.18653/v1/N19-1423')
paper('sun2020mobilebert','MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices','Zhiqing Sun;Hongkun Yu;Xiaodan Song;Renjie Liu;Yiming Yang;Denny Zhou',2020,'ACL','https://aclanthology.org/2020.acl-main.195/',doi='10.18653/v1/2020.acl-main.195')
paper('cho2025kcomp','K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor','Jeonghun Cho;Gary Lee',2025,'NAACL','https://aclanthology.org/2025.naacl-long.351/',doi='10.18653/v1/2025.naacl-long.351')
paper('vazhentsev2025truth','Unconditional Truthfulness: Learning Unconditional Uncertainty of Large Language Models','Artem Vazhentsev;Ekaterina Fadeeva;Rui Xing;Gleb Kuzmin;Ivan Lazichny;Alexander Panchenko;Preslav Nakov;Timothy Baldwin;Maxim Panov;Artem Shelmanov',2025,'EMNLP','https://aclanthology.org/2025.emnlp-main.1807/',doi='10.18653/v1/2025.emnlp-main.1807')
paper('liang2025expertcfg','Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models','Xiao Liang;Di Wang;Zhicheng Jiao;Ronghan Li;Pengfei Yang;Quan Wang;Tat-Seng Chua',2025,'ICCV','https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.pdf')
paper('pan2026ducor','Dual-Level Confidence based Implicit Self-Refinement for Medical Visual Question Answering','Meihong Pan;Yefeng Zheng',2026,'CVPR','https://openaccess.thecvf.com/content/CVPR2026/html/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html')
paper('santosh2025cocolex','CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation','Santosh T.Y.S.S;Youssef Tarek Elkhayat;Oana Ichim;Pranav Shetty;Dongsheng Wang;Zhiqiang Ma;Armineh Nourbakhsh;Xiaomo Liu',2025,'ACL','https://aclanthology.org/2025.acl-long.931/',doi='10.18653/v1/2025.acl-long.931')
paper('shin2025multidocfusion','MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents','Joongmin Shin;Chanjun Park;Jeongbae Park;Jaehyung Seo;Heuiseok Lim',2025,'EMNLP','https://aclanthology.org/2025.emnlp-main.1062/',doi='10.18653/v1/2025.emnlp-main.1062')
paper('sun2024apollo','APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning','Jiashuo Sun;Hang Zhang;Chen Lin;Xiangdong Su;Yeyun Gong;Jian Guo',2024,'LREC-COLING','https://aclanthology.org/2024.lrec-main.122/')
paper('li2023dyrren','DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data','Xiao Li;Yin Zhu;Sichen Liu;Jiangzhou Ju;Yuzhong Qu;Gong Cheng',2023,'AAAI','https://ojs.aaai.org/index.php/AAAI/article/view/26543',doi='10.1609/aaai.v37i11.26543')
paper('zhu2024doc2soar','Doc2SoarGraph: Discrete Reasoning over Visually-Rich Table-Text Documents via Semantic-Oriented Hierarchical Graphs','Fengbin Zhu;Chao Wang;Fuli Feng;Zifeng Ren;Moxin Li;Tat-Seng Chua',2024,'LREC-COLING','https://aclanthology.org/2024.lrec-main.456/')
paper('faysse2025colpali','ColPali: Efficient Document Retrieval with Vision Language Models','Manuel Faysse;Hugues Sibille;Tony Wu;Bilel Omrani;Gautier Viaud;Celine Hudelot;Pierre Colombo',2025,'ICLR','https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf')
paper('bleidt2024artquest','ArtQuest: Countering Hidden Language Biases in ArtVQA','Tibor Bleidt;Sedigheh Eslami;Gerard de Melo',2024,'WACV','https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf')
paper('lee2023pix2struct','Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding','Kenton Lee;Mandar Joshi;Iulia Raluca Turc;Hexiang Hu;Fangyu Liu;Julian Martin Eisenschlos;Urvashi Khandelwal;Peter Shaw;Ming-Wei Chang;Kristina Toutanova',2023,'ICML','https://proceedings.mlr.press/v202/lee23g.html')
paper('liu2023deplot','DePlot: One-shot visual language reasoning by plot-to-table translation','Fangyu Liu;Julian Eisenschlos;Francesco Piccinno;Syrine Krichene;Chenxi Pang;Kenton Lee;Mandar Joshi;Wenhu Chen;Nigel Collier;Yasemin Altun',2023,'Findings of ACL','https://aclanthology.org/2023.findings-acl.660/',doi='10.18653/v1/2023.findings-acl.660')
paper('kim2022donut','OCR-free Document Understanding Transformer','Geewook Kim;Teakgyu Hong;Moonbin Yim;Jeongyeon Nam;Jinyoung Park;Jinyeong Yim;Wonseok Hwang;Sangdoo Yun;Dongyoon Han;Seunghyun Park',2022,'ECCV','https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf')
paper('irvin2025teochat','TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data','Jeremy Irvin;Emily Liu;Joyce Chen;Ines Dormoy;Jinyoung Kim;Samar Khanna;Zhuo Zheng;Stefano Ermon',2025,'ICLR','https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html')

D=[]
def use(key, purpose, locator, detail, evidence_url=None):
    return dict(paper_key=key, relation='actual_experimental_use', purpose=purpose,
                evidence_locator=locator, evidence_summary=detail,
                evidence_url=evidence_url or P[key]['url'], verified_on='2026-10-12')
def dataset(folder, domain, modality, original, official, uses, split, caveat, status='candidate; final ten-task protocol not locked'):
    D.append(dict(dataset=folder, domain=domain, modality=modality, original_paper_key=original,
                  official_source=official, later_uses=uses, split_policy=split,
                  limitations=caveat, experiment_status=status))

sm=use('wang2025smolora','continual instruction tuning and evaluation','Main paper: upstream continual learning section; results table 1','TextVQA, GQA and VQAv2 are sequential upstream tasks with per-dataset results; includes additional classification tasks that we do not adopt.')
co=use('wang2026cope','continual instruction tuning and evaluation','Main paper: table 1 and task-order table 3','CoIN results explicitly report TextVQA, GQA and VQAv2; not just references in the bibliography.')
for folder,orig,url in [('VQAv2','goyal2017vqav2','https://visualqa.org/'),('GQA','hudson2019gqa','https://cs.stanford.edu/people/dorarad/gqa/'),('TextVQA','singh2019textvqa','https://textvqa.org/')]:
    dataset(folder,'自然/日常场景','image+question',orig,url,[sm,co],'优先公开官方划分；hidden test 不造标签；开放式任务处理需记录与论文的差异。','目前本地 JSONL 为开放式派生视图，不等同于论文全部任务或原始完整计数；下载镜像不是提出者。')
dataset('SQuAD11','自然/通用阅读','text','rajpurkar2016squad','https://rajpurkar.github.io/SQuAD-explorer/',[
 use('devlin2019bert','fine-tuning and evaluation','Abstract; SQuAD experiment subsection','Explicit SQuAD v1.1 QA fine-tuning and test F1; also v2.0, which is a distinct version.'),
 use('sun2020mobilebert','fine-tuning and evaluation','SQuAD v1.1 results table','SQuAD v1.1 is used to evaluate the compressed model.')],
 '官方 train/dev；公开 dev 只能固定承担一种开发或最终评测用途，不能同时调参和报最终成绩。',
 'SQuAD 是广泛主题的 Wikipedia 阅读集，不是纯自然科学；此前人工 General/Geography 分域不作为当前正式方案。')
dataset('MedQuAD','医学','text','benabacha2019medquad','https://github.com/abachaa/MedQuAD',[
 use('cho2025kcomp','medical QA training and evaluation','Appendix A.2','MedQuAD has no provided split in this protocol; authors randomly split 80/10/10.'),
 use('vazhentsev2025truth','uncertainty estimation training/evaluation','Appendix dataset descriptions; tables 20/21','MedQuAD is a long-answer QA dataset, with a 500 training / 1000 evaluation budget in this work.')],
 '可参考 K-COMP 的 80/10/10；随机种子未核实，不能声称逐条精确复现。',
 '原论文 collection 47,457 QA 不等于当前公开有答案数量；部分答案因版权移除，当前约 16,407 非空公开回答，不能补造。')
med1=use('liang2025expertcfg','medical VQA fine-tuning and reliability evaluation','Experiments; section 5.2 figure 3; official repository datasets','VQA-RAD, SLAKE and PathVQA are evaluated; open and closed questions have separate evaluation treatment.','https://github.com/ecoxial2007/Expert-CFG')
med2=use('pan2026ducor','medical VQA and cross-domain adaptation','Section 4.1; open/closed per-dataset results table, proceedings p.17219','All three datasets are evaluated. Standard split and cross-domain adaptation are distinct settings; pseudo-label adaptation is not our default evaluation protocol.','https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Dual-Level_Confidence_based_Implicit_Self-Refinement_for_Medical_Visual_Question_Answering_CVPR_2026_paper.pdf')
for folder,orig,url in [('VQA_RAD','lau2018vqarad','https://osf.io/89kps/'),('PathVQA','he2020pathvqa','https://github.com/UCSD-AI4H/PathVQA'),('SLAKE','liu2021slake','https://www.med-vqa.com/slake/')]:
    dataset(folder,'医学','image+question',orig,url,[med1,med2],'沿用明确公开的划分和 open-ended 标签；格式化镜像版本须单列。','医学开放式 QA 子集和 yes/no 子集不同；原论文计数、镜像计数、派生子集计数须分开。PathVQA 2020 introduction 的记录不伪称顶会发表。')
dataset('CUAD','法律/合同','text; original PDFs available','hendrycks2021cuad','https://zenodo.org/records/4595826',[
 use('santosh2025cocolex','grounded legal answer generation','Dataset description and CUAD results','Extractive answer spans are concatenated for reference generation; multiple spans can be required components, not alternative answers.'),
 use('shin2025multidocfusion','layout/OCR-based document RAG evaluation','Section 4; tables 2/3; appendix A.1','CUAD official test contracts are used for retrieval and answer generation; parsed text chunks feed the final LLM.')],
 '官方 train/test；图像连续训练任务仍需独立核验，不能把测试文档全部用于训练。',
 '510 原始合同 PDF 已下载。MultiDocFusion 文档解析管线不等于已验证端到端 VLM 训练协议；文本与 PDF 使用同一合同需严格保持 split 一致。')
dataset('FinQA','金融','text+serialized tables','chen2021finqa','https://github.com/czyssrs/FinQA',[
 use('sun2024apollo','retriever-generator training and numerical evaluation','Abstract; FinQA/ConvFinQA experiment results','Evaluates FinQA and ConvFinQA leaderboard tasks; program/execution measures are task-specific.'),
 use('li2023dyrren','numerical QA training and evaluation','AAAI abstract; official repository training/inference scripts','FinQA is the actual training/evaluation dataset for dynamic retriever-reranker-generation.','https://github.com/nju-websoft/dyrren')],
 '官方 train/dev/test，不自行重划分。','qa.answer 与 exe_ans 可能尺度不同；保留原始单位和程序标注，不能把 gold_inds 或参考程序放入问答输入。')
dataset('TAT_DQA','金融','document image+question','zhu2022tatdqa','https://github.com/NExTplusplus/TAT-DQA',[
 use('zhu2024doc2soar','document QA training and evaluation','Abstract; TAT-DQA experimental section/results','Discrete reasoning on financial document QA with semantic-oriented hierarchical graphs.'),
 use('faysse2025colpali','document retrieval evaluation, not QA generation','Table 1; appendix dataset descriptions','ViDoRe contains 1600 TAT-DQA query/document pairs for retrieval. This supports reuse, not a generation-score comparison.')],
 '官方 train/dev/test_gold，保留 answer、scale 与页对应关系。','TAT-QA 纯文本表格任务与 TAT-DQA 文档图像任务不能混淆；原始数据 16,558 QA，派生视图计数另报。')
dataset('AQUA','艺术与文化遗产','painting+question; knowledge QA subset','garcia2020aqua','https://github.com/noagarcia/ArtVQA',[
 use('bleidt2024artquest','ArtVQA evaluation and question-only bias analysis','Sections on visual and knowledge AQUA questions; comparison tables','Evaluates existing AQUA rather than merely citing it; finds substantial language bias.')],
 '官方 train/val/test，knowledge 标记和原始 QA 保留。','原始工作是 ECCV workshop；已核实 1 篇独立后续 QA 评测，加原始论文共 2 篇实验工作。知识题不能未经协议核实就当成独立文本阅读任务；不能宣称所有视觉题必须看图。',status='fifth-domain candidate; QA files downloaded; image download in progress')
dataset('SemArt','艺术与文化遗产','painting images+artistic comments','garcia2018semart','https://researchdata.aston.ac.uk/380/1/SemArt.zip',[
 use('garcia2020aqua','source corpus for QA benchmark construction','Official AQUA README; task definition','Provides original paintings/comments underlying AQUA questions.'),
 use('bleidt2024artquest','source corpus for art QA benchmark construction','Section 5','Uses SemArt paintings and structured attributes to construct ArtQuest.')],
 '沿用对应 AQUA/SemArt 官方划分关系。','SemArt 本身是图文检索/理解资源，不能单独冒充人工开放问答集；不采用 ArtQuest 固定属性识别来凑分类任务。',status='supporting image/comment corpus, not a separate CL task')
dataset('ChartQA','辅助图表评测，不自动归金融','chart image+question','masry2022chartqa','https://github.com/vis-nlp/ChartQA',[
 use('lee2023pix2struct','fine-tuning and QA evaluation','Main results table; visual QA setup','ChartQA is a downstream visual QA benchmark.'),
 use('liu2023deplot','plot-to-table and numerical QA evaluation','ChartQA human/augmented query results','Distinguishes human-written and augmented ChartQA questions; original publication is Findings of ACL 2023, not ICML.')],
 '官方 train/val/test，human/augmented 分别报告。','图表形式不等于金融领域；目前仅辅助资源。',status='auxiliary; downloaded')
dataset('DocVQA','辅助文档评测，不自动归法律','document image+question','mathew2021docvqa','https://www.docvqa.org/',[
 use('lee2023pix2struct','document QA fine-tuning/evaluation','Main results table and official dataset preparation','DocVQA is an actual downstream QA benchmark.'),
 use('kim2022donut','document QA fine-tuning/evaluation','DocVQA experiment subsection/results','OCR-free document understanding is evaluated on DocVQA.')],
 '官方可公开标注划分；hidden test 不造标签。','单页 DocVQA 与 Multi-page DocVQA 不是同一个实验；文档外观不能据此改成法律领域。',status='auxiliary; downloaded')
dataset('GeoChat_Instruct','遥感，当前五域外备用','remote sensing image+instruction','kuckreja2024geochat','https://github.com/mbzuai-oryx/GeoChat',[
 use('irvin2025teochat','multimodal instruction training','Section 3.1.1 and training-data appendix','Explicitly incorporates GeoChat instructions in training.','https://proceedings.iclr.cc/paper_files/paper/2025/file/ac3af725ae398b6184faae0828bdbd6c-Paper-Conference.pdf')],
 '当前不构建第五域任务；如以后使用须采用论文明确的任务协议。','原始混合集含分类和定位，不能整包算开放 QA；本次 pinned JSON 实际 308,861 条与 318k brochure 不同。',status='backup raw assets; not in current five domains')

catalog={'updated_on':'2026-10-12','scope':'New five-domain assets and previously downloaded auxiliary multimodal datasets; not the complete historical SuperNI pilot bibliography.',
 'citation_policy':'Original introduction and independent later experiments are separate. Actual use does not automatically authorize copying preprocessing or establish an identical task. This catalog contains paraphrased evidence summaries, not reproduced paper passages.',
 'datasets':D,'papers':P,
 'excluded':[{'dataset':'SEC_Contracts_VQA','reason':'No verified original peer-reviewed introduction and multiple independent experimental-use evidence; automatic field labels are not used as the formal legal task.'},
 {'dataset':'QASPER / SPIQA','reason':'User rejected the computer-science domain; downloads stopped; not adopted.'},
 {'dataset':'SQuAD11_General / SQuAD11_Geography','reason':'Our former manual derived subsets are not standalone author-proposed benchmarks and are not the current formal protocol.'}]}
data=ROOT/'data';data.mkdir(exist_ok=True)
(data/'dataset_provenance.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def citation(key):
    p=P[key];return f"[{p['authors'][0]} 等：{p['title']}]({p['url']})（{p['venue']} {p['year']}）"
intro=['# 数据集提出者与实际使用证据','', '更新：2026-10-12。用于论文数据集部分的溯源，**不是十任务已经全部训练就绪的声明**。', '',
 '原始提出论文与后续使用论文分开列出；训练、回答生成评测、检索评测和基准构建分别注明。只把有实验正文/结果/作者代码依据的工作计入使用记录。原始数据集允许早于 2024；2024–2026 是持续学习方法论文收集范围。', '',
 '机器可读记录：[dataset_provenance.json](dataset_provenance.json)；引用：[dataset_references.bib](dataset_references.bib)。以下作者均为原论文作者，格式化镜像上传者不是数据集提出者。','',
 '| 数据集 / 领域 | 谁提出 / 原始发表 | 谁实际使用 / 用途 | 当前角色 |','|---|---|---|---|']
for d in D:
    later='<br>'.join(citation(u['paper_key'])+'：'+u['purpose'] for u in d['later_uses'])
    intro.append(f"| {d['dataset']} / {d['domain']} | {citation(d['original_paper_key'])} | {later} | {d['experiment_status']} |")
intro+=['','## 各数据集的证据与处理界限','']
for d in D:
    lines=[f"### {d['dataset']}",'',f"提出者：{'、'.join(P[d['original_paper_key']]['authors'])}。原始论文：{citation(d['original_paper_key'])}。",'',f"官方资源：{d['official_source']}",'',f"划分：{d['split_policy']}",'',f"限制与版本：{d['limitations']}",'', '后续实际使用证据：','']
    for u in d['later_uses']:
        lines.append(f"- {citation(u['paper_key'])}。用途：{u['purpose']}。定位：{u['evidence_locator']}。证据概述：{u['evidence_summary']} [依据]({u['evidence_url']})")
    intro.extend(lines+[''])
    folder=data/d['dataset'];folder.mkdir(exist_ok=True)
    (folder/'PROVENANCE.md').write_text('\n'.join([f"# {d['dataset']} 来源和实际使用证据",'',f"完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。",'']+lines[2:])+'\n',encoding='utf-8')
intro+=['## 不纳入正式实验的资源','']+[f"- {x['dataset']}：{x['reason']}" for x in catalog['excluded']]
intro+=['','## 写论文时可以怎样表述','',
 '“我们选用已有公开问答资源构建五领域、文本与图像交替的持续学习序列，并分别引用数据集提出工作和后续实际评测工作。自然图像问答资源 VQAv2、GQA 和 TextVQA 已被 SMoLoRA（ICCV 2025）与 CoPE（ICML 2026）用于持续视觉指令实验；医学 VQA 资源 VQA-RAD、SLAKE 和 PathVQA 已被 Expert-CFG（ICCV 2025）和 DuCoR（CVPR 2026）用于医学问答评测。”', '',
 '这段可描述选源依据，但文章正式实验设置必须另外补上实际采用的任务清单、版本、划分、开放式子集和计数。不能提前写“我们采用完全相同的划分”，也不能把备选/下载中资源写成已完成训练数据。艺术域的独立文本任务与法律图像训练协议尚未锁定。','']
(data/'DATASET_PROVENANCE.md').write_text('\n'.join(intro),encoding='utf-8')

def bib_name(name):
    for surname in ['Ben Abacha', 'de Melo']:
        if name.endswith(' '+surname):
            return surname+', '+name[:-len(surname)-1]
    parts=name.rsplit(' ',1)
    return parts[-1]+', '+parts[0] if len(parts)==2 else name
bib=[]
for key,p in P.items():
    kind='article' if p['venue'] in ['BMC Bioinformatics','Scientific Data'] else ('misc' if p['publication_status'].startswith('preprint') else 'inproceedings')
    fields={'title':p['title'],'author':' and '.join(bib_name(n) for n in p['authors']),'year':str(p['year']),'url':p['url']}
    fields['journal' if kind=='article' else 'note' if kind=='misc' else 'booktitle']=p['venue']
    if p['doi']:fields['doi']=p['doi']
    fields['title']='{'+fields['title']+'}'  # Preserve dataset and method acronym capitalization.
    bib.append('@'+kind+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}')
(data/'dataset_references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
assert len({d['dataset'] for d in D})==len(D)
assert all(u['paper_key'] in P for d in D for u in d['later_uses'])
print('PROVENANCE_CREATED',len(D),'datasets',len(P),'papers',sum(len(d['later_uses']) for d in D),'use records')

if __name__=='__main__':
    pass
