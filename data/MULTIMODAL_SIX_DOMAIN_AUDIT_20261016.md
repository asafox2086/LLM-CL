# 六域多模态身份、金融替换与交通域依据

核验日期：2026-10-16。当前实验` six_domain_cl_20261016_finmme_coda `；本文覆盖旧五任务方案中金融来源和任务数，旧调查记录仅作历史来源。

## 多模态能证明到什么程度

每个任务提供原图像与问题文本，当前数据适配器和Qwen2-VL输入链路均保留这两个输入。图像不是将另一份纯文本问题拍照造出：自然照片、医学影像、遥感影像、金融统计图、画作、道路照片分别具有自己的视觉内容。

这可证明任务是图文多模态，但不能证明每题必须看图。AQUA的知识题、SLAKE知识题和各源答案偏置均可能产生语言捷径。尊重源构成，不为声称视觉依赖而删题。若后续论文要主张学习到了视觉能力，需要固定相同测试ID另做遮图／错配图对照；本次不伪称已有该结果。

| 来源 | 原始发表／视觉依据 | 独立实际使用依据与边界 |
|---|---|---|
| TextVQA | Singh等，[Towards VQA Models That Can Read，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html)。原照片、问题和场景文字读取。 | CoIN（NeurIPS 2024 D&B）、SMoLoRA（ICCV 2025）的训练任务，详细页和正文位置见[原调查](MULTIDOMAIN_DATASET_SURVEY_20261016.md)。 |
| SLAKE | Liu等，[作者医学VQA页](https://www.med-vqa.com/slake/)，ISBI 2021。医学图像、语义标注和中英问题。 | Uncertainty-Driven Expert Control，ICCV 2025，Table1实际医学VQA评测。其通常用英语，本实验保留官方双语，数量不可混写。ISBI首发不写成ICCV首发。 |
| RSVQA-LR | Lobry等，[作者页](https://rsvqa.sylvainlobry.com/)，TGRS 2020，真实遥感图像及由OSM生成的问题答案。 | GeoChat（CVPR 2024）、LHRS-Bot（ECCV 2024）、RSAdapter（TGRS 2024）实际评测。部分论文分箱数值，本次保留作者原数字。 |
| FinMME | Luo等，[ACL 2025主会](https://aclanthology.org/2025.acl-long.1426/)。真实金融图表、问题、选项与答案。 | [Beyond Real Weights（WACV 2026正式论文）](https://openaccess.thecvf.com/content/WACV2026/papers/Ibn_Ahad_Beyond_Real_Weights_Hypercomplex_Representations_for_Stable_Quantization_WACV_2026_paper.pdf)，[作者公开正文](https://arxiv.org/html/2512.08524)，§4.1、Table2：训练caption后在FinMME评测，不是FinMME监督训练。[AgentFinVQA](https://arxiv.org/html/2606.19782)，§4、Table1实际评测；2026预印本，未核实正式会刊，不能写成第二篇正式顶会复用。 |
| AQUA | Garcia等，[作者ArtVQA](https://github.com/noagarcia/ArtVQA)，ECCV VISART Workshop 2020；原画作、视觉和知识问题。 | CL-CrossVQA（WACV 2025）实际使用视觉子集；ArtQuest（WACV 2024）研究艺术VQA偏置。两者不能证明全部AQUA知识题视觉必需，也不能说首发是ECCV主会。 |
| CODA-LM | Chen等，[Automated Evaluation of Large Vision-Language Models on Self-Driving Corner Cases（WACV 2025）](https://openaccess.thecvf.com/content/WACV2025/html/Chen_Automated_Evaluation_of_Large_Vision-Language_Models_on_Self-Driving_Corner_Cases_WACV_2025_paper.html)，[作者代码／原始配置](https://github.com/DLUT-LYZ/CODA-LM)。真实道路图、一般感知、区域感知、驾驶建议。 | [MPDrive，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_MPDrive_Improving_Spatial_Understanding_with_Marker-Based_Prompt_Learning_for_Autonomous_CVPR_2025_paper.html)，Table2实际CODA-LM评测；[RoboTron-Drive，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_RoboTron-Drive_All-in-One_Large_Multimodal_Model_for_Autonomous_Driving_ICCV_2025_paper.pdf)，§4.1、Table3将CODA-LM列入六个训练／评测来源之一，报告三类任务分数。不是只列参考文献。 |

## 金融：已替换为FinMME

原MME-Finance并非没有发表（ACM MM 2025），被替换原因是此前未补齐足够独立复用证据及当前用户明确要求。新源FinMME首发ACL 2025主会，独立正式评测WACV 2026已核实；另有实际预印本评测，发表性质分开记录，不能统称多篇顶会复用。

- 作者数据仓库：[luojunyu/FinMME](https://huggingface.co/datasets/luojunyu/FinMME)。原Parquet398554212字节，SHA256 `e673e2df4f795b9e96d4895a235f47de0c20c97c093220e9f7086f63a984f26d`，完整下载并核对。
- 实际11099 QA、4450个不同图像SHA256；逐图PIL解码核验完成。原题型：6567 single_choice、2680 multiple_choice、1852 numerical。两条multiple_choice金答案为空（ID3698、9806）；保留审计，不作监督或评分，不按题型筛选。
- 有标注11097条；seed42图像组80/20研究划分：8864训练、2233测试、1%89。图像组不跨训练／测试。作者发布只有名为train的池，不能称其具有官方train/test。
- 原图、问题、选项、答案、单位与容差保留。verified_caption、related_sentences可能揭示答案，仅保存在元数据，不进入模型提示。

## 第六域：交通／自动驾驶CODA-LM

交通决策是独立应用领域，不以另一个通用自然图片集或自然科学题充数。CODA-LM依赖真实道路状态、风险对象和空间关系；区域题使用作者发布的红框图。

- 采用作者[English LLaVA配置](https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format)：完整文件实际Train20495、Test2123 QA，与源卡一致；1%205。训练原场景4884、测试500；QA数与原图数分开报告，区域标记图是额外图像表示。Val18455 QA、Mini193 QA不使用。36个原始Parquet已全部下载、逐文件核对作者LFS SHA256；全部QA非空金答案及图像／问题输入检查通过，六域清单已冻结。
- 本次只下载上述Train/Test：36个Parquet，共17926724529字节（约16.70GiB），是嵌入图像的存储量，不代表有上百万样本。作者Parquet会在多个QA记录中重复存图；准备时按原图字节SHA256落盘去重，内容不变。
- 作者2024-10已公开Test答案。代码中早期convert2vqa仍有不导出Test答案的分支，因此以作者已发布的LLaVA Test实际非空答案为准逐条检查，不靠旧代码推断测试答案隐藏。
- 选择English是作者正式配置，不按题目自行过滤。general／region／suggestion原任务构成均保留，全部作为同一个交通域任务；原长答案不缩写为自造标签。
- 原论文采用LLM评审。统一EM/F1/ROUGE-L用于本实验跨域比较，不等同原生驾驶语义评分，尤其EM对长描述严苛。

完整Test实际构成：general500、regional1123、suggestion500。两条件测试文件SHA256相同，1%训练ID严格包含于完整训练ID。固定清单和协议在`data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/`，资产与全量QA文件仅留服务器，Git只保存审计和清单元数据。

DriveLM同样是真实多模态，但当前HF释放受限，且训练QA规模比本次预期大；不将它当作已下载的新域。CODA-LM有公开原始图文Train/Test、更合适的QA规模及CVPR/ICCV独立实验依据。
