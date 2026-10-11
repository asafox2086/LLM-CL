import concurrent.futures, hashlib, html, json, re, subprocess, time
from pathlib import Path
from urllib.parse import urljoin
import requests

ROOT = Path('/data2/liyapeng_grp/program/LLMCL/paper')
records = []
def cv(method, year, slug, category, mechanism, scope, venue='CVPR'):
    page = f'https://openaccess.thecvf.com/content/{venue}{year}/html/{slug}_{venue}_{year}_paper.html'
    records.append(dict(method=method, year=year, venue=venue, page=page, category=category, mechanism=mechanism, scope=scope))
def other(method, title, year, venue, page, pdf, category, mechanism, scope):
    records.append(dict(method=method,title=title,year=year,venue=venue,page=page,pdf=pdf,category=category,mechanism=mechanism,scope=scope))

cv('Knowledge replay',2024,'Duan_Towards_Backward-Compatible_Continual_Learning_of_Image_Compression','replay','知识回放；兼有模型架构设计','预训练图像压缩模型的序贯微调')
cv('EIR',2025,'Yin_Beyond_Background_Shift_Rethinking_Instance_Replay_in_Continual_Semantic_Segmentation','replay','实例存储与粘贴回放，缓解背景漂移','持续语义分割')
cv('View-batch model',2025,'Kang_Do_Your_Best_and_Get_Enough_Rest_for_Continual_Learning','replay','调节样本再训练间隔；兼有自监督学习','通用持续学习训练策略（非专门 LLM 微调）')
cv('GIFT',2025,'Wu_Synthetic_Data_is_an_Elegant_GIFT_for_Continual_Vision-Language_Models','replay','扩散生成合成样本；兼有蒸馏和权重正则化','VLM 持续微调')
cv('Smart Replay',2026,'Chen_Smart_Replay_Adaptive_Scheduling_of_Memory_Rehearsal_for_Computational_Resource-Aware','replay','在计算预算下自适应调度回放比例','资源受限增量学习（通用训练策略）')
cv('UCo-VQA baseline',2026,'Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal','replay','仅问题回放；兼有双层蒸馏','持续 VQA；论文兼含评测基准')
cv('InfLoRA',2024,'Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning','regularization','限制低秩更新子空间，抑制任务干扰；兼有 LoRA 架构','ViT 参数高效持续微调')
cv('One-for-More',2025,'Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection','regularization','梯度投影约束保护历史知识','扩散模型持续异常检测')
cv('ConceptGuard',2025,'Guo_ConceptGuard_Continual_Personalized_Text-to-Image_Generation_with_Forgetting_and_Confusion_Mitigation','regularization','记忆保持正则；兼有 shift embedding 和概念绑定提示','扩散模型持续个性化')
cv('CoMasTRe',2024,'Gong_Continual_Segmentation_with_Disentangled_Objectness_Learning_and_Class_Recognition','regularization','objectness 蒸馏、多标签类别蒸馏；兼有两阶段结构','持续语义分割')
cv('LfI',2026,'Gong_Learning_from_Itself_Mining_Internal_Knowledge_from_Vision_Language_Models','regularization','自适应双向蒸馏；兼有伪字幕生成','CLIP 持续微调')
cv('EWC Done Right',2026,'Liu_Elastic_Weight_Consolidation_Done_Right_for_Continual_Learning','regularization','修正权重重要性估计的权重正则化','通用持续学习正则方法')
cv('CCL-DC',2024,'Wang_Improving_Plasticity_in_Online_Continual_Learning_via_Collaborative_Learning','regularization','蒸馏链和协同学习；可与回放方法结合','在线持续学习（非专门 LLM 微调）')
cv('MoE-Adapters / DDAS',2024,'Yu_Boosting_Continual_Learning_of_Vision-Language_Models_via_Mixture-of-Experts_Adapters','architecture','动态专家 Adapter 扩展和输入路由','CLIP 持续微调')
cv('CPrompt',2024,'Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning','architecture','任务提示与分类器；兼有一致性学习','冻结预训练模型的持续提示微调')
cv('S6MOD',2025,'Liu_Enhancing_Online_Continual_Learning_with_Plug-and-Play_State_Space_Model_and','architecture','附加状态空间分支与类别条件路由；兼有对比损失','在线持续学习结构插件')
cv('CHEEM',2026,'Savadikar_CHEEM_Continual_Learning_by_Reuse_New_Adapt_and_Skip_-','architecture','架构搜索选择复用、新建、适配和跳过','预训练 ViT 持续适配')
cv('GRAPA',2026,'Chen_Parameter-efficient_Continual_Learning_for_Enhancing_Plasticity_without_Forgetting_under_Limited','architecture','梯度引导参数复用和强化学习参数分配','有限容量下参数高效持续学习')
cv('MedQwen / Sparse Spectral LoRA',2026,'Nejatimanzari_Sparse_Spectral_LoRA_Routed_Experts_for_Medical_VLMs','architecture','谱分段 LoRA 专家和路由','医学 VLM 序贯微调')
other('Learn more, but bother less','Learn more, but bother less: parameter efficient continual learning',2024,'NeurIPS','https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0bc711f48724237b38823c4d9cee10b-Abstract-Conference.html','https://proceedings.neurips.cc/paper_files/paper/2024/file/b0bc711f48724237b38823c4d9cee10b-Paper-Conference.pdf','regularization','梯度投影保持低秩子空间正交；兼有敏感度初始化','LLM 持续任务微调')

cv('MedCoSS',2024,'Ye_Continual_Self-supervised_Learning_Towards_Universal_Multi-modal_Medical_Data_Representation_Learning','pretraining','跨模态自监督预训练；回放、特征蒸馏和 mixup','医学多模态序贯预训练')
cv('CoSMIC',2025,'Liu_CoSMIC_Continual_Self-supervised_Learning_for_Multi-Domain_Medical_Imaging_via_Conditional','pretraining','条件互信息最大化和 Anatomy-Guided Calibration','医学多域序贯自监督预训练',venue='ICCV')
cv('CHIPS',2026,'Zhuang_CHIPS_Efficient_CLIP_Adaptation_via_Curvature-aware_Hybrid_Influence-based_Data_Selection','pretraining','曲率和影响函数驱动的数据选择','CLIP 域持续预训练；单域扩展型 CPT')
other('128K CPT','Data Engineering for Scaling Language Models to 128K Context',2024,'ICML','https://proceedings.mlr.press/v235/fu24d.html','https://raw.githubusercontent.com/mlresearch/v235/main/assets/fu24d/fu24d.pdf','pretraining','领域平衡与长文本上采样','长上下文能力扩展型 CPT，非多任务遗忘专用方法')
other('DiffuGPT / DiffuLLaMA','Scaling Diffusion Language Models via Adaptation from Autoregressive Models',2025,'ICLR','https://proceedings.iclr.cc/paper_files/paper/2025/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2025/file/0fa81c3f0d57f95b8776de3a248ef0ed-Paper-Conference.pdf','pretraining','通过持续预训练从自回归模型适配扩散语言模型','模型范式转换型 CPT，非多任务遗忘专用方法')
other('Mix-CPT','Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment',2025,'ICLR','https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc721384c26c0bdff3ec31a7de31d8d5-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2025/file/cc721384c26c0bdff3ec31a7de31d8d5-Paper-Conference.pdf','pretraining','混合领域文本、通用指令和对齐数据，解耦知识与格式','领域适配型 CPT')
other('CPPO','CPPO: Continual Learning for Reinforcement Learning with Human Feedback',2024,'ICLR','https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2024/file/6246f93ebf4f2d76ad2bcb3158220d34-Paper-Conference.pdf','alignment','按奖励和生成概率加权学习及知识保持，持续奖励模型','动态人类偏好下持续 RLHF（核心持续对齐）')
other('SeRA','SeRA: Self-Reviewing and Alignment of LLMs using Implicit Reward Margins',2025,'ICLR','https://proceedings.iclr.cc/paper_files/paper/2025/hash/fdcf2565ea86530d65b3622c06d90841-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2025/file/fdcf2565ea86530d65b3622c06d90841-Paper-Conference.pdf','alignment','隐式奖励边际筛样与迭代偏好数据自举','迭代持续偏好优化（扩展阅读，非序贯多任务遗忘基准）')

folders={'pretraining':'pre_training','alignment':'pre_alignment','replay':'pre_finetuning/method_replay','regularization':'pre_finetuning/method_regularization','architecture':'pre_finetuning/method_architecture'}
def get(url):
    for attempt in range(4):
        try:
            r=requests.get(url,timeout=120); r.raise_for_status(); return r
        except Exception:
            if attempt==3: raise
            time.sleep(2*(attempt+1))
def download(r):
    page=get(r['page']).text
    if 'openaccess.thecvf.com' in r['page']:
        title=re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"',page)
        if not title:
            title=re.search(r'<div id="papertitle">\s*(.*?)\s*</div>',page,re.S)
        if not title: raise ValueError('No title '+r['page'])
        r['title']=html.unescape(re.sub('<.*?>','',title.group(1))).strip()
        links=re.findall(r'href="([^"]+\.pdf)"',page)
        r['pdf']=urljoin(r['page'],next(x for x in links if '/papers/' in x))
    slug=re.sub(r'[^A-Za-z0-9]+','_',r['method']).strip('_')
    path=Path(folders[r['category']])/f"{r['year']}_{r['venue']}_{slug}.pdf"
    dest=ROOT/path; dest.parent.mkdir(parents=True,exist_ok=True)
    data=dest.read_bytes() if dest.exists() else get(r['pdf']).content
    if not data.startswith(b'%PDF-'): raise ValueError('Not PDF '+r['pdf'])
    dest.write_bytes(data)
    subprocess.run(['pdftotext',str(dest),'-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    r.update(file=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),verified_on='2026-10-11')
    print('OK',r['method'],len(data),flush=True)
    return r

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        verified=list(pool.map(download,records))
    counts={k:sum(r['category']==k for r in verified) for k in folders}
    assert sum(counts[k] for k in ['replay','regularization','architecture'])==20
    (ROOT/'continual_learning_manifest.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text='''# 2024–2026 顶会持续学习论文索引

检索及下载核验日期：2026-10-11。收录 28 篇已在正式会议主会发表的论文：持续预训练 6 篇、持续微调 20 篇、持续对齐 2 篇；其中 CVPR 21 篇、ICCV 1 篇、ICML 1 篇、ICLR 4 篇、NeurIPS 1 篇。仅收录经 CVF / 正式会议 proceedings 页面核实的主会论文，不将 workshop、Findings 或未录用预印本当成主会论文。年份按会议发表年份。

## 目录与分类口径

```text
paper/
├── README.md
├── continual_learning_manifest.json
├── collect_cl_papers.py
├── pre_training/                  # 持续预训练：6
├── pre_finetuning/                # 持续微调：20
│   ├── method_replay/             # 回放：6
│   ├── method_regularization/     # 正则化：8
│   └── method_architecture/       # 架构：6
└── pre_alignment/                # 持续偏好／奖励对齐：2
```

这里的 `/paper` 指仓库内的 `paper/`；服务器实际路径为 `/data2/liyapeng_grp/program/LLMCL/paper`。原有医学论文文件保留，未计入本次 28 篇。

- **持续预训练**：用自监督／语言建模／图文对比目标继续训练。区分严格序贯多域预训练与领域、上下文、模型范式扩展型 continued pretraining；后者不能直接视为抗遗忘算法。
- **持续微调**：广义任务监督的序贯适配，包括 LLM、VLM、ViT、扩散模型、压缩、分割和在线 CL。并非全部为 LLM 指令微调；表格逐条标出适用对象。通用在线训练策略作为方法参考单独注明。
- **持续对齐**：偏好／奖励／RLHF 的持续或迭代更新。CPPO 是明确的动态偏好持续学习；SeRA 是论文明确描述的 continual alignment 迭代自举扩展阅读，不等价于标准多任务遗忘评测。普通视觉特征／跨模态表征对齐不据此归入本类。本轮未收录已核验的 2026 主会偏好持续对齐论文，避免以同名表征对齐或审稿稿件补数。
- **微调小类**按主要抗遗忘机制单选归档：回放涵盖真实样本／实例／生成样本；正则化涵盖权重约束、蒸馏、梯度投影和子空间约束；架构涵盖参数隔离、专家／Adapter／Prompt 和动态容量分配。混合方法在机制栏列次要机制。该归档是整理者基于摘要及方法的判断，不是作者统一分类。

## 方法 × 方向二维表

✓ 表示所属大类或归档小类；横轴同时列三大方向及持续微调的三小类。每行提供正式论文链接和本地 PDF。

| 方法 / 论文超链接 | 会议·年份 | 持续预训练 | 持续微调 | 微调·回放 | 微调·正则化 | 微调·架构 | 持续对齐 | 分类依据 / 适用对象 | PDF |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
'''
    for r in verified:
        c=r['category']; ft=c in ['replay','regularization','architecture']
        flags=['✓' if v else '—' for v in [c=='pretraining',ft,c=='replay',c=='regularization',c=='architecture',c=='alignment']]
        text+=f"| **{r['method']}** — [{r['title']}]({r['page']}) | {r['venue']} {r['year']} | "+' | '.join(flags)+f" | {r['mechanism']}；{r['scope']} | [PDF]({r['file']}) |\n"
    text+='''
## 数据与复现

`continual_learning_manifest.json` 保存标题、方法、会议、年份、归类、分类依据、正式来源页、原始 PDF URL、本地相对路径、字节数及 SHA-256。每个下载文件验证 `%PDF-` 文件头，并通过 `pdftotext` 完整解析。`collect_cl_papers.py` 可在服务器使用 Python 3、requests 和 Poppler 的 pdftotext 重新收集；运行会更新本索引及 manifest，不删除原有文件。

本清单是有明确筛选口径的精选阅读列表，非 2024–2026 全量系统综述。建议先阅读 InfLoRA、MoE-Adapters、GIFT 和 CPPO，再按模型与场景比较其他方法。
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    print('COMPLETE',counts,flush=True)
