# 001：SuperNI 七任务持续学习实验设置

协议标识：`superni_generation7_v2`
更新日期：2026-10-06  
状态：数据任务和配额已确认；用户已授权首篇论文实现与启动；首跑配置见第 13 节。

## 1. 范围与确认状态

本文替代 `exp/PROTOCOL_DRAFT.md` 的历史建议，为当前设置的唯一入口。路径均相对于项目根目录。

用户已确定：统一 Llama 基座；七个生成任务，每任务 1000/200/500；直接使用完整官方 SuperNI 原始任务；学一个任务、测试全部任务；暂不做 MixLoRA；归档原始源码；每篇论文一个方法文件和一个实验入口；结果保存至 `exp/result/`，由 `summary/` 汇总。

本文件给出确定性数据划分、可执行核验命令及训练流程规格。数据和划分指纹已核验，O-LoRA 执行器及数据清单生成已实现。第 5–12 节描述完整研究方案，其中未落实部分仍为建议；第 13 节锁定用户随后授权的首轮配置，首轮以第 13 节为准。模型权重尚需就绪，不能宣称已有 Llama 训练结果。

SAPT 仅作为方法、指标及任务顺序的参考，不使用其现成样本划分。之前讨论的 900/100/100 方案不适用。

## 2. 数据来源及下载复现

- 官方仓库：<https://github.com/allenai/natural-instructions>。
- 固定 commit：`55a365637381ce7f3748fa2eac7aef1a113bbb82`。
- 本地位置：`data/SuperNI_full/tasks/<task_id>.json`。
- 完整快照包含 1613 个任务、5,040,134 条 Instances。已核验 1662 个上游文件的 Git blob SHA-1，全部任务 JSON 可解析。
- 根 `.gitignore` 忽略 `/data/SuperNI_full/`。数据版本、文件哈希和派生划分指纹保存在 `rules/001_data_manifest.json`，可随 Git 版本管理。
- 原始文件不修改，完整下载不表示所有任务都参加实验。

在项目根目录恢复固定快照（目标必须尚不存在；遇错误停止）：

```bash
(
  set -eu
  test ! -e data/SuperNI_full
  mkdir -p data/SuperNI_full
  curl --fail --location --retry 3 \
    https://codeload.github.com/allenai/natural-instructions/tar.gz/55a365637381ce7f3748fa2eac7aef1a113bbb82 \
    -o data/SuperNI_full/upstream.tar.gz
  tar -xzf data/SuperNI_full/upstream.tar.gz \
    -C data/SuperNI_full --strip-components=1
)
```

已有数据直接执行第 4 节核验，无需覆盖。许可证随上游快照保留。上述命令恢复上游文件，不生成首次下载时本地额外保存的 upstream_commit.json、upstream_tree.json、verification.json。

SuperNI 官方 `splits/default/train_tasks.txt`、`test_tasks.txt` 按任务划分，研究跨任务泛化。本实验在选定任务内部划分实例，应称为“基于 SuperNI 官方原始数据的自定义持续学习实验”，不能标为官方泛化评测复现。

## 3. 已确定的任务与配额

| 任务 | task_id | 内容 | 上游实例数 | Train | Dev | Test | 未使用 |
|---|---|---|---:|---:|---:|---:|---:|
| Quoref | task002_quoref_answer_generation | 阅读理解答案生成 | 6500 | 1000 | 200 | 500 | 4800 |
| XSUM | task1290_xsum_summarization | 新闻摘要 | 6493 | 1000 | 200 | 500 | 4793 |
| EVALution | task1510_evalution_relation_extraction | 词汇关系抽取 | 6494 | 1000 | 200 | 500 | 4794 |
| PersonaChat | task1729_personachat_generate_next | 对话回复生成 | 6499 | 1000 | 200 | 500 | 4799 |
| Reddit TIFU | task511_reddit_tifu_long_text_summarization | 帖子摘要 | 6500 | 1000 | 200 | 500 | 4755 |
| SciQ | task591_sciq_answer_generation | 科学问答 | 6500 | 1000 | 200 | 500 | 4800 |
| GLUCOSE | task748_glucose_reverse_cause_event_detection | 因果事件关系生成 | 6497 | 1000 | 200 | 500 | 4797 |

合计 7000 条训练、1400 条验证、3500 条测试，共 11900 条。Reddit TIFU 另有 45 条空参考答案被排除，故该行“未使用”仅计有效余量。任务不是分类标签。训练多轮不增加独立样本数，未使用样本不能自动加入回放、额外预训练或调参。

五个任务在官方 default train_tasks 中，Reddit TIFU 和 GLUCOSE 在 excluded_tasks 中，涉及官方测试任务的同源隔离。它们可用于此自定义协议，但不直接追加官方测试任务并宣称来源独立。

## 4. 划分算法与核验

### 4.1 确定性样本选择

固定划分种子字符串 `20261006`，所有方法、顺序和训练 seed 共用。使用 SHA-256 排序，避免随机库版本改变抽样结果：

1. 校验源 JSON 的 SHA-256 和 Instances 数量，必须符合 `001_data_manifest.json`。
2. 先排除空输入或无有效参考答案的实例：input 必须是非空字符串，output 统一为列表后须非空且每项均为非空字符串（以 strip 判断）。排除 ID 必须与清单一致。对有效实例计算 `SHA256(UTF8("20261006\n" + task_id + "\n" + instance_id))`。
3. 按 `(哈希十六进制字符串, instance_id)` 升序排序。
4. 使用左闭右开切片：Train=`[0:1000]`，Dev=`[1000:1200]`，Test=`[1200:1700]`，剩余不使用。
5. 同一实例的全部参考答案保留在同一集合，不拆成不同实例。
6. 验证和测试保持该顺序；训练可按训练 seed 打乱，但不改变成员归属。

每个 split 的有序 ID 指纹定义：全部 ID 用 LF 连接，末尾再加一个 LF，UTF-8 编码后取 SHA-256。清单中记录七个源文件和 21 个有序 ID 指纹，复现时必须匹配。

### 4.2 数据边界检查

实例 ID 在每个任务内唯一。输入规范化定义为 Unicode NFKC 后按空白分词，以单个空格连接，不转小写、不删除标点。检查任务内重复输入，以及七任务合并后 Train/Dev/Test 两两输入交集。

当前固定快照已通过上述核验：各任务实例 ID 和规范化输入唯一，派生三个集合跨任务合并后的两两输入交集均为 0。此检查不宣称排除了所有近重复、同文档不同问题或基座预训练污染。

计数不足、指纹变化或边界失败则停止并报告，不静默丢弃、补采样或更换 seed。修改数据须发布新协议版本，让所有方法统一采用。

### 4.3 可直接运行的核验命令

在项目根目录执行，仅需 Python 3 标准库，只读取数据，不修改文件：

```bash
python3 - <<'PY'
import hashlib
import json
import unicodedata
from pathlib import Path

manifest = json.loads(Path('rules/001_data_manifest.json').read_text())
bounds = {'train': (0, 1000), 'dev': (1000, 1200), 'test': (1200, 1700)}
inputs = {split: set() for split in bounds}
totals = {split: 0 for split in bounds}

def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())

for task in manifest['tasks']:
    content = Path(task['source_path']).read_bytes()
    assert hashlib.sha256(content).hexdigest() == task['source_sha256']
    rows = json.loads(content)['Instances']
    assert len(rows) == task['total_instances']
    assert len({row['id'] for row in rows}) == len(rows)
    assert len({normalize(row['input']) for row in rows}) == len(rows)
    valid, rejected = [], []
    for row in rows:
        refs = row['output'] if isinstance(row['output'], list) else [row['output']]
        if not isinstance(row['input'], str) or not row['input'].strip() or not refs or any(not isinstance(value, str) or not value.strip() for value in refs):
            rejected.append(row['id'])
        else:
            valid.append(row)
    assert rejected == task['excluded_instance_ids']
    assert len(valid) == task['valid_instances']
    rows = valid
    def sort_key(row):
        value = manifest['split_seed'] + '\n' + task['task_id'] + '\n' + row['id']
        return hashlib.sha256(value.encode('utf-8')).hexdigest(), row['id']
    rows.sort(key=sort_key)
    for split, (start, end) in bounds.items():
        selected = rows[start:end]
        assert len(selected) == task['splits'][split]['count']
        ids = '\n'.join(row['id'] for row in selected) + '\n'
        assert hashlib.sha256(ids.encode('utf-8')).hexdigest() == task['splits'][split]['ordered_ids_sha256']
        inputs[split].update(normalize(row['input']) for row in selected)
        totals[split] += len(selected)
for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
    assert not inputs[left].intersection(inputs[right]), (left, right)
assert len(manifest['tasks']) == 7
assert totals == {'train': 7000, 'dev': 1400, 'test': 3500}
print('Verified: 7 tasks; train=7000, dev=1400, test=3500')
PY
```

正式执行器还应输出完整样本清单，包括实例 ID、原数组下标、split、内容哈希，随结果归档。以上命令重建并验证划分，不生成派生文件。

## 5. 任务顺序与训练种子建议

参考 SAPT 两个顺序，筛为七任务并保持相对顺序：

```text
order_1: XSUM → Quoref → EVALution → PersonaChat → GLUCOSE → Reddit TIFU → SciQ
order_2: GLUCOSE → SciQ → EVALution → PersonaChat → Reddit TIFU → Quoref → XSUM
training_seeds: [42, 43, 44]
split_seed: 20261006
```

每方法两个顺序×三个 seed，共六次持续学习。数据表显示顺序不代表训练顺序。设置 Python、NumPy、PyTorch CPU/CUDA、sampler 和 worker 种子，并记录确定性开关及环境。GPU 运算可能仍非确定，不仅凭 seed 声称位级复现。

## 6. 共同基座、方法与公平比较

实施状态更新：O-LoRA、MIGU-LoRA、SAPT-LoRA 的首轮均使用第 13 节的固定公共设置，包括 NF4/FP16、1 epoch 和同一 Llama-2-7B；各方法适配及额外预算详见 [exp/README.md](../exp/README.md)。本节和第 7 节中的早期建议不覆盖已锁定的首轮配置。运行汇报统一放在 `summary/`。

建议统一 `meta-llama/Llama-2-7b-hf` base 与同一 tokenizer，实际权重 revision/文件哈希待锁定。不得静默替换为 Chat、其他模型或参数规模。

当前机器为 4 张 RTX 2080 Ti，各约 11 GiB。量化、计算精度、冻结范围、分布式方案需要方法兼容性和显存核验；NF4 4-bit / FP16 仍为候选，不是已确定设置。不同精度结果分组比较。

| 方法 | 核心机制与应记录内容 |
|---|---|
| O-LoRA | 逐任务低秩空间、历史空间冻结和正交约束；目标层、rank、损失系数 |
| Progressive Prompts | 历史提示冻结、渐进拼接；decoder-only 适配、提示长度和重参数化 |
| SAPT-LoRA | 共享注意力和抗遗忘模块；伪样本、注意力目标、记忆及回放来源 |
| UNLOCK/MIGU | 激活幅度控制梯度更新；LoRA 接入、门限与选择粒度 |
| MAC | 摊销上下文、记忆和聚合网络；额外模型及生成任务适配 |
| SeqLoRA（建议对照） | 顺序更新同一 LoRA；公共 LoRA 设置 |

MixLoRA 暂不实施。MAC 从文档流问答迁移到本生成实验的方式尚待确定；若不能保留机制则报告不支持，不以普通检索或 LoRA 代替。

各方法共享基座、数据、模板、顺序、seed、长度、解码和计分器；保留必要方法差异并披露。报告可训练参数、累计新增参数、额外模型、提示长度、记忆大小、回放/生成样本量、更新步数、耗时、GPU 数与峰值显存。相同基座不等于相同总算力。

## 7. 输入与训练默认建议

### 7.1 统一模板

读取 `Definition[0].strip()` 和 `instance.input.strip()`，使用 LF 换行：

```text
Instruction: {definition}

Input: {input}

Response:
```

不加入 Positive/Negative Examples 或其他带答案的少样本示例。output 为字符串时包装为列表；训练目标取 `output[0].strip()`，评价保留全部参考。空定义、空目标、非法结构报错，不静默跳过。

建议 prompt 最大 768 tokens、目标/生成最大 256 tokens，超限右截断并记录比例。手动添加一次 BOS，目标预留 EOS。PAD 复用 EOS 时用 attention mask 区分，prompt/padding 标签为 -100，损失只计算目标。额外软提示计入上下文占用并检查模型上限。预测只解码新 tokens。

### 7.2 训练预算

以下为待确认的具体建议，实施时必须显式写入配置：

| 设置 | 建议 |
|---|---|
| 每任务训练 | 20 epochs，不提前结束 |
| 有效 batch | 全部 GPU 合计每更新 16 条当前任务实例 |
| micro-batch/累积 | 显存核验后固定，满足上述有效 batch |
| 尾批 | 不丢弃、不重复补齐，按实际实例数归一化 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 梯度裁剪 | 全局 norm 1.0 |
| 调度 | 每任务重置 optimizer/scheduler；warmup 占总更新步数 3%（向上取整），随后线性衰减 |
| 选模 | 每轮评估当前 Dev，选 ROUGE-L 最佳检查点；并列选更早轮次 |
| 下一任务 | 从所选权重及配套方法状态继续 |
| 学习率及方法参数 | 各方法分别锁定，尚未确定 |

回放、伪样本生成和辅助网络训练单独计账。方法需不同优化器或状态机制时记录依据，不隐含改变预算。

建议主实验事先固定超参数。确需搜索时，提前规定各方法相同候选配置数量，在独立开发运行中仅用 Dev 选参，记录搜索空间、成本和选择规则；不得看 Test 调参。使用未来任务 Dev 的离线开发需披露，不能称为完全未知未来任务的在线设定。调参预算尚待锁定。

## 8. 数据可见性与阶段流程

Train 用于更新；Dev 用于选模及合法开发；Test 仅用于报告。正式阶段只用当前任务 Train/Dev 和方法从已学 Train 保留的合法记忆。未来任务 Train/Dev 不参与该阶段更新或选模；任何 Dev/Test 不用于回放或路由拟合。

建议任务已知评估：可提供 task_id，记录方法是否使用。未来任务尚无已训练专属模块，必须预先规定如何用已有模块推理，禁止提前训练未来模块。不能合法推理时显示 N/A 和原因，不填零，相应零样本迁移不能完整计算。

测试采用 eval/no_grad，不持久更新模型、记忆或路由状态。当前输入驱动的正常推理路由允许，测试分数不反馈训练或模型选择。

执行步骤：

1. 核验代码、配置、数据、模型指纹，建立 method/order/seed 独立运行。
2. 基座在七个 Test 上初测，保存 b[j] 和预测；完全相同配置可按指纹复用。
3. 训练当前任务，按当前 Dev 选择检查点。
4. 所选状态评价全部七个 Test，包括未来任务，保存预测和矩阵行。
5. 依序重复至七任务完成；下一任务不重置到初始基座。
6. 从完整矩阵直接计算 AP、F.Rate、FWT、BWT；不启动独立单任务训练。

每次顺序运行产生 8×7 矩阵：行 0 为基座，行 1–7 为阶段；列按该运行顺序排列。每格 500 条预测，逻辑上共 28000 条。初测缓存不改变矩阵定义。

每个方法/顺序/seed 只有一条持续学习主线，保留上一阶段学到的方法状态。FWT 在任务 j 首次训练前取 R[j−1,j]，不会为计算指标提前训练该任务。

恢复中断需要模型、优化器、调度器、RNG、sampler、阶段/epoch、记忆和路由状态；只恢复权重不称为无缝恢复。重复尝试使用不同 run_id。

## 9. 指标与评分口径

### 9.1 任务分数

建议 greedy 解码：`do_sample=false`、`num_beams=1`、`max_new_tokens=256`，遇 EOS 停止，不设采样温度。真实生成后评分，不用 teacher-forced loss 代替成绩。

主分数为 ROUGE-L F1（0–100）。建议 `rouge-score==0.1.2`、`use_stemmer=True`、英文默认分词，不使用 rougeLsum。每实例对所有参考分别评分取最大，任务内平均后乘 100。锁定依赖版本，中间得分不舍入。

Quoref、SciQ 另报 SQuAD 风格 EM/token-F1：小写、去 ASCII 标点、去英文冠词 a/an/the、合并空白。F1 按 token 多重集合交集计算，双方空记 1，仅一方空记 0，多参考取最大。附加指标分列，不与 ROUGE-L 混合求 AP。开放对话的文本重合不足以完整表示回复合理性，保留预测供检查。

### 9.2 持续学习指标

T=7；R[i,j] 为学完第 i 个任务后在第 j 个 Test 的 ROUGE-L，R[0,j]=b[j]。j 表示当前运行顺序位置。

$$
\begin{aligned}
\mathrm{AP} &= \frac{1}{T}\sum_{j=1}^{T}R_{T,j},\\
\mathrm{F.Rate} &= \frac{1}{T-1}\sum_{j=1}^{T-1}\left(\max_{j\le i\le T-1}R_{i,j}-R_{T,j}\right),\\
\mathrm{BWT} &= \frac{1}{T-1}\sum_{j=1}^{T-1}\left(R_{T,j}-R_{j,j}\right),\\
\mathrm{FWT}_{\mathrm{GEM}} &= \frac{1}{T-1}\sum_{j=2}^{T}\left(R_{j-1,j}-R_{0,j}\right).
\end{aligned}
$$

AP/FWT/BWT 越大越好，F.Rate 越小越好。差值单位为百分点；F.Rate 不除以历史最大，不截断负数。FWT 采用 GEM（Lopez-Paz & Ranzato, 2017）§2 式 (4) 的训练前迁移口径。原文初始模型为随机初始化，本项目为固定预训练 Llama-2-7B；原文任务分数为准确率，本项目统一为 ROUGE-L。第一任务没有先前学习，排除于 FWT 平均。

缺少必要矩阵项则对应指标 N/A，不填零、不改分母。评价协议标识为 `cl_standard_fwt_v1`，数据协议保持 `superni_generation7_v2`。SAPT §5.1.2 的 FWT 比较训练后分数和独立单任务训练分数，属于不同定义，本项目不采用。旧尝试保留原始记录，不与新口径混合汇总。

依据：[GEM §2 式 (4)](https://arxiv.org/abs/1706.08840)；Progressive Prompts 附录引用该工作的 FWT/BWT；[SAPT §5.1.2](https://aclanthology.org/2024.acl-long.625/) 使用不同参照。并非所有论文都报告同一套指标；这里统一四项指标以便公平比较。

## 10. 汇总与统计

先计算每次完整运行的指标；每个 seed 先平均两个顺序，再对三个 seed 的三个均值报告均值和样本标准差（ddof=1）。同时保留两个顺序各自的三种子统计及六个原始值。F.Rate 含 max，不能先平均矩阵再算。

各任务在 AP 中同权。三种子标准差表示训练随机性，不是测试抽样置信区间。模型、精度、数据、任务集合、输入/解码/计分设置或预算不同的运行分表。

报告资源及状态，失败/缺失/不适用分别标记；重复 method/order/seed 明确选定 run_id，不自动挑最高 Test、不重复入账。六次未齐只能显示进度，不能标为完整主表。

## 11. 输出与复现记录

计划输出结构，当前尚未生成：

```text
exp/result/<protocol>/<method>/<order>/seed_<seed>/<run_id>/
  config.json
  provenance.json
  split_manifest.json
  status.json
  score_matrix.json
  metrics.json
  resources.json
  predictions/stage_00/<task_id>.jsonl
  predictions/stage_01/<task_id>.jsonl
  ...
  checkpoints/
```

每条预测包含协议、方法、顺序、seed、阶段、task_id、instance_id、prediction、references。矩阵记录行阶段和列 task_id，指标记录 evaluation_protocol。

provenance 至少记录 Git commit、dirty diff 哈希、协议/配置哈希、源和派生数据哈希、模型/tokenizer revision、Python/PyTorch/Transformers/PEFT/Accelerate/bitsandbytes/CUDA/驱动版本、完整环境锁文件、GPU、命令、时间、恢复来源和确定性设置。当前不预填未经验证的依赖组合。

后续 `summary/` 脚本读取真实结果，生成 CSV 和 Markdown 表。原始大文件保持 Git 忽略，协议、配置、ID 清单及元信息可入库。认证令牌不写入日志或配置。

## 12. 实施前剩余事项

任务名单和 1000/200/500 配额已确认，无需重问；数据算法、快照及指纹可按第 4 节复现。

尚需一次性落实：模型权重及可用性，量化/冻结/精度/分布式，两个顺序与三个 seed，各方法参数和 MAC 适配，20 epochs 等预算及调参安排，任务已知和未来任务推理规则，兼容依赖环境。

最初的实施前确认要求已由后续“push，然后启动，先跑一个论文”授权推进首轮。完整多顺序多种子研究方案仍有待定部分；本次先执行第 13 节，不声称已完成整套研究。

## 13. 已授权的 O-LoRA 首轮执行配置

用户随后明确要求“push，然后启动，先跑一个论文的结果”。据此开始实现和启动首跑，不再等待整套六次重复实验配置的再次确认。

- 方法：语言模型 O-LoRA，正确上游为 cmnfriend/O-LoRA，commit `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`。原 `source_code/O-LoRA-src` 实为视觉 Online-LoRA，保留原归档；新增正确 `source_code/O-LoRA-language-src`。已核验该上游快照的 426 个文件。上游生成日志仅本地保留、Git 忽略。
- 实现：`code/olora.py`。q_proj/v_proj 注入 rank=8、alpha=32、dropout=0.1 的低秩适配器。每任务新增一组，冻结历史组，推理时累加所有已学组。正交损失为历史 A 与当前 A 转置乘积的绝对值之和，系数 0.5；L2 系数 0。冻结 NF4 基座，adapter 参数 FP32，前向主要 FP16。
- 配置：`exp/configs/olora_first.json`。order_1、seed 42、每任务 **1 epoch**、固定学习率峰值 1e-4、有效 batch 16、micro-batch 1、eval batch 2。其余输入长度、warmup、优化器、裁剪和生成设置沿用本文建议。1 epoch 参考原论文训练轮数，但固定学习率和生成任务等属于统一适配；这不是原始论文数值复现，也不是前文建议的 20-epoch 实验。
- 当前答案损失按每实例有效目标 token 平均，再按实例平均；正交项随 micro-batch 权重一起累积，避免梯度累积次数隐式放大约束。
- 首轮包含基座初测和七阶段持续学习，最终从同一矩阵计算四项指标。它只代表一个顺序、一个 seed，不标为两顺序三种子的最终统计。
- 首轮使用一张 GPU 执行顺序学习，默认 GPU 0；若传入多个 GPU ID，仅使用第一个。其余卡不启动单任务对照，有效 batch 保持 16。
- 数据 v2：实际读取时发现 Reddit TIFU 45 条空答案，列入清单后在排序前排除。保持用户指定 1000/200/500，更新 Reddit TIFU 指纹；其余任务指纹不变。v1 仅核查输入边界，未训练，不能与 v2 结果混用。
- 模型权重需提供有效本地 Llama-2-7B 路径或已授权 Hugging Face 登录。启动器锁定下载 commit，或对本地权重逐文件计算 SHA-256。模型未就绪记录 blocked_model，不能声称已训练。

启动：`bash exp/run_olora.sh --model /absolute/path/to/Llama-2-7b-hf`；使用已授权 Hugging Face 自动下载时省略 `--model`。可加 `--gpus 0,1,2,3`。仅数据准备：`bash exp/run_olora.sh --prepare-only`。汇总：`.conda-env/bin/python summary/collect.py`。

环境使用项目独立 Conda 前缀 `.conda-env/`，Python 3.10.21、PyTorch 2.6.0+cu118；不继承现有 MM 环境。重建入口为 `bash exp/setup_env.sh`，Conda 定义见 `exp/environment.yml`，主要依赖见 `exp/requirements.txt`，完整版本锁见 `exp/requirements.lock`，执行时另保存 pip freeze。PyTorch 通过官方 cu118 索引安装，其余默认使用清华镜像，可用 LLMCL_PIP_INDEX_URL 覆盖。启动脚本禁用用户 site-packages 并清除 PYTHONPATH，避免外部包混入。早期功能验证用过共享系统包的 `.venv`，正式启动入口已切换，不使用该旧环境。

当前支持每个任务结束的最佳 adapter 检查点与全套状态日志，不支持优化器级断点续训；中断需新建 run_id 重跑，不能宣称无缝恢复。功能小模型运行必须标记 smoke=true，汇总器排除其分数。

验证记录：`cl_standard_fwt_v1` 已通过本地小 Llama 的 CPU 单主线七阶段完整启动/汇总检查，得到 8×7 矩阵和四项指标，确认无单任务目录；另核对正/负 FWT 手算、缺失矩阵拒绝、未完成指标留空和 smoke 排除。历史版本还通过 RTX 2080 Ti 上的 NF4/FP16 七阶段功能运行。AMP 初始 loss scale 为 1024，非有限 loss/gradient 立即报告失败。小模型功能测试不能作为正式论文结果。
