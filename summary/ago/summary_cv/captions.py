"""Chinese figure captions shared by generated and hand-written reports."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = '蓝色实线圆点＝SeqLoRA；绿色虚线菱形＝MIGU-LoRA；红色点划线三角＝O-LoRA；紫色点线方块＝SAPT-LoRA。'
CV_STAGES = '阶段 0 是未训练基座；1–5 依次学习摘要、阅读理解、关系抽取、对话、因果推理；6–7 学习医学对话记录和影像报告；8–9 学习自然与医学图像问答。'
T5_STAGES = '阶段 0 是未训练基座；1–7 依次学习 XSum 摘要、Quoref 阅读理解、关系抽取、PersonaChat 对话、GLUCOSE 因果推理、Reddit 摘要和 SciQ 科学问答。'
HEATMAP_TASKS = 'XSum＝摘要，Quoref＝阅读理解，Relation＝关系抽取，Dialogue＝对话，Cause＝因果推理，MTS＝医学对话转记录，IU-Xray＝影像报告，VQAv2＝自然图像问答，VQA-RAD＝医学图像问答；T5 的 Reddit＝帖子摘要、SciQ＝科学问答。'

def dashboard(model, stages):
    return (model+'：整体学习与保留过程', [
        '**坐标与阶段：** 横轴是已完成学习的任务数。'+stages,
        '**四个子图：** 左上在固定全部任务上算平均 ROUGE-L（0–100），看整体水平；右上只平均已学任务，看已学表现，但任务集合会增加。左下是本阶段更新前后，同一组旧任务的平均分差，负值说明旧任务受损；右下是尚未学习任务相对基座的平均分差，正值说明前序学习可能有帮助。下面两个子图的单位是分数点，0 线表示无变化；没有可比较任务时留空。',
        '**方法与读法：** '+METHODS+'先看左上是否整体提高，再看左下在哪一步出现下降。右上和右下的任务构成随阶段变化，曲线起伏也可能来自构成变化，需结合逐任务曲线。ROUGE-L 衡量参考文字重合，不是事实正确率。'])

def heatmap(model, stages):
    tasks = HEATMAP_TASKS.split('；T5')[0]+'。' if model.startswith('Qwen') else 'XSum＝摘要，Quoref＝阅读理解，Relation＝关系抽取，Dialogue＝对话，Cause＝因果推理，Reddit＝帖子摘要，SciQ＝科学问答。'
    return (model+'：每个任务相对基座的变化', [
        '**坐标与每个格子：** 四个子图分别对应四方法；横轴是被评测任务，纵轴是学习阶段。'+stages+'格子中的数值＝该阶段任务得分−同一任务基座得分，单位为 ROUGE-L 分数点。',
        '**颜色与黑框：** 蓝色表示高于基座，红色表示低于基座，接近白色表示变化较小。色条给出数值范围，四方法共用同一范围；黑框标出该任务刚学完的格子。第 0 行全为 0，因为基座与自身比较。',
        '**任务缩写与读法：** '+tasks+'沿同一列向下看一个任务如何变化；黑框以后颜色变浅或数值变小，说明刚学到的表现发生回落。该图比较基座，不能直接当作相邻阶段的学习收益。'])

CAPTIONS = {
    'cv_trajectories.png': dashboard('Qwen2-VL 九任务', CV_STAGES),
    't5_trajectories.png': dashboard('T5 七任务', T5_STAGES),
    'cv_delta_baseline.png': heatmap('Qwen2-VL 九任务', CV_STAGES),
    't5_delta_baseline.png': heatmap('T5 七任务', T5_STAGES),
    'cv_domain_trajectories.png': ('四个领域的表现如何随学习变化', [
        '**坐标与子图：** 横轴是已完成任务数，纵轴是固定领域内任务的平均 ROUGE-L（0–100）。左上＝5 个通用语言任务，右上＝2 个医学语言任务，左下＝自然图像问答，右下＝医学图像问答；每个领域在所有阶段使用同一任务集合。',
        '**方法与分界线：** '+METHODS+'竖直灰色虚线位于阶段 5/6、7/8、8/9 之间，分别表示开始医学语言、自然图像和医学图像学习。'+CV_STAGES,
        '**怎样读：** 看某领域在开始训练后是否提高，以及后续训练其他领域时是否回落。例如蓝线在医学语言学习后提高、随后略降。各子图纵轴自动缩放、起点不同，不能用线条高度或视觉斜率跨子图比较变化大小；应比较坐标数值。']),
    'cv_delta_transition.png': ('每次任务切换带来了什么变化', [
        '**坐标与每个格子：** 四个子图分别对应四方法；横轴是被评测任务，纵轴是本次学习阶段（1–9）。数值＝本阶段得分−上一阶段同一任务得分，单位为 ROUGE-L 分数点。'+HEATMAP_TASKS,
        '**颜色与黑框：** 蓝色＝本次更新后提高，红色＝本次更新后下降，接近白色＝变化较小；四方法共用同一色条范围。黑框是本阶段刚训练的任务，通常反映直接学习收益；它左侧是旧任务、右侧是未来任务。',
        '**怎样读：** 沿一行看一次训练同时怎样影响新、旧、未来任务。例如阶段 7 的 MIGU-LoRA 行可定位医学报告训练时旧能力的下降。这里比较相邻阶段，不是相对基座；很大的黑框收益也可能包含恢复之前丢失的能力。']),
    'cv_task_trajectories.png': ('同一任务从基座到最终的完整轨迹', [
        '**坐标与九个子图：** 每个小图是一个固定任务，横轴是已完成任务数（0–9），纵轴是该任务 ROUGE-L（0–100）；所有小图共用 0–100 纵轴范围。'+HEATMAP_TASKS,
        '**方法与学习时点：** '+METHODS+'每个小图中的竖直灰色虚线标出该任务刚学完的阶段；虚线前是尚未直接学习，虚线后是后续任务的影响。',
        '**怎样读：** 比较虚线前后相邻点，得到直接学习收益；比较阶段 0 与虚线处，判断是否超越基座；再看虚线处到阶段 9，判断保留还是回落。例如关系抽取的跃升很明显，医学对话记录没有同样的跃升，不能仅凭最终排名认为所有任务都学得好。']),
    'cv_external_knowledge.png': ('训练之外的通用与医学知识题是否受到影响', [
        '**坐标与三个子图：** 横轴是已完成任务数（0–9），纵轴是固定选择题的准确率（%，0–100）。左图＝102 道通用学科题，中图＝96 道医学相关题，右图＝57 学科准确率的等权宏平均，避免医学题多导致权重过大。',
        '**方法与阶段：** '+METHODS+CV_STAGES+'所有点使用完全相同的 198 道题及提示，只读取各阶段 checkpoint，不再训练。',
        '**怎样读：** 每条线与自己的阶段 0 比较，观察学习新任务后，外部知识题是否下降；例如 MIGU-LoRA 通用组从 53.92% 到 50.98%，医学组从 56.25% 到 58.33%。这是零样本 A/B/C/D 候选评分的小样本探针，非完整 MMLU，也不能代表全部知识；图中没有多种子误差条。']),
    'natural_vqav2_COCO_val2014_000000388829.jpg': ('自然图像问答的真实测试输入', ['本图用于问题 “Is it cold outside?”（外面冷吗？）。参考答案来自十位标注者，包含 yes 与 no，因此不是单一一致标签。模型回答及评分需与该节参考一起阅读；这张输入图片本身不表示模型已经答对。']),
    'natural_vqav2_COCO_val2014_000000343606.jpg': ('自然图像问答：数量问题的测试输入', ['本图用于问题 “How many white birds?”（有几只白鸟？）。十位标注者给出的参考包含 0 和 1；报告按多参考匹配规则评分，不能把某一个标注直接当成唯一答案。']),
    'natural_vqav2_COCO_val2014_000000356949.jpg': ('自然图像问答：描述问题的测试输入', ['本图用于问题 “Is the zebra mane spiky or soft?”（斑马鬃毛是尖硬的还是柔软的？）。参考标注包含 spiky 和 soft；图片、提问与模型输出共同构成此测试例。']),
    'medical_vqa_rad_synpic29265.jpg': ('医学图像问答的真实测试输入', ['这是 VQA-RAD 测试图片 synpic29265。同一图片用于不同问题，例如肺部外观是否正常、成像方向是什么；应以当前章节列出的具体问题和参考答案为准。图像未被修改，模型文字回答及对应分数列在后文。']),
}

START = '<!-- figure-caption:start -->'
END = '<!-- figure-caption:end -->'

def add_captions(text):
    # Re-rendering a report must refresh captions without duplicating them.
    text = re.sub(r'\n\n'+re.escape(START)+r'.*?'+re.escape(END), '', text, flags=re.S)
    index = 0
    def attach(match):
        nonlocal index
        filename = Path(match.group(1)).name
        if filename not in CAPTIONS:
            raise ValueError('Missing caption for '+filename)
        index += 1
        title, paragraphs = CAPTIONS[filename]
        body = '\n>\n'.join('> '+p for p in paragraphs)
        return match.group(0)+'\n\n'+START+f'\n**图 {index}｜{title}。**\n\n'+body+'\n'+END
    return re.sub(r'^!\[[^\]]*\]\(([^)]+)\)$', attach, text, flags=re.M)

if __name__ == '__main__':
    paths = [ROOT/'README.md', ROOT/'CV_data/README.md', *sorted((ROOT/'summary').glob('*.md')),
             *sorted((ROOT/'summary_cv').glob('*.md')), ROOT/'sample/README.md']
    count = 0
    for path in paths:
        original = path.read_text()
        updated = add_captions(original)
        if updated != original:
            path.write_text(updated)
        count += updated.count(START)
    print(f'Added or refreshed {count} figure captions across {len(paths)} reports.')
