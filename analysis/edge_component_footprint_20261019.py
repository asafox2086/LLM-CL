"""Read real deployed tensor headers; bound decoder-only compression savings."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/component_footprint/v1'
MODELS={'onevision':'LLaVA-OneVision-Qwen2-0.5B-OV','qwen':'Qwen2-VL-2B-Instruct'}
SIZES={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


def role(backbone,name):
    if backbone=='onevision':
        if name.startswith('vision_tower.'):return 'vision_encoder'
        if name.startswith('multi_modal_projector.'):return 'projector'
        if name.startswith('language_model.'):return 'language'
    else:
        if name.startswith('visual.merger.'):return 'projector'
        if name.startswith('visual.'):return 'vision_encoder'
        if name.startswith(('model.','lm_head.')):return 'language'
    return 'other'


def main():
    REPORT.mkdir(parents=True,exist_ok=True);result={}
    for backbone,folder in MODELS.items():
        model=ROOT/'model'/folder;files=sorted(model.glob('*.safetensors'));assert files
        groups={};records=[];seen=set()
        for p in files:
            with p.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0];assert 0<length<100*1024**2
                raw=f.read(length);header=json.loads(raw)
            intervals=[]
            for name,value in header.items():
                if name=='__metadata__':continue
                assert name not in seen;seen.add(name)
                n=math.prod(value['shape']);width=SIZES[value['dtype']]
                begin,end=value['data_offsets'];assert end-begin==n*width
                intervals.append((begin,end));group=groups.setdefault(role(backbone,name),dict(parameters=0,serialized_tensor_bytes=0,tensors=0,dtypes={}))
                group['parameters']+=n;group['serialized_tensor_bytes']+=end-begin;group['tensors']+=1
                group['dtypes'][value['dtype']]=group['dtypes'].get(value['dtype'],0)+n
            intervals.sort();assert intervals[0][0]==0 and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
            assert 8+length+intervals[-1][1]==p.stat().st_size
            records.append(dict(file=str(p.relative_to(ROOT)),header_sha256=sha_bytes(raw),file_bytes=p.stat().st_size,
                header_bytes=length,tensor_data_bytes=intervals[-1][1],full_weight_values_not_read=True))
        total=sum(x['serialized_tensor_bytes'] for x in groups.values());parameters=sum(x['parameters'] for x in groups.values())
        fixed=total-groups['language']['serialized_tensor_bytes']
        config_bytes=(model/'config.json').read_bytes();config=json.loads(config_bytes)
        for g in groups.values():g['fraction_of_tensor_bytes']=g['serialized_tensor_bytes']/total
        result[backbone]=dict(model=folder,total_parameters=parameters,total_tensor_bytes=total,groups=groups,
            decoder_only_hypothetical_weight_compression=dict(
                language_16_to_8bit_total_tensor_ratio=(fixed+groups['language']['serialized_tensor_bytes']/2)/total,
                language_16_to_4bit_total_tensor_ratio=(fixed+groups['language']['serialized_tensor_bytes']/4)/total,
                uncompressed_non_language_fraction=fixed/total,
                not_actual_export_or_runtime=True,scales_padding_runtime_not_included=True),
            config_sha256=sha_bytes(config_bytes),headers=records,
            selected_vision_feature_layer=config.get('vision_feature_layer'),vision_config=config.get('vision_config'))
    out=dict(complete=True,models=result,source_sha256=sha_bytes(Path(__file__).read_bytes()),
        model_forwards=0,GPU_started=False,TEST_QA=0,
        scope='Actual base tensor shape/dtype footprint; arithmetic bounds, not trained quality or speed')
    (REPORT/'COMPLETE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# 小语言基座不等于小视觉编码器','',
        '直接读取两骨干现有权重文件的实际tensor header；核对shape×dtype字节、偏移连续与实际文件大小。仅核对结构／存储，不读取训练或测试答案，不运行模型。未把header SHA说成全权重SHA。','',
        '| 骨干 | 实际总参数 | 视觉编码器参数 | 视觉tensor字节占比 | 只将语言权重16→8bit的理想总字节减少 |',
        '|---|---:|---:|---:|---:|']
    for name,v in result.items():
        g=v['groups']['vision_encoder'];reduction=1-v['decoder_only_hypothetical_weight_compression']['language_16_to_8bit_total_tensor_ratio']
        lines.append(f"| {v['model']} | {v['total_parameters']:,} | {g['parameters']:,} | {100*g['fraction_of_tensor_bytes']:.2f}% | {100*reduction:.2f}% |")
    lines+=['','结论：对小语言基座，只压缩语言部分无法同比缩小完整图文模型；视觉与投影形成固定负担。它是结构／字节记账，不是新理论、质量因果或实测速率。理想比例未计量化scale、对齐和运行时，不填主表。','',
        'OneVision实际选最后一层（vision_feature_layer=-1），视觉配置vision_use_head=false；不能照搬CLIP常见倒数第二层假设删除最后一层，也没有可直接删除的池化头。其HuggingFace输入路径虽收集全部层hidden_states，但输出保留优化仍需独立数值／资源验证，不声称已经完成。']
    (REPORT/'RESEARCH_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(parameters=v['total_parameters'],groups=v['groups']) for k,v in result.items()}))


if __name__=='__main__':main()
