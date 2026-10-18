"""Idle-GPU selection before importing the independent feedback experiment."""
import argparse,json,os,traceback
import onevision_research_device_entry as device


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--phase',choices=['build','full'],required=True)
    p.add_argument('--view',choices=['cache_AnswerF1','cache_AnswerF1_metric']);a=p.parse_args()
    assert device.select(device.inventory(),os.getuid())==a.gpu
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import onevision_answer_compatibility_trial as worker
    try:worker.main(False,a.phase,a.view)
    except Exception:
        worker.REPORT.mkdir(parents=True,exist_ok=True)
        (worker.REPORT/'needs_attention.json').write_text(json.dumps(dict(
            traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
