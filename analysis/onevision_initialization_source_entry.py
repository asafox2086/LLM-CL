"""Select idle research GPU before importing torch-dependent native controls."""
import argparse
import os
import onevision_research_device_entry as device


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--phase',choices=['sources','case'],required=True);p.add_argument('--case-index',type=int)
    a=p.parse_args();assert device.select(device.inventory(),os.getuid())==a.gpu
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import onevision_initialization_source_controls as worker
    try:worker.main(False,a.phase,a.case_index)
    except Exception:
        import json,traceback
        worker.REPORT.mkdir(parents=True,exist_ok=True)
        (worker.REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
