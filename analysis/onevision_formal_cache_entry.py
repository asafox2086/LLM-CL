"""Set admitted physical GPU before importing the formal worker or torch."""
import argparse
import os
import onevision_research_device_entry as device


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--phase',choices=['build','full'],required=True)
    p.add_argument('--view',choices=['cache_geometry','cache_quality'])
    a=p.parse_args()
    assert device.select(device.inventory(),os.getuid())==a.gpu, 'Actual idle admission changed; no model load'
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import onevision_formal_state_cache as worker
    worker.main(False,a.phase,a.view)


if __name__=='__main__':main()
