"""Independent fresh retry after reboot: unchanged PAS math, durable optimizer evidence."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]
V1=ROOT/'exp/pas_fit_reference_20261022/v1'
spec=importlib.util.spec_from_file_location('pas_reference_frozen_v1_core',V1/'core.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
PORT,PILOT,study=legacy.PORT,legacy.PILOT,legacy.study
sha,write,digest=legacy.sha,legacy.write,legacy.digest
gradient_directory,save_first_raw_gradients=legacy.gradient_directory,legacy.save_first_raw_gradients
make_method,observed_optimizer,learner_digest=legacy.make_method,legacy.observed_optimizer,legacy.learner_digest
REPORT=ROOT/'summary/edge_deployment_20261019/pas_fit_reference_20261022/v2'
WORK=Path(__file__).parent/'private'

class DurableRecords(list):
    def __init__(self,path):
        super().__init__();self.path=Path(path);self.commits=0
        assert not self.path.exists()
    def append(self,value):
        super().append(value)
        write(self.path,list(self))
        directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        self.commits+=1

def cpu_gate():
    """Use the already-reviewed tiny comparison through this new persistence path once."""
    original=legacy.observed_optimizer
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'absent'/'OPTIMIZER_EVIDENCE.json'
        durable=DurableRecords(path)
        def wrapped(method,factory,records):
            class Mirror:
                def append(self,value):
                    durable.append(value)
                    assert json.loads(path.read_text())==list(durable)
                    records.append(value)
            return original(method,factory,Mirror())
        legacy.observed_optimizer=wrapped
        try:gate=legacy.cpu_gate()
        finally:legacy.observed_optimizer=original
        assert durable.commits==len(durable)==4
        assert json.loads(path.read_text())==gate['optimizer_evidence']==list(durable)
        gate.update(durable_optimizer_witness_each_of_four_steps_strict_reload=True,durable_commit_count=durable.commits,
            repair_only='Persist original optimizer observations immediately; no change to PAS math, parameter updates or RNG')
        return gate
