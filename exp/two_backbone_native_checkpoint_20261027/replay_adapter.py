"""Standard ER + LoRA control; reservoir capacity5000, seed42, train-only.

ER is a standard baseline also evaluated by CL-MoE, not a new CVPR method.
Memory capacity follows CL-MoE section5; LoRA and a 1:1 old/current loss
are explicit adaptations to the shared lightweight backbone protocol.
"""
import hashlib,random
from methods import Method
from native_auxiliary import NativeIO


class ERMethod(Method):
    def __init__(self,model,name,instructions,embedding=None):
        super().__init__(model,'SeqLoRA',instructions,embedding)
        self.name=name;self.memory=[];self.capacity=5000;self.stage=0;self.replay_seen=0

    def bind_context(self,processor,manifest,data,run,preflight=False):
        self.io=NativeIO(self,processor,manifest,data,run)

    def begin_stage(self,stage):
        self.stage=stage
        # Deterministic bottom-hash reservoir of all PAST native train IDs.
        history=[(t,row) for t in range(1,stage) for row in self.io.data[t,'train']]
        self.memory=sorted(history,key=lambda item:hashlib.sha256(('42:ER:buffer:'+item[1]['id']).encode()).digest())[:self.capacity]
        self.io.record(f'er_stage_{stage:02d}_memory.json',{'capacity':self.capacity,'train_only':True,
                       'past_train_available':len(history),'ids':[row['id'] for _,row in self.memory],
                       'selection':'seed42 deterministic hash reservoir','old_to_current_loss_ratio':'1:1'})

    def training_loss(self,batch,row,processor):
        from runtime_utils import answer_loss,input_batch
        from frozen_vision import prepare
        loss,length=answer_loss(self.model,batch,row['answer'],processor)
        if not self.memory:return loss,length
        index=int.from_bytes(hashlib.sha256(f"42:ER:draw:{self.stage}:{row['id']}".encode()).digest()[:8],'big')%len(self.memory)
        task,old=self.memory[index]
        previous,_=input_batch(old,self.io.manifest['tasks'][task-1],processor,self.model.device)
        prepare(previous,old,self.model,self.vision_original)
        replay_loss,replay_length=answer_loss(self.model,previous,old['answer'],processor)
        self.replay_seen+=1
        return .5*(loss+replay_loss),length+replay_length

    def method_state(self,stage):
        return {'method':self.name,'completed_stage':stage,'buffer_capacity':self.capacity,'memory_count':len(self.memory),
                'replay_seen':self.replay_seen,'source':'standard ER, also CL-MoE CVPR2025 section5',
                'adaptation':'shared q/v LoRA rank8 alpha32; 1:1 current/replay; deterministic hash reservoir'}
