"""Actual PEFT/trainer/export/resume checks using a tiny CPU causal model."""
from dataclasses import dataclass
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig, TrainingArguments, set_seed
from transformers.modeling_outputs import CausalLMOutput
from peft.utils import set_peft_model_state_dict

from llava.model.lorasculpt_plus import configure_lorasculpt_plus_adapters
from llava.model.orbit_stiefel import install, exported_state, layers
from llava.train.OrbitStiefel_Trainer import OrbitStiefelLoRASculptPlus


class Tiny(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config=None):
        super().__init__(config or PretrainedConfig())
        self.embed = nn.Embedding(20, 16)
        self.q_proj = nn.Linear(16,16)
        self.mm_projector = nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,16))
        self.lm_head = nn.Linear(16,20)

    def prepare_inputs_for_generation(self,input_ids,**kwargs):
        return dict(input_ids=input_ids)

    def forward(self,input_ids=None,labels=None,**kwargs):
        h=self.q_proj(self.embed(input_ids));h=h+self.mm_projector(h)
        logits=self.lm_head(h)
        loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten()) if labels is not None else None
        return CausalLMOutput(loss=loss,logits=logits)


class Data(Dataset):
    def __len__(self):return 16
    def __getitem__(self,i):
        ids=torch.tensor([(i+j)%20 for j in range(4)])
        return dict(input_ids=ids,labels=ids)


@dataclass
class Args(TrainingArguments):
    mm_projector_lr: float=.001
    group_by_modality_length: bool=False
    tune_mm_mlp_adapter: bool=False


def make():
    set_seed(123)
    model,metadata=configure_lorasculpt_plus_adapters(Tiny(),llm_rank=4,llm_alpha=4,connector_rank=4,connector_alpha=4,dropout=0,bias='none')
    install(model,dict(warmup_steps=2,keep_ratio=.5,orbit_iterations=3,protection_rank=3,seed=123))
    model.orbit_layout_metadata=metadata
    return model,metadata


def run():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        model,meta=make()
        args=Args(output_dir=tmp+'/full',use_cpu=True,max_steps=4,per_device_train_batch_size=2,
                  learning_rate=.001,weight_decay=0,save_steps=2,logging_steps=1,gradient_accumulation_steps=2,
                  report_to=[],seed=123,disable_tqdm=True)
        trainer=OrbitStiefelLoRASculptPlus(model=model,args=args,train_dataset=Data())
        trainer.train();trainer.save_model()
        assert trainer.state.global_step==4
        assert any(bool(m.lora_orbit_pruned) for _,m in layers(model))
        # Verify materialized standard PEFT load and merge against training forward.
        standard,_=configure_lorasculpt_plus_adapters(Tiny(),llm_rank=4,llm_alpha=4,connector_rank=4,connector_alpha=4,dropout=0,bias='none')
        # Match frozen weights independently of seed consumed by adapter setup.
        base={k:v for k,v in model.state_dict().items() if 'lora_' not in k}
        standard.load_state_dict(base,strict=False)
        weights=torch.load(Path(tmp)/'full/adapter_model.bin',map_location='cpu',weights_only=True)
        result=set_peft_model_state_dict(standard,weights,adapter_name='default')
        assert not [k for k in result.missing_keys if 'lora_' in k]
        inputs=torch.tensor([[1,2,3,4]])
        model.eval();standard.eval()
        torch.testing.assert_close(model(input_ids=inputs).logits,standard(input_ids=inputs).logits,rtol=1e-4,atol=1e-5)
        merged=standard.merge_and_unload()
        torch.testing.assert_close(model(input_ids=inputs).logits,merged(input_ids=inputs).logits,rtol=1e-4,atol=1e-5)
        resumed,_=make();args.output_dir=tmp+'/resume'
        resumed_trainer=OrbitStiefelLoRASculptPlus(model=resumed,args=args,train_dataset=Data())
        resumed_trainer.train(resume_from_checkpoint=tmp+'/full/checkpoint-2')
        for k,v in exported_state(model).items():
            torch.testing.assert_close(v,exported_state(resumed)[k],rtol=1e-5,atol=1e-6)
        print('PEFT_TRAIN_EXPORT_MERGE_RESUME=PASS')


if __name__=='__main__':run()
