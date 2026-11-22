"""CPU mathematical and lifecycle tests; no LLaVA/PEFT installation required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location('orbit_core', Path(__file__).resolve().parents[1] / 'llava/model/orbit_stiefel.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)


class Adapter(nn.Linear):
    def __init__(self, n=16, m=12, rank=4):
        super().__init__(n, m)
        self.weight.requires_grad_(False); self.bias.requires_grad_(False)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(n, rank, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(rank, m, bias=False)})
        nn.init.zeros_(self.lora_B['default'].weight)
        self.lora_dropout = nn.ModuleDict({'default': nn.Dropout(0)})
        self.scaling = {'default': 2.0}


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Adapter()
        self.mm_projector = Adapter()


class OrbitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def model(self):
        return core.install(Toy(), dict(keep_ratio=.5, warmup_steps=2, orbit_iterations=4, protection_rank=3))

    def test_equivalent_rotations_balancing_and_error_contraction(self):
        b,a=torch.randn(12,4),torch.randn(4,16)
        r,_=core.qr_frame(torch.randn(4,4))
        torch.testing.assert_close(b@a,(b@r)@(r.T@a))
        bc,ac=core.balanced_factors(b,a)
        torch.testing.assert_close(b@a,bc@ac,atol=1e-5,rtol=1e-5)
        sb,sa,mb,ma,report=core.orbit_prune(b,a,.5,6)
        expected=((sb.double()@sa.double())-(b.double()@a.double())).square().sum()
        torch.testing.assert_close(core.product_error(b,a,sb,sa),expected)
        original=next(c for c in report['candidates'] if c['label']=='original')
        self.assertTrue(original['admissible'])
        self.assertLessEqual(report['error'],original['error']+1e-8)
        self.assertEqual(int(mb.sum()),24);self.assertEqual(int(ma.sum()),32)

    def test_zero_initialization_orthogonality_and_scale_gradients(self):
        model=self.model();x=torch.randn(3,16)
        torch.testing.assert_close(model.q_proj(x),F.linear(x,model.q_proj.weight,model.q_proj.bias))
        qb,qa=core.directions(model.q_proj)
        torch.testing.assert_close(qb.T@qb,torch.eye(4),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(qa.T@qa,torch.eye(4),atol=1e-5,rtol=1e-5)
        model.q_proj(x).square().mean().backward()
        self.assertGreater(float(model.q_proj.lora_spectral_scale.grad.abs().sum()),0)

    def test_pruning_transition_support_proximal_export_and_resume(self):
        model=self.model();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0)
        for step in range(2):
            opt.zero_grad();x=torch.randn(5,16)
            loss=model.q_proj(x).square().mean()+model.mm_projector(x).square().mean()
            loss.backward();opt.step();core.post_step(model,opt,.01)
        b,a=core.effective_factors(model.q_proj);before=(b@a).detach()
        self.assertEqual(core.initialize_orbit(model,opt,1),[])
        reports=core.initialize_orbit(model,opt,2)
        self.assertEqual(len(reports),1)
        afterb,aftera=core.effective_factors(model.q_proj)
        torch.testing.assert_close((afterb@aftera-before).double().square().sum(),torch.tensor(reports[0]['error'],dtype=torch.double),atol=1e-6,rtol=1e-3)
        self.assertEqual(core.initialize_orbit(model,opt,3),[])
        self.assertFalse(bool(model.mm_projector.lora_orbit_pruned))
        opt.zero_grad();model.q_proj(torch.randn(5,16)).square().mean().backward();opt.step();core.post_step(model,opt,.01)
        layer=model.q_proj
        for key in ['A','B']:
            p=getattr(layer,'lora_'+key)['default'].weight;mask=getattr(layer,'lora_mask_'+key)
            self.assertTrue(torch.equal(p[~mask],torch.zeros_like(p[~mask])))
        x=torch.randn(3,16);state=core.exported_state(model)
        expected=F.linear(x,layer.weight,layer.bias)+2*F.linear(F.linear(x,state['q_proj.lora_A.default.weight']),state['q_proj.lora_B.default.weight'])
        torch.testing.assert_close(layer(x),expected)
        with tempfile.TemporaryDirectory() as directory:
            core.save_training_state(model,directory)
            restored=self.model()
            restored.q_proj.weight.data.copy_(layer.weight);restored.q_proj.bias.data.copy_(layer.bias)
            core.load_training_state(restored,directory)
            torch.testing.assert_close(restored.q_proj(x),layer(x))
            self.assertEqual(core.initialize_orbit(restored,opt,100),[])

    def test_subspace_penalty_equals_explicit_scaled_delta(self):
        model=self.model();layer=model.q_proj
        layer.lora_spectral_scale.data.copy_(torch.tensor([.1,-.2,.3,.05]))
        protected,_,_=core.regularizers(model)
        b,a=core.effective_factors(layer);delta=2*b@a
        expected=model.orbit_stiefel_config['protection_lambda']*((layer.lora_protected_U.T@delta).square().sum()+(delta@layer.lora_protected_V).square().sum())
        torch.testing.assert_close(protected,expected)
        protected.backward();self.assertTrue(torch.isfinite(layer.lora_spectral_scale.grad).all())

    def test_proximal_scale_sparsity_and_full_rank_rejection(self):
        model=self.model();layer=model.mm_projector
        layer.lora_spectral_scale.data.copy_(torch.tensor([1e-9,-1e-9,.1,-.1]))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0)
        core.post_step(model,opt,.01)
        self.assertEqual(int(torch.count_nonzero(layer.lora_spectral_scale)),2)
        b,a=torch.randn(12,4),torch.randn(4,16)
        with self.assertRaises(RuntimeError):core.orbit_prune(b,a,.01,2)
        mask=core.topk_mask(torch.ones(2,4),.5)
        self.assertEqual(mask.flatten().tolist(),[True]*4+[False]*4)

    def test_weak_direction_uses_equivalent_coordinates_without_budget_increase(self):
        qb,_=core.qr_frame(torch.randn(128,4))
        qa,_=core.qr_frame(torch.randn(128,4))
        scale=torch.tensor([1.,1.,1.,1e-4]).sqrt()
        b,a=qb*scale,scale[:,None]*qa.T
        sb,sa,mb,ma,report=core.orbit_prune(b,a,.1,10,seed=42)
        self.assertTrue(core.full_rank(sb));self.assertTrue(core.full_rank(sa.T))
        self.assertEqual(int(mb.sum()),int(b.numel()*.1))
        self.assertEqual(int(ma.sum()),int(a.numel()*.1))
        self.assertTrue(report['selected'].startswith('mixed-fallback'))
        self.assertTrue(all(not c['admissible'] for c in report['candidates'] if not c['label'].startswith('mixed-fallback')))


if __name__=='__main__':unittest.main()
