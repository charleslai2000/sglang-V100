import torch
from sglang.srt.layers.attention.mamba.ops.mamba_ssm import selective_state_update
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID

def ref(state,x,dt,A,B,C,D,bias):
    dt=torch.nn.functional.softplus(dt+bias)
    ns=state*torch.exp(A*dt[...,None])+B.repeat_interleave(h//B.shape[1],dim=1)[:, :, None, :]*dt[...,None]*x[...,None]
    C_h=C.repeat_interleave(h//C.shape[1],dim=1)
    D_h=D
    return ns,((ns*C_h[:,:,None,:]).sum(-1)+x*D_h).to(x.dtype)
for dtype in (torch.float32,torch.float16):
  for dim,dstate,ng in ((128,256,1),(256,128,2)):
    dev='cuda'; b,h=4,4; torch.manual_seed(73)
    state=torch.randn(8,h,dim,dstate,device=dev,dtype=torch.float32); original=state.clone();idx=torch.tensor([1,3,4,6],device=dev,dtype=torch.int32)
    x=torch.randn(b,h,dim,device=dev,dtype=dtype);dtbase=torch.randn(b,h,device=dev,dtype=dtype);Abase=-torch.rand(h,device=dev)-.5;bb=torch.randn(h,device=dev,dtype=dtype);Db=torch.randn(h,device=dev,dtype=dtype)
    dt=dtbase[...,None].expand(-1,-1,dim);A=Abase[:,None,None].expand(-1,dim,dstate);bias=bb[:,None].expand(-1,dim);D=Db[:,None].expand(-1,dim)
    B=torch.randn(b,ng,dstate,device=dev,dtype=dtype);C=torch.randn_like(B);out=torch.empty_like(x)
    assert A.stride(-1)==A.stride(-2)==dt.stride(-1)==bias.stride(-1)==0
    selective_state_update(state,x,dt,A,B,C,D=D,dt_bias=bias,dt_softplus=True,state_batch_indices=idx,pad_slot_id=PAD_SLOT_ID,out=out)
    ref_state,ref_out=ref(original[idx.long()],x,dtbase[...,None].expand(-1,-1,dim),A,B,C,D,bias)
    print(dtype,dim,dstate,'state max',(state[idx.long()]-ref_state).abs().max().item(),'out max',(out-ref_out).abs().max().item())
    assert torch.allclose(state[idx.long()],ref_state,rtol=5e-3,atol=3e-2)
    assert torch.allclose(out,ref_out,rtol=5e-3,atol=3e-2)
print('M1B tied-head direct regression PASS')
