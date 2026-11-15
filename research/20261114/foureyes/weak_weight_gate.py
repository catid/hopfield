"""H4z exact dyadic arithmetic and positive-energy stopping gates."""
import json
import time
from weak_weight_study import ROOT,HERE,OUT,PROTOCOL,sha,save,verify,setup


def run(bank):
    lock,torch,np,GS,SS,SP=setup(bank)
    from weak_weight_math import check_state
    from run import captured_search_class
    from sparse_backend import sparse_search_class
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic()
    dense=captured_search_class(GS);sparse=sparse_search_class(GS,SP)
    sol,free,_=SS.load_puzzles('full-all',410000,16)
    kw=dict(temp=.4,theta=.45,focused=True,C=4,tau=0.,tol=.001,ebar=0.,cap=4e11,check=4096,seed=0,anneal=(.4,.31,524288))
    pairs=torch.triu_indices(81,81,1,device='cuda');eye=torch.eye(9,device='cuda')
    def energy(w):
        J=torch.as_tensor(w.astype(np.float32)/256,device='cuda')
        S=torch.zeros(81,81,9,9,device='cuda');W=J[:,None,None]*eye
        S[pairs[0],pairs[1]]=W;S[pairs[1],pairs[0]]=W.transpose(1,2)
        return S,torch.zeros(81,9,device='cuda')
    compared=hidden=integer=0
    for arm in ['weak16','weak64','weak256']:
        with np.load(OUT/'models'/f'b{bank:02d}-{arm}.npz',allow_pickle=False)as z:w=z['weight256']
        S,B=energy(w);x=dense(S,B,sol,free,**kw);y=sparse(S,B,sol,free,**kw)
        altered=sol.copy();rng=np.random.default_rng(20261117000+bank);altered[free]=rng.integers(0,9,int(free.sum()))
        z=sparse(S,B,altered,free,**kw)
        for step in range(6):
            check_state(y,w);integer+=1
            a,b,c=x.state_dict(),y.state_dict(),z.state_dict();assert a.keys()==b.keys()==c.keys()
            for key,value in a.items():
                assert torch.equal(value,b[key])if torch.is_tensor(value)else value==b[key]
                compared+=1
                if key not in('tgt','ok','bestok'):
                    assert torch.equal(b[key],c[key])if torch.is_tensor(b[key])else b[key]==c[key]
                    hidden+=1
            assert torch.equal(x.boards(),y.boards())and torch.equal(y.boards(),z.boards())
            if step<4:
                for state in(x,y,z):state.step_check()
            elif step==4:
                for state in(x,y,z):state._moves(127);state.t+=127
        del x,y,z
    toy=[]
    for numerator in [16,4,1]:
        w=np.zeros(3240,np.uint16);w[0]=numerator;S,B=energy(w)
        for equal in [True,False]:
            labels=np.zeros((1,81),np.int64);labels[0,1]=0 if equal else 1
            mask=np.ones((1,81),bool);mask[0,:2]=False
            state=sparse(S,B,labels,mask,**kw)
            check_state(state,w)
            for _ in range(3):
                if not len(state.act):break
                state.step_check();check_state(state,w)
            if equal:
                assert not state.stopped.any()and not(state.found>=0).any()
                assert torch.equal(state.P,torch.full_like(state.P,numerator/256))
            else:assert state.stopped.all()
            toy.append(dict(numerator=numerator,unavoidable_conflict=equal,stopped=int(state.stopped.sum())))
    result=dict(status='passed',bank=bank,protocol_sha256=sha(PROTOCOL),gate_sha256=sha(__file__),integer_helper_sha256=sha(HERE/'weak_weight_math.py'),general_sparse_comparisons=compared,hidden_target_comparisons=hidden,integer_checks=integer,toy_cases=toy,wall_seconds=time.monotonic()-start)
    save(OUT/f'gate-bank{bank}.json',result);print(json.dumps(result))
