"""Independent 60-digit Decimal inversion for all 257 binomial outcomes."""
import ast
import hashlib
import math
from decimal import Decimal,localcontext
from run import ROOT,HERE,save,sha,now


def main():
    tree=ast.parse((HERE/'audit_recovery_confirm.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='cp_limits')
    ns={'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'cp','exec'),ns)
    def decimal_lower(k,n):
        if k==0:return Decimal(0)
        a,b=Decimal(0),Decimal(1)
        for _ in range(120):
            p=(a+b)/2;q=1-p
            tail=sum(Decimal(math.comb(n,x))*p**x*q**(n-x) for x in range(k,n+1))
            if tail<Decimal('.05'):a=p
            else:b=p
        return (a+b)/2
    maximum=0.;rows=[]
    with localcontext() as context:
        context.prec=60
        for k in range(257):
            lower=float(decimal_lower(k,256));upper=float(1-decimal_lower(256-k,256))
            actual=ns['cp_limits'](k,256)
            error=max(abs(actual[0]-lower),abs(actual[1]-upper));maximum=max(maximum,error)
            assert error<3e-15;rows.append({'k':k,'lower':lower,'upper':upper})
    assert rows[238]['lower']<.9<=rows[239]['lower']
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'cp_function_ast_sha256':hashlib.sha256(ast.unparse(fn).encode()).hexdigest(),
        'outcomes_checked':257,'limits_checked':514,'decimal_precision':60,'maximum_absolute_error':maximum,
        'support_minimum_successes':239,'rows':rows}
    save(ROOT/'artifacts/research/20261114/ripper/recovery_confirm_cp_validation.json',r);print({k:v for k,v in r.items() if k!='rows'})


if __name__=='__main__':main()
