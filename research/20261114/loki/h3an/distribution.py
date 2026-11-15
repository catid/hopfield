"""Exact reachable paired-union DP; complexity failures are whole-study gates."""
import itertools
import math
def paired_mass(frequency,dose=32,cap=512):
    dp=[{}for _ in range(dose+1)];dp[0][0]=1;processed=0;peaks=[1]+[0]*dose
    for mask,count in sorted(frequency.items()):
        choices=[math.comb(count,j)for j in range(min(count,dose)+1)];new=[{}for _ in range(dose+1)]
        for k in range(min(processed,dose)+1):
            for union,weight in dp[k].items():
                for j in range(min(count,dose-k)+1):
                    key=union|mask if j else union;row=new[k+j];row[key]=row.get(key,0)+weight*choices[j]
                    assert len(row)<=cap,'Reachable paired-union complexity gate exceeded'
        processed+=count;dp=new
        for k,row in enumerate(dp):peaks[k]=max(peaks[k],len(row))
    assert all(sum(dp[k].values())==math.comb(processed,k)for k in range(dose+1))
    return dp[dose],dict(population=processed,dose=dose,peak_reachable_unions_by_dose=peaks,all_dose_mass_normalizations_exact=True)
def validate():
    fixtures=[[0]*8,[3]*8,[0,1,2,3,1,2,0,3],[0,1,4,5,8,9,12,13]];cases=[]
    for fixture,values in enumerate(fixtures):
        frequency={m:values.count(m)for m in set(values)}
        for k in range(9):
            result,_=paired_mass(frequency,k);literal={}
            for selected in itertools.combinations(values,k):
                union=0
                for mask in selected:union|=mask
                literal[union]=literal.get(union,0)+1
            assert literal==result;cases.append(dict(fixture=fixture,dose=k,every_mass_exact=True))
    return dict(status='passed',cases=cases)
