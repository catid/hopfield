"""Unchanged model eligibility, forward masks and Python literal reverse reference."""
import numpy as np
def eligibility(m):
    q=int(m['q']);assert q==15;F=m['F'];J=m['J'];assert F.shape==J.shape==(q*q,q*q)and np.array_equal(F,F.T)and np.isin(F,[0,1]).all()and not F.diagonal().any();assert np.isfinite(J).all()and np.array_equal(J,J.T)and(J[F.astype(bool)]>0).all()and not J[~F.astype(bool)].any()and not m['h'].any()
    for row in range(q):
        cells=np.arange(q)+q*row;assert(F[np.ix_(cells,cells)]+np.eye(q)==1).all()
    for col in range(q):
        cells=q*np.arange(q)+col;assert(F[np.ix_(cells,cells)]+np.eye(q)==1).all()
    return dict(q=q,strict_positive_F_energy=True,complete_row_and_column_cliques=True,proof='At most one occupied cell per row/column; q=n implies exactly one per row/column. Exhaustive row-wise F-consistent recursion therefore enumerates every cardinality-q zero-energy state.')

def forward(F,q):
    blocked={(r,c):[sum(1<<j for j in range(q)if F[q*r+c,q*t+j])for t in range(q)]for r in range(q)for c in range(q)};answers=[];full=(1<<q)-1
    def visit(r,masks,columns):
        if r==q:answers.append(columns);return
        choices=full&~masks[r]
        while choices:
            bit=choices&-choices;choices^=bit;c=bit.bit_length()-1;new=masks[:]
            for t in range(r+1,q):new[t]|=blocked[r,c][t]
            visit(r+1,new,columns+[c])
    visit(0,[0]*q,[]);return np.array(answers,np.uint8).reshape(-1,q)

def reverse_literal(F,q):
    answers=[];selected=[];columns=[0]*q
    def visit(row):
        if row<0:answers.append(columns.copy());return
        for col in range(q):
            cell=q*row+col
            if any(F[cell,old]for old in selected):continue
            columns[row]=col;selected.append(cell);visit(row-1);selected.pop()
    visit(q-1);a=np.array(answers,np.uint8).reshape(-1,q)
    if len(a):a=a[np.lexsort(a.T[::-1])]
    return a



