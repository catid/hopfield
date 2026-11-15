"""Independent original-factor deletion delta; no flip/quadratic-cache scorer.

Only factors incident to a changed site can change. For each directed factor,
its source belongs to C union (C-offset); unique source coordinates count that
factor once even when both endpoints change. Evaluate before/after table/link
entries directly in longdouble, including self-links and boundary truncation.
"""
import numpy as np
LD=np.longdouble


def deletion_delta(tokens,board,readout,theta,mu_offsets,aoffs,cells):
    h,w=board.shape;cells=np.asarray(cells,dtype=np.int64)
    if not len(cells):return LD(0)
    assert np.array_equal(np.unique(cells),cells);flat=board.reshape(-1);assert np.all(tokens.reshape(-1)[cells]==1) and np.all(flat[cells]);changed=np.zeros(h*w,bool);changed[cells]=True;rows,cols=np.divmod(cells,w);theta=np.asarray(theta,dtype=LD);table=theta[:16].reshape(4,2,2);total=LD(len(cells))*(theta[17]-theta[16])
    def affected(dr,dc):
        # A factor starts at a changed cell, or terminates at a changed cell.
        rr=np.concatenate((rows,rows-dr));cc=np.concatenate((cols,cols-dc));valid=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)&(rr+dr>=0)&(rr+dr<h)&(cc+dc>=0)&(cc+dc<w);left=np.unique(rr[valid]*w+cc[valid]);right=left+dr*w+dc;return left,right
    for k,(dr,dc) in enumerate(mu_offsets):
        left,right=affected(int(dr),int(dc));old_left=flat[left].astype(np.int8);old_right=flat[right].astype(np.int8);new_left=np.where(changed[left],0,old_left);new_right=np.where(changed[right],0,old_right);total+=np.sum(table[k,new_left,new_right]-table[k,old_left,old_right],dtype=LD)
    for k,(dr,dc) in enumerate(aoffs):
        left,right=affected(int(dr),int(dc));old=flat[left].astype(np.int8)*flat[right].astype(np.int8);new=np.where(changed[left]|changed[right],0,old);weights=np.asarray(readout[k],dtype=LD).reshape(-1)[left];total-=np.sum(theta[18+k]*weights*(new-old),dtype=LD)
    return total


def deltas(tokens,boards,readout,theta,mu_offsets,aoffs,items):
    return np.array([deletion_delta(tokens[owner],boards[owner],readout[owner],theta,mu_offsets,aoffs,cells) for owner,cells in items],dtype=LD)
