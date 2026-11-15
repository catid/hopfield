/* Exact graph-feasibility check: no shortest distances or learned quantities. */
#include <stdint.h>
#include <stddef.h>

static int valid(const uint8_t *t,const uint8_t *p,int h,int w) {
    const int cells=h*w;int start=-1,goal=-1,ns=0,ng=0,total=0;
    for(int q=0;q<cells;q++) {
        if(t[q]==2){start=q;ns++;} if(t[q]==3){goal=q;ng++;}
        if(!p[q])continue;
        if(!t[q])return 0;
        total++;int row=q/w,col=q%w;
        int d=(row>0&&p[q-w])+(row+1<h&&p[q+w])+(col>0&&p[q-1])+(col+1<w&&p[q+1]);
        if(d!=((t[q]==2||t[q]==3)?1:2))return 0;
    }
    if(ns!=1||ng!=1||!p[start]||!p[goal])return 0;
    int prev=-1,q=start,seen=1;
    while(q!=goal&&seen<=total) {
        int next=-1,row=q/w,col=q%w;
        if(row>0&&p[q-w]&&q-w!=prev)next=q-w;
        if(row+1<h&&p[q+w]&&q+w!=prev)next=q+w;
        if(col>0&&p[q-1]&&q-1!=prev)next=q-1;
        if(col+1<w&&p[q+1]&&q+1!=prev)next=q+1;
        if(next<0)return 0;
        prev=q;q=next;seen++;
    }
    return q==goal&&seen==total;
}

void paths_valid(const uint8_t *tokens,const uint8_t *predictions,int n,int h,int w,uint8_t *out) {
    size_t stride=(size_t)h*w;
    for(int i=0;i<n;i++)out[i]=(uint8_t)valid(tokens+i*stride,predictions+i*stride,h,w);
}
