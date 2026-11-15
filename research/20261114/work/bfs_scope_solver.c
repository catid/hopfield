/* Ordinary single-threaded BFS on supplied orthogonal open-cell graphs. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int solve_graphs(const uint8_t *tokens,int n,int h,int w,uint8_t *boards,int32_t *distances,uint8_t *status) {
    int cells=h*w;int32_t *parent=malloc((size_t)cells*sizeof(int32_t));int32_t *queue=malloc((size_t)cells*sizeof(int32_t));
    if(!parent||!queue){free(parent);free(queue);return -1;}
    for(int sample=0;sample<n;sample++) {
        const uint8_t *t=tokens+(size_t)sample*cells;uint8_t *out=boards+(size_t)sample*cells;
        memset(out,0,(size_t)cells);memset(parent,255,(size_t)cells*sizeof(int32_t));distances[sample]=-1;status[sample]=0;
        int start=-1,goal=-1,ns=0,ng=0;
        for(int q=0;q<cells;q++){if(t[q]==2){start=q;ns++;}if(t[q]==3){goal=q;ng++;}}
        if(ns!=1||ng!=1)continue;
        int head=0,tail=1;queue[0]=start;parent[start]=start;
        while(head<tail) {
            int q=queue[head++];if(q==goal)break;
            int r=q/w,c=q%w;int neighbors[4]={r>0?q-w:-1,c>0?q-1:-1,c+1<w?q+1:-1,r+1<h?q+w:-1};
            for(int d=0;d<4;d++){int v=neighbors[d];if(v>=0&&t[v]&&parent[v]<0){parent[v]=q;queue[tail++]=v;}}
        }
        if(parent[goal]<0)continue;
        int q=goal,length=0;out[q]=1;
        while(q!=start){q=parent[q];out[q]=1;length++;}
        distances[sample]=length;status[sample]=1;
    }
    free(parent);free(queue);return 0;
}
