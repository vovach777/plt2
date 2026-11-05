// 1-D k-means (K=8) of per-tile H_G via a 256-bin histogram; input: tile features scaled to u8 (H_G*32, 0..255)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#define K 8
static void km_hist(const uint8_t* f,int n,uint8_t th[K-1]){
  uint32_t h[256]={0}; for(int i=0;i<n;i++) h[f[i]]++;
  uint32_t cs[257]; uint64_t sx[257]; cs[0]=0; sx[0]=0;            // prefix counts and sums -> each Lloyd step is O(K)
  for(int v=0;v<256;v++){cs[v+1]=cs[v]+h[v]; sx[v+1]=sx[v]+(uint64_t)h[v]*v;}
  int c2[K]; // centres *2 (half-integer precision)
  for(int k=0,v=0;k<K;k++){ uint32_t t=(uint32_t)((2ull*k+1)*n/(2*K)); while(v<255&&cs[v+1]<=t)v++; c2[k]=2*v; }
  for(int it=0;it<64;it++){
    int b[K+1]; b[0]=0; b[K]=256; for(int k=1;k<K;k++) b[k]=(c2[k-1]+c2[k])/4+1;   // bin v -> class k if b[k]<=v<b[k+1]
    int ch=0; for(int k=0;k<K;k++){ uint32_t m=cs[b[k+1]]-cs[b[k]]; if(!m) continue;
      int nc=(int)((2*(sx[b[k+1]]-sx[b[k]])+m/2)/m); if(nc!=c2[k]){c2[k]=nc;ch=1;} }
    if(!ch) break;
  }
  for(int k=1;k<K;k++) th[k-1]=(uint8_t)((c2[k-1]+c2[k])/4+1);
}
static void quant(const uint8_t* f,int n,uint8_t th[K-1]){
  uint32_t h[256]={0}; for(int i=0;i<n;i++) h[f[i]]++;
  uint32_t c=0; int k=1; for(int v=0;v<256&&k<K;v++){ c+=h[v]; while(k<K && c*(uint64_t)K>=(uint64_t)k*n){th[k-1]=v+1;k++;} }
  while(k<K) th[k++ -1]=255;
}
int main(){
  int ns[]={336,1247,5200,20000}; srand(1);
  for(int t=0;t<4;t++){ int n=ns[t]; uint8_t* f=malloc(n); for(int i=0;i<n;i++){ int a=0; for(int j=0;j<4;j++) a+=rand()%64; f[i]=a; }
    uint8_t th[K-1]; volatile int s=0; int R=200000/(n/300+1);
    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a); for(int r=0;r<R;r++){ f[r%n]^=1; km_hist(f,n,th); s+=th[3]; } clock_gettime(CLOCK_MONOTONIC,&b);
    double tk=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/R;
    clock_gettime(CLOCK_MONOTONIC,&a); for(int r=0;r<R;r++){ f[r%n]^=1; quant(f,n,th); s+=th[3]; } clock_gettime(CLOCK_MONOTONIC,&b);
    double tq=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/R;
    printf("tiles %6d: kmeans_hist %7.2f us   quantile %7.2f us\n",n,tk/1e3,tq/1e3); free(f); }
}
