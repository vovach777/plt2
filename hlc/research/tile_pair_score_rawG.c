// per-tile cost: user equal-pair score (C+4, 8-byte windows) on the RAW green plane (deinterleaved rrrr gggg bbbb), 48x48 tiles; vs exact MED entropy
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static const int8_t sh[32]={1,2,3,4,5,6,7,0, 2,3,4,5,6,7,0,1, 3,4,5,6,7,0,1,2, 4,5,6,7,4,5,6,7};
__attribute__((target("avx2,popcnt"))) static uint32_t tile_avx2(const uint8_t*z,int W){
  const __m256i shuf=_mm256_loadu_si256((const __m256i*)sh); uint32_t s=0;
  for(int y=0;y<48;y++){ const uint8_t*p=z+y*W; for(int x=0;x<48;x+=8){
    const __m256i v=_mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i*)(p+x)));
    s+=_mm_popcnt_u32((uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v,_mm256_shuffle_epi8(v,shuf))))+4; } }
  return s; }
__attribute__((target("sse4.2,popcnt"))) static uint32_t tile_sse(const uint8_t*z,int W){
  const __m128i s0=_mm_loadu_si128((const __m128i*)sh), s1=_mm_loadu_si128((const __m128i*)(sh+16)); uint32_t s=0;
  for(int y=0;y<48;y++){ const uint8_t*p=z+y*W; for(int x=0;x<48;x+=8){
    const __m128i v=_mm_loadl_epi64((const __m128i*)(p+x)); const __m128i vv=_mm_unpacklo_epi64(v,v);
    uint32_t m=(uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(vv,_mm_shuffle_epi8(vv,s0)))|(uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(vv,_mm_shuffle_epi8(vv,s1)))<<16;
    s+=_mm_popcnt_u32(m)+4; } }
  return s; }
static float tile_ent(const uint8_t*z,int W,const float*L){
  uint16_t h[4][256]; memset(h,0,sizeof h);
  for(int y=0;y<48;y++){ const uint8_t*p=z+y*W; for(int x=0;x<48;x+=4){h[0][p[x]]++;h[1][p[x+1]]++;h[2][p[x+2]]++;h[3][p[x+3]]++;} }
  float s=0; for(int v=0;v<256;v++) s+=L[h[0][v]+h[1][v]+h[2][v]+h[3][v]]; return 2304*log2f(2304)-s; }
int main(int argc,char**argv){
  static float L[2305]; for(int c=1;c<=2304;c++) L[c]=c*log2f(c);
  FILE*o=fopen(argv[1],"w");
  for(int a=2;a<argc;a++){
    FILE*f=fopen(argv[a],"rb"); int W,H,mx; if(fscanf(f,"P6 %d %d %d",&W,&H,&mx)!=3) return 1; fgetc(f);
    uint8_t*rgb=malloc((size_t)W*H*3); if(fread(rgb,1,(size_t)W*H*3,f)!=(size_t)W*H*3) return 1; fclose(f);
    uint8_t*z=malloc((size_t)W*H+64); uint8_t*gp=malloc((size_t)W*H+64);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
      int g=rgb[(y*W+x)*3+1]; int b=y?rgb[((y-1)*W+x)*3+1]:(x?rgb[(y*W+x-1)*3+1]:0); int aa=x?rgb[(y*W+x-1)*3+1]:b; int c=(x&&y)?rgb[((y-1)*W+x-1)*3+1]:b;
      int mn=aa<b?aa:b, M=aa<b?b:aa, p=c>=M?mn:(c<=mn?M:aa+b-c); int e=(int8_t)(uint8_t)(g-p); z[y*W+x]=e>=0?2*e:-2*e-1; }
    double td0=now(); for(int r=0;r<50;r++) for(size_t k=0;k<(size_t)W*H;k++) gp[k]=rgb[3*k+1]; double tdi=(now()-td0)/50;
    int tx=(W+47)/48, ty=(H+47)/48, NT=tx*ty; int R=300; volatile uint32_t sink=0; double t0,t1,t2,t3;
    #define TILEG(i,j) (gp+(size_t)(((j)*48+48<=H)?(j)*48:H-48)*W+(((i)*48+48<=W)?(i)*48:W-48))
    #define TILEP(i,j) (z+(size_t)(((j)*48+48<=H)?(j)*48:H-48)*W+(((i)*48+48<=W)?(i)*48:W-48))
    t0=now(); for(int r=0;r<R;r++) for(int j=0;j<ty;j++) for(int i=0;i<tx;i++) sink+=(uint32_t)tile_ent(TILEP(i,j),W,L);
    t1=now(); for(int r=0;r<R;r++) for(int j=0;j<ty;j++) for(int i=0;i<tx;i++) sink+=tile_sse(TILEG(i,j),W);
    t2=now(); for(int r=0;r<R;r++) for(int j=0;j<ty;j++) for(int i=0;i<tx;i++) sink+=tile_avx2(TILEG(i,j),W);
    t3=now(); double n=(double)R*NT*1e-9;
    printf("%s: deinterleave G %.0f ns/tile (scalar), raw-G score SSE %.0f ns/tile (%.2f GB/s of G), AVX2 %.0f ns/tile; exact entropy %.0f ns/tile\n",argv[a],tdi/NT*1e9,(t2-t1)/n,2304/((t2-t1)/n),(t3-t2)/n,(t1-t0)/n);
    for(int j=0;j<ty;j++) for(int i=0;i<tx;i++) fprintf(o,"%s %d %d %.1f %u\n",argv[a],i,j,tile_ent(TILEP(i,j),W,L),tile_sse(TILEG(i,j),W));
    free(rgb); free(z); }
  fclose(o); }
