// per-tile cost of the G-noise feature on 48x48 tiles: entropy (histogram + c*log2c LUT) vs sum|e| (psadbw)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <emmintrin.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); int W,H,mx; if(fscanf(f,"P6 %d %d %d",&W,&H,&mx)!=3) return 1; fgetc(f);
  uint8_t*rgb=malloc(W*H*3); if(fread(rgb,1,W*H*3,f)!=(size_t)W*H*3) return 1; fclose(f);
  uint8_t*z=malloc(W*H),*ae=malloc(W*H);  // zigzag residual and |e| of G (MED), what the encoder already has
  for(int y=0;y<H;y++)for(int x=0;x<W;x++){
    int g=rgb[(y*W+x)*3+1]; int b=y?rgb[((y-1)*W+x)*3+1]:(x?rgb[(y*W+x-1)*3+1]:0); int a=x?rgb[(y*W+x-1)*3+1]:b; int c=(x&&y)?rgb[((y-1)*W+x-1)*3+1]:b;
    int mn=a<b?a:b, M=a<b?b:a, p=c>=M?mn:(c<=mn?M:a+b-c); int e=(int8_t)(uint8_t)(g-p);
    z[y*W+x]=e>=0?2*e:-2*e-1; ae[y*W+x]=e<0?-e:e; }
  static float L[2305]; for(int c=1;c<=2304;c++) L[c]=c*log2f(c); L[0]=0;
  int tx=W/48, ty=H/48, NT=tx*ty; float*E=malloc(NT*4); uint32_t*Sa=malloc(NT*4); int R=200;
  double t0=now();
  for(int r=0;r<R;r++) for(int j=0;j<ty;j++) for(int i=0;i<tx;i++){
    uint16_t h[4][256]; memset(h,0,sizeof h);
    for(int y=0;y<48;y++){ const uint8_t*p=z+(j*48+y)*W+i*48; for(int x=0;x<48;x+=4){h[0][p[x]]++;h[1][p[x+1]]++;h[2][p[x+2]]++;h[3][p[x+3]]++;} }
    float s=0; for(int v=0;v<256;v++) s+=L[h[0][v]+h[1][v]+h[2][v]+h[3][v]];
    E[j*tx+i]=2304*log2f(2304)-s; }
  double t1=now();
  for(int r=0;r<R;r++) for(int j=0;j<ty;j++) for(int i=0;i<tx;i++){
    __m128i acc=_mm_setzero_si128(), zr=_mm_setzero_si128();
    for(int y=0;y<48;y++){ const uint8_t*p=ae+(j*48+y)*W+i*48;
      acc=_mm_add_epi64(acc,_mm_sad_epu8(_mm_loadu_si128((const __m128i*)p),zr));
      acc=_mm_add_epi64(acc,_mm_sad_epu8(_mm_loadu_si128((const __m128i*)(p+16)),zr));
      acc=_mm_add_epi64(acc,_mm_sad_epu8(_mm_loadu_si128((const __m128i*)(p+32)),zr)); }
    Sa[j*tx+i]=_mm_cvtsi128_si32(acc)+_mm_extract_epi16(acc,4); }
  double t2=now(); double n=(double)R*NT;
  printf("%dx%d, %d tiles. entropy: %.0f ns/tile (%.2f ns/px)   sum|e|: %.0f ns/tile (%.3f ns/px)   ratio %.0fx\n",W,H,NT,
    (t1-t0)/n*1e9,(t1-t0)/n/2304*1e9,(t2-t1)/n*1e9,(t2-t1)/n/2304*1e9,(t1-t0)/(t2-t1));
  printf("check tile0: H %.0f bits, sum|e| %u\n",E[0],Sa[0]);
}
