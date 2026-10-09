/* Column-order reuse GEMM with an LDS-staged epilogue (same tiles, MMA sequence
 * and block order as mma_reuse_ncol_bf16).
 *
 * The direct epilogue stores one BF16 per lane: each store instruction writes 32 B
 * into each of two output rows, and the executor charges a full-line write per
 * touched line. Here each accumulator register pair is staged through a 256 B LDS
 * buffer (2 rows x 64 columns) and read back as one 128 B row segment per store
 * instruction (32 lanes x 2 BF16). One warp per block, so the warp's program order
 * orders its LDS accesses (the ELF provider has no barrier yet); the buffer is
 * volatile and accessed with one element type so the compiler keeps that order.
 * The buffer is a __local argument: the ELF provider starts the LDS stack at the
 * LDS base, so the launcher places these 256 B above the compiler's LDS stack.
 *
 * Rows below 4 live only in A register 0 of the packed fragment (element
 * i = reg*64 + lane*2 is row i/16), so m <= 4 loads that register only and uses
 * zeros for the others: the same values the packed zero rows would supply. The
 * epilogue stops at the last row pair below m (one pair for single-row decode). */
#define GET4(p) (uint4)((p)[0],(p)[32],(p)[64],(p)[96])
#define GETA(p) (small_m ? (uint4)((p)[0],0u,0u,0u) : GET4(p))
#define MMA(a,b,c) __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_bf16_bf16_f32(a,b,c)
#define BF(x) ((ushort)__builtin_riscv_ventus_vcvt_bf16_fp32(x))
#define BIAS(nt) ((has_bias && (nt)*16+(lane&15)<n) ? __builtin_riscv_ventus_vcvt_fp32_bf16((uint)bias[(nt)*16+(lane&15)]) : 0.0f)
#define STAGE(a0,a1,a2,a3,mt) do { \
 for(uint reg=0;reg<8&&(mt)*16+reg*2<m;reg++){ \
  uint hr=lane>>4,cc=lane&15; \
  stage[hr*64+cc]=BF((a0)[reg]); stage[hr*64+16+cc]=BF((a1)[reg]); \
  stage[hr*64+32+cc]=BF((a2)[reg]); stage[hr*64+48+cc]=BF((a3)[reg]); \
  for(uint h=0;h<2;h++){ \
   uint row=(mt)*16+reg*2+h,col=nn*16+lane*2; \
   uint v=(uint)stage[h*64+lane*2]|((uint)stage[h*64+lane*2+1]<<16); \
   if(row<m&&col<n)((__global uint *)out)[(row*n+col)>>1]=v; \
  } \
 } }while(0)

kernel void mma_stage_bf16(__global const uint *a,__global const uint *b,
                          __global const ushort *bias,__global ushort *out,
                          uint m,uint n,uint k,uint has_bias,uint two_m,
                          volatile __local ushort *stage){
 uint tid=get_global_id(0),lane=tid&31,tile=tid>>5;
 uint rows=two_m?32:16,mtiles=(m+rows-1)/rows,kt=(k+15)/16;
 uint mt=(tile%mtiles)*(two_m?2:1),nn=(tile/mtiles)*4;
 uint small_m=m<=4;
 float8 c0=(float8)(BIAS(nn)),c1=(float8)(BIAS(nn+1)),c2=(float8)(BIAS(nn+2)),c3=(float8)(BIAS(nn+3));
 float8 d0=c0,d1=c1,d2=c2,d3=c3;
 __global const uint *ap=a+mt*kt*128+lane;
 __global const uint *bp=b+nn*kt*128+lane;
 for(uint kk=0;kk<kt;kk++,ap+=128,bp+=128){
  uint4 aa=GETA(ap),b0=GET4(bp),b1=GET4(bp+kt*128),b2=GET4(bp+kt*256),b3=GET4(bp+kt*384);
  c0=MMA(aa,b0,c0);c1=MMA(aa,b1,c1);c2=MMA(aa,b2,c2);c3=MMA(aa,b3,c3);
  if(two_m){uint4 ab=GET4(ap+kt*128);d0=MMA(ab,b0,d0);d1=MMA(ab,b1,d1);d2=MMA(ab,b2,d2);d3=MMA(ab,b3,d3);}
 }
 STAGE(c0,c1,c2,c3,mt);
 if(two_m)STAGE(d0,d1,d2,d3,mt+1);
}

/* Direct-epilogue variant with the same m <= 4 A-fragment trim (no LDS, no extra
 * staging registers): isolates the A-load saving from the store staging. */
#define STORE_ONE(acc,mt,nt) do { \
 for(uint reg=0;reg<8;reg++){ \
  uint linear=reg*32+lane,row=(mt)*16+linear/16,col=(nt)*16+linear%16; \
  if(row<m&&col<n)out[row*n+col]=BF((acc)[reg]); \
 } }while(0)

kernel void mma_trim_bf16(__global const uint *a,__global const uint *b,
                         __global const ushort *bias,__global ushort *out,
                         uint m,uint n,uint k,uint has_bias,uint two_m){
 uint tid=get_global_id(0),lane=tid&31,tile=tid>>5;
 uint rows=two_m?32:16,mtiles=(m+rows-1)/rows,kt=(k+15)/16;
 uint mt=(tile%mtiles)*(two_m?2:1),nn=(tile/mtiles)*4;
 uint small_m=m<=4;
 float8 c0=(float8)(BIAS(nn)),c1=(float8)(BIAS(nn+1)),c2=(float8)(BIAS(nn+2)),c3=(float8)(BIAS(nn+3));
 float8 d0=c0,d1=c1,d2=c2,d3=c3;
 __global const uint *ap=a+mt*kt*128+lane;
 __global const uint *bp=b+nn*kt*128+lane;
 for(uint kk=0;kk<kt;kk++,ap+=128,bp+=128){
  uint4 aa=GETA(ap),b0=GET4(bp),b1=GET4(bp+kt*128),b2=GET4(bp+kt*256),b3=GET4(bp+kt*384);
  c0=MMA(aa,b0,c0);c1=MMA(aa,b1,c1);c2=MMA(aa,b2,c2);c3=MMA(aa,b3,c3);
  if(two_m){uint4 ab=GET4(ap+kt*128);d0=MMA(ab,b0,d0);d1=MMA(ab,b1,d1);d2=MMA(ab,b2,d2);d3=MMA(ab,b3,d3);}
 }
 STORE_ONE(c0,mt,nn);STORE_ONE(c1,mt,nn+1);STORE_ONE(c2,mt,nn+2);STORE_ONE(c3,mt,nn+3);
 if(two_m){STORE_ONE(d0,mt+1,nn);STORE_ONE(d1,mt+1,nn+1);STORE_ONE(d2,mt+1,nn+2);STORE_ONE(d3,mt+1,nn+3);}
}
