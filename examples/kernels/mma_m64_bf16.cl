/* 64x64 output tile per warp in column block order (four row tiles x four column
 * tiles, 16 accumulators). Weight (B) traffic per output halves against the 32x64
 * tile and activation (A) traffic stays equal, so GEMM memory traffic falls by a
 * third at the cost of more registers. Requires m to be covered by 64-row tiles of
 * the packed A scratch (row tails are computed and not stored). */
#define GET4(p) (uint4)((p)[0],(p)[32],(p)[64],(p)[96])
#define MMA(a,b,c) __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_bf16_bf16_f32(a,b,c)
#define BF(x) ((ushort)__builtin_riscv_ventus_vcvt_bf16_fp32(x))
#define BIAS(nt) ((has_bias && (nt)*16+(lane&15)<n) ? __builtin_riscv_ventus_vcvt_fp32_bf16((uint)bias[(nt)*16+(lane&15)]) : 0.0f)
#define STORE_ONE(acc,mt,nt) do { \
 for(uint reg=0;reg<8;reg++){ \
  uint linear=reg*32+lane,row=(mt)*16+linear/16,col=(nt)*16+linear%16; \
  if(row<m&&col<n)out[row*n+col]=BF((acc)[reg]); \
 } }while(0)
#define ROW(ab,x0,x1,x2,x3) do{x0=MMA(ab,b0,x0);x1=MMA(ab,b1,x1);x2=MMA(ab,b2,x2);x3=MMA(ab,b3,x3);}while(0)

kernel void mma_m64_bf16(__global const uint *a,__global const uint *b,
                        __global const ushort *bias,__global ushort *out,
                        uint m,uint n,uint k,uint has_bias){
 uint tid=get_global_id(0),lane=tid&31,tile=tid>>5;
 uint mtiles=(m+63)/64,kt=(k+15)/16;
 uint mt=(tile%mtiles)*4,nn=(tile/mtiles)*4;
 float8 c0=(float8)(BIAS(nn)),c1=(float8)(BIAS(nn+1)),c2=(float8)(BIAS(nn+2)),c3=(float8)(BIAS(nn+3));
 float8 d0=c0,d1=c1,d2=c2,d3=c3,e0=c0,e1=c1,e2=c2,e3=c3,f0=c0,f1=c1,f2=c2,f3=c3;
 __global const uint *ap=a+mt*kt*128+lane;
 __global const uint *bp=b+nn*kt*128+lane;
 for(uint kk=0;kk<kt;kk++,ap+=128,bp+=128){
  uint4 b0=GET4(bp),b1=GET4(bp+kt*128),b2=GET4(bp+kt*256),b3=GET4(bp+kt*384);
  uint4 a0=GET4(ap);ROW(a0,c0,c1,c2,c3);
  uint4 a1=GET4(ap+kt*128);ROW(a1,d0,d1,d2,d3);
  uint4 a2=GET4(ap+kt*256);ROW(a2,e0,e1,e2,e3);
  uint4 a3=GET4(ap+kt*384);ROW(a3,f0,f1,f2,f3);
 }
 STORE_ONE(c0,mt,nn);STORE_ONE(c1,mt,nn+1);STORE_ONE(c2,mt,nn+2);STORE_ONE(c3,mt,nn+3);
 STORE_ONE(d0,mt+1,nn);STORE_ONE(d1,mt+1,nn+1);STORE_ONE(d2,mt+1,nn+2);STORE_ONE(d3,mt+1,nn+3);
 STORE_ONE(e0,mt+2,nn);STORE_ONE(e1,mt+2,nn+1);STORE_ONE(e2,mt+2,nn+2);STORE_ONE(e3,mt+2,nn+3);
 STORE_ONE(f0,mt+3,nn);STORE_ONE(f1,mt+3,nn+1);STORE_ONE(f2,mt+3,nn+2);STORE_ONE(f3,mt+3,nn+3);
}
