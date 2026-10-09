/* Column-major block order variant of mma_reuse_bf16 (same tiles, registers and
 * MMA sequence). Consecutive blocks take consecutive row tiles of the same 64-wide
 * weight strip, so concurrently resident blocks request the same B lines instead
 * of re-streaming every weight strip once per row tile. Only the block index map
 * differs from mma_reuse_bf16. */
#define GET4(p) (uint4)((p)[0],(p)[32],(p)[64],(p)[96])
#define MMA(a,b,c) __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_bf16_bf16_f32(a,b,c)
#define BF(x) ((ushort)__builtin_riscv_ventus_vcvt_bf16_fp32(x))
#define STORE_ONE(acc,mt,nt) do { \
 for(uint reg=0;reg<8;reg++){ \
  uint linear=reg*32+lane,row=(mt)*16+linear/16,col=(nt)*16+linear%16; \
  if(row<m&&col<n)out[row*n+col]=BF((acc)[reg]); \
 } }while(0)
#define BIAS(nt) ((has_bias && (nt)*16+(lane&15)<n) ? __builtin_riscv_ventus_vcvt_fp32_bf16((uint)bias[(nt)*16+(lane&15)]) : 0.0f)

kernel void mma_reuse_ncol_bf16(__global const uint *a,__global const uint *b,
                               __global const ushort *bias,__global ushort *out,
                               uint m,uint n,uint k,uint has_bias,uint two_m){
 uint tid=get_global_id(0),lane=tid&31,tile=tid>>5;
 uint rows=two_m?32:16,mtiles=(m+rows-1)/rows,kt=(k+15)/16;
 uint mt=(tile%mtiles)*(two_m?2:1),nn=(tile/mtiles)*4;
 float8 c0=(float8)(BIAS(nn)),c1=(float8)(BIAS(nn+1)),c2=(float8)(BIAS(nn+2)),c3=(float8)(BIAS(nn+3));
 float8 d0=c0,d1=c1,d2=c2,d3=c3;
 __global const uint *ap=a+mt*kt*128+lane;
 __global const uint *bp=b+nn*kt*128+lane;
 for(uint kk=0;kk<kt;kk++,ap+=128,bp+=128){
  uint4 aa=GET4(ap),b0=GET4(bp),b1=GET4(bp+kt*128),b2=GET4(bp+kt*256),b3=GET4(bp+kt*384);
  c0=MMA(aa,b0,c0);c1=MMA(aa,b1,c1);c2=MMA(aa,b2,c2);c3=MMA(aa,b3,c3);
  if(two_m){uint4 ab=GET4(ap+kt*128);d0=MMA(ab,b0,d0);d1=MMA(ab,b1,d1);d2=MMA(ab,b2,d2);d3=MMA(ab,b3,d3);}
 }
 STORE_ONE(c0,mt,nn);STORE_ONE(c1,mt,nn+1);STORE_ONE(c2,mt,nn+2);STORE_ONE(c3,mt,nn+3);
 if(two_m){STORE_ONE(d0,mt+1,nn);STORE_ONE(d1,mt+1,nn+1);STORE_ONE(d2,mt+1,nn+2);STORE_ONE(d3,mt+1,nn+3);}
}

/* Column-major block order variant of mma_packed_bf16 (one 16x16 tile per warp). */
kernel void mma_packed_ncol_bf16(__global const uint *a,__global const uint *b,
                                __global const ushort *bias,__global ushort *out,
                                uint m,uint n,uint k,uint has_bias){
 uint tid=get_global_id(0),lane=tid&31,tile=tid>>5;
 uint mtiles=(m+15)/16,kt=(k+15)/16,mt=tile%mtiles,nn=tile/mtiles;
 float bv=0;
 if(has_bias && nn*16+(lane&15)<n)bv=__builtin_riscv_ventus_vcvt_fp32_bf16((uint)bias[nn*16+(lane&15)]);
 float8 acc=(float8)(bv);
 uint ai=mt*kt*128+lane,bi=nn*kt*128+lane;
 for(uint kk=0;kk<kt;kk++,ai+=128,bi+=128){
  uint4 aa=(uint4)(a[ai],a[ai+32],a[ai+64],a[ai+96]);
  uint4 bb=(uint4)(b[bi],b[bi+32],b[bi+64],b[bi+96]);
  acc=__builtin_riscv_ventus_mma_m16n16k16_row_col_f32_bf16_bf16_f32(aa,bb,acc);
 }
 for(uint reg=0;reg<8;reg++){
  uint linear=reg*32+lane,row=mt*16+linear/16,col=nn*16+linear%16;
  if(row<m&&col<n)out[row*n+col]=(ushort)__builtin_riscv_ventus_vcvt_bf16_fp32(acc[reg]);
 }
}
