/* SwiGLU written straight into the packed MMA A-tile layout of the down GEMM.
 *
 * Replaces packed_swiglu_bf16 (row-major output) followed by pack_a_bf16 over that
 * output. Each lane computes the same bf16x2 SiLU and multiply on the same element
 * pair that pack_a_bf16 would read from the SwiGLU buffer (pairs are aligned since
 * k is even), so the packed tiles are bit-identical and the row-major SwiGLU
 * buffer is never written. Rows beyond m are zero, as in pack_a_bf16. */
kernel void swiglu_pack_bf16(__global const uint *g,__global const uint *u,__global uint *out,
                             uint m,uint k){
 uint tid=get_global_id(0),lane=tid&31,tile=tid>>5;
 uint kt=(k+15)/16,mt=tile/kt,kk=tile%kt;
 for(uint reg=0;reg<4;reg++){
  uint i=reg*64+lane*2,row=mt*16+i/16,col=kk*16+i%16;
  uint v=0;
  if(row<m&&col<k){
   uint j=(row*k+col)>>1;
   v=__builtin_riscv_ventus_vmul_bf16x2(__builtin_riscv_ventus_vsilu_approx_bf16x2(g[j]),u[j]);
  }
  out[tile*128+reg*32+lane]=v;
 }
}
