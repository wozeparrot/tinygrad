import math

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import GL, TileLayout

NUM_WORKERS = 1
Q_BLOCK_SIZE = 16
KV_BLOCK_SIZE = 16

def flash_attention(xq, xk, xv, attn_mask:Tensor|None=None, is_causal:bool=False):
  if len(xq.shape) == 3: xq, xk, xv = xq.unsqueeze(0), xk.unsqueeze(0), xv.unsqueeze(0)

  odtype = xq.dtype
  xq, xk, xv = xq.transpose(1, 2).cast(dtypes.bfloat16), xk.transpose(1, 2).cast(dtypes.bfloat16), xv.transpose(1, 2).cast(dtypes.bfloat16)

  _, N_, _, D_ = xq.shape
  block_size = max(Q_BLOCK_SIZE, KV_BLOCK_SIZE)
  assert D_ % block_size == 0, f"embedding dimension must be multiple of block size, got {D_=} {block_size=}"

  # pad to multiple of block size
  xq = xq.pad(((0, 0), (0, (block_size - (xq.shape[1] % block_size)) % block_size), (0, 0), (0, 0)))
  xk = xk.pad(((0, 0), (0, (block_size - (xk.shape[1] % block_size)) % block_size), (0, 0), (0, 0)))
  xv = xv.pad(((0, 0), (0, (block_size - (xv.shape[1] % block_size)) % block_size), (0, 0), (0, 0)))

  B, N, H, D = xq.shape
  H_KV = xk.shape[2]
  GROUP_SIZE = H // H_KV
  print(f"Flash Attention {B=} {N=} {H=} {D=} {H_KV=} {GROUP_SIZE=}")

  def custom_forward(ou:UOp, l_vecu:UOp, qu:UOp, ku:UOp, vu:UOp, mu:UOp) -> UOp:
    with Kernel("fa_custom_forward", (H, N // (Q_BLOCK_SIZE*NUM_WORKERS), B), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      o, q, k, v, mask, l_vec = GL(ou, ker), GL(qu, ker), GL(ku, ker), GL(vu, ker), GL(mu, ker), GL(l_vecu, ker)

      head = ker.blockIdx_x
      head_kv = head // GROUP_SIZE
      batch = ker.blockIdx_z
      q_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      k_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      v_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)

      q_reg_fl = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_transposed = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)
      o_reg = ker.rt((D, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      o_reg_transposed = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      mask_reg = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.float32)
      mask_reg_transposed = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      max_vec_last = ker.rv(KV_BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(KV_BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(KV_BLOCK_SIZE, dtypes.float32)
      scale_vec = ker.rv(KV_BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)
      o_reg = warp.zero(o_reg)
      scale_vec = warp.ones(scale_vec)

      # load q tile
      # Fixing indexing: (batch, head, q_seq, 0) for (B, H, N, D) layout
      q_reg_fl = warp.load(q_reg_fl, q, (), (batch, head, q_seq, 0), axis=2)
      q_reg_fl *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      q_reg = warp.copy(q_reg, q_reg_fl)
      q_reg_transposed = warp.transpose(q_reg_transposed, q_reg)

      for kv_idx in ker.range(N // KV_BLOCK_SIZE):
        k_smem = warp.load(k_smem, k, (), (batch, head_kv, kv_idx, 0), axis=2)
        v_smem = warp.load(v_smem, v, (), (batch, head_kv, kv_idx, 0), axis=2)

        k_reg = warp.load(k_reg, k_smem)
        v_reg = warp.load(v_reg, v_smem)

        # mma qk^t
        att_block = warp.zero(att_block.after(kv_idx))
        k_reg_transposed = warp.transpose(k_reg_transposed, k_reg)
        att_block = warp.mma_AtB(att_block, k_reg_transposed, q_reg_transposed)

        # apply attention mask
        mask_reg = warp.load(mask_reg, mask, (), (batch, 0, q_seq, kv_idx), axis=2)
        mask_reg_transposed = warp.transpose(mask_reg_transposed, mask_reg)
        att_block += mask_reg_transposed

        # softmax
        max_vec_last = warp.copy(max_vec_last.after(kv_idx), max_vec)
        max_vec = warp.row_reduce(max_vec.after(max_vec_last), att_block, lambda a, b: a.maximum(b), init_value=-math.inf)

        scale_vec = warp.map(scale_vec.after(max_vec_last, max_vec), lambda _, idx: max_vec_last[*idx] - max_vec[*idx])
        scale_vec = scale_vec.exp2()

        o_reg *= scale_vec
        norm_vec *= scale_vec

        att_block -= max_vec
        att_block = att_block.exp2()

        norm_vec = warp.row_reduce(norm_vec.after(scale_vec), att_block, lambda a, b: a + b)

        # mma av
        att_block_mma = warp.copy(att_block_mma.after(kv_idx, norm_vec), att_block)
        o_reg = warp.mma_AtB(o_reg, v_reg, att_block_mma)
      o_reg = ker.endrange()
      norm_vec = norm_vec.after(o_reg)
      max_vec = max_vec.after(o_reg)

      o_reg /= norm_vec

      o_reg_transposed = warp.transpose(o_reg_transposed, o_reg)
      o = warp.store(o, o_reg_transposed, (batch, head, q_seq, 0), (), axis=2)

      norm_vec = norm_vec.after(o)
      max_vec = max_vec.after(o)

      max_vec *= math.log(2)
      norm_vec = norm_vec.log2() * math.log(2)
      norm_vec += max_vec
      l_vec = warp.store(l_vec, norm_vec, (batch, head, 0, q_seq), (), axis=3)
      o = o.after(l_vec)

      return ker.finish()

  def custom_backward_q(out_qu:UOp, gradu:UOp, qu:UOp, ku:UOp, vu:UOp, l_vecu:UOp, delta_vecu:UOp, mu:UOp) -> UOp:
    with Kernel("fa_custom_backward_q", (H, N // (Q_BLOCK_SIZE*NUM_WORKERS), B), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      dq, grad, q, k, v, l_vec, delta_vec, mask = GL(out_qu, ker), GL(gradu, ker), GL(qu, ker), GL(ku, ker), GL(vu, ker), GL(l_vecu, ker), GL(delta_vecu, ker), GL(mu, ker)

      head = ker.blockIdx_x
      head_kv = head // GROUP_SIZE
      batch = ker.blockIdx_z
      q_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      q_reg_fl = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      do_reg_fl = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      do_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)

      k_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      v_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_transposed = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)

      dq_reg = ker.rt((D, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      dq_reg_transposed = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)

      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      dp_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      ds_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      ds_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      mask_reg = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.float32)
      mask_reg_transposed = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      l_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      delta_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)

      dq_reg = warp.zero(dq_reg)

      # Load L and Delta
      l_vec_reg = warp.load(l_vec_reg, l_vec, (), (batch, head, 0, q_seq), axis=3)
      delta_vec_reg = warp.load(delta_vec_reg, delta_vec, (), (batch, head, 0, q_seq), axis=3)

      # Load Q and Scale
      q_reg_fl = warp.load(q_reg_fl, q, (), (batch, head, q_seq, 0), axis=2)
      q_reg_fl *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      q_reg = warp.copy(q_reg, q_reg_fl)
      q_reg_transposed = warp.transpose(q_reg_transposed, q_reg)

      # Load dO
      do_reg_fl = warp.load(do_reg_fl, grad, (), (batch, head, q_seq, 0), axis=2)
      do_reg = warp.copy(do_reg, do_reg_fl)

      for kv_idx in ker.range(N // KV_BLOCK_SIZE):
        k_smem = warp.load(k_smem, k, (), (batch, head_kv, kv_idx, 0), axis=2)
        v_smem = warp.load(v_smem, v, (), (batch, head_kv, kv_idx, 0), axis=2)

        k_reg = warp.load(k_reg, k_smem)
        v_reg = warp.load(v_reg, v_smem)

        # S = Q @ K^T
        att_block = warp.zero(att_block.after(kv_idx))
        k_reg_transposed = warp.transpose(k_reg_transposed, k_reg)
        att_block = warp.mma_AtB(att_block, k_reg_transposed, q_reg_transposed)

        # Mask
        if is_causal:
          # idx[0] is row (KV), idx[1] is col (Q).
          att_block = warp.map(att_block, lambda x, idx: UOp.alu(Ops.WHERE,
            UOp.alu(Ops.CMPLT, q_seq * Q_BLOCK_SIZE + idx[1], kv_idx * KV_BLOCK_SIZE + idx[0]),
            UOp.const(dtypes.float32, -float('inf')), x))
        else:
          mask_reg = warp.load(mask_reg, mask, (), (batch, 0, q_seq, kv_idx), axis=2)
          mask_reg_transposed = warp.transpose(mask_reg_transposed, mask_reg)
          att_block += mask_reg_transposed

        # P = exp2(S - L)
        l_vec_reg_scaled = l_vec_reg * (1.0 / math.log(2))
        att_block -= l_vec_reg_scaled
        att_block = att_block.exp2()

        # dP = dO @ V^T
        v_reg_transposed_local = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
        do_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

        v_reg_transposed_local = warp.transpose(v_reg_transposed_local, v_reg)
        do_reg_transposed = warp.transpose(do_reg_transposed, do_reg)

        dp_block = warp.zero(dp_block)
        dp_block = warp.mma_AtB(dp_block, v_reg_transposed_local, do_reg_transposed)

        # dS = P * (dP - Delta)
        dp_block -= delta_vec_reg
        ds_block = warp.copy(ds_block, att_block)
        ds_block *= dp_block

        # dQ += dS @ K
        ds_block_mma = warp.copy(ds_block_mma, ds_block)
        dq_reg = warp.mma_AtB(dq_reg, k_reg, ds_block_mma)

      dq_reg = ker.endrange()

      scale = (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      dq_reg *= scale

      dq_reg_transposed = warp.transpose(dq_reg_transposed, dq_reg)
      dq = warp.store(dq, dq_reg_transposed, (batch, head, q_seq, 0), (), axis=2)

      return ker.finish()

  def custom_backward_kv(out_ku:UOp, out_vu:UOp, gradu:UOp, qu:UOp, ku:UOp, vu:UOp, l_vecu:UOp, delta_vecu:UOp, mu:UOp) -> UOp:
    # Grid X is H_KV, not H. We loop over groups inside.
    with Kernel("fa_custom_backward_kv", (H_KV, N // (KV_BLOCK_SIZE*NUM_WORKERS), B), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      dk, dv, grad, q, k, v, l_vec, delta_vec, mask = GL(out_ku, ker), GL(out_vu, ker), GL(gradu, ker), GL(qu, ker), GL(ku, ker), GL(vu, ker), GL(l_vecu, ker), GL(delta_vecu, ker), GL(mu, ker)

      head_kv = ker.blockIdx_x
      batch = ker.blockIdx_z
      kv_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      dk_reg = ker.rt((D, KV_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      dk_reg_transposed = ker.rt((KV_BLOCK_SIZE, D), dtypes.float32)
      dv_reg = ker.rt((D, KV_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      dv_reg_transposed = ker.rt((KV_BLOCK_SIZE, D), dtypes.float32)

      q_smem = ker.st((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      do_smem = ker.st((Q_BLOCK_SIZE, D), dtypes.bfloat16)

      k_reg_fl = ker.rt((KV_BLOCK_SIZE, D), dtypes.float32)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_transposed = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      v_reg_fl = ker.rt((KV_BLOCK_SIZE, D), dtypes.float32)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      v_reg_transposed = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      do_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      do_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      dp_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      ds_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      ds_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      mask_reg = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.float32)
      mask_reg_transposed = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      l_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      delta_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)

      dk_reg = warp.zero(dk_reg)
      dv_reg = warp.zero(dv_reg)

      # Load K, V
      k_reg_fl = warp.load(k_reg_fl, k, (), (batch, head_kv, kv_seq, 0), axis=2)
      k_reg = warp.copy(k_reg, k_reg_fl)
      k_reg_transposed = warp.transpose(k_reg_transposed, k_reg)

      v_reg_fl = warp.load(v_reg_fl, v, (), (batch, head_kv, kv_seq, 0), axis=2)
      v_reg = warp.copy(v_reg, v_reg_fl)
      v_reg_transposed = warp.transpose(v_reg_transposed, v_reg)

      # Loop over Query Heads in the Group
      for g in range(GROUP_SIZE):
        head = head_kv * GROUP_SIZE + g

        for q_idx in ker.range(N // Q_BLOCK_SIZE):
          q_smem = warp.load(q_smem, q, (), (batch, head, q_idx, 0), axis=2)
          do_smem = warp.load(do_smem, grad, (), (batch, head, q_idx, 0), axis=2)

          q_reg = warp.load(q_reg, q_smem)
          do_reg = warp.load(do_reg, do_smem)

          # Scale Q
          q_reg *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))

          # Load L, Delta
          l_vec_reg = warp.load(l_vec_reg, l_vec, (), (batch, head, 0, q_idx), axis=3)
          delta_vec_reg = warp.load(delta_vec_reg, delta_vec, (), (batch, head, 0, q_idx), axis=3)

          # S = Q @ K^T
          q_reg_transposed = warp.transpose(q_reg_transposed, q_reg)
          att_block = warp.zero(att_block.after(q_idx))
          att_block = warp.mma_AtB(att_block, k_reg_transposed, q_reg_transposed)

          # Mask
          if is_causal:
            att_block = warp.map(att_block, lambda x, idx: UOp.alu(Ops.WHERE,
              UOp.alu(Ops.CMPLT, q_idx * Q_BLOCK_SIZE + idx[1], kv_seq * KV_BLOCK_SIZE + idx[0]),
              UOp.const(dtypes.float32, -float('inf')), x))
          else:
            mask_reg = warp.load(mask_reg, mask, (), (batch, 0, q_idx, kv_seq), axis=2)
            mask_reg_transposed = warp.transpose(mask_reg_transposed, mask_reg)
            att_block += mask_reg_transposed

          # P = exp2(S - L)
          l_vec_reg_scaled = l_vec_reg * (1.0 / math.log(2))
          att_block -= l_vec_reg_scaled
          att_block = att_block.exp2()

          # dP = dO @ V^T
          do_reg_transposed = warp.transpose(do_reg_transposed, do_reg)
          dp_block = warp.zero(dp_block)
          dp_block = warp.mma_AtB(dp_block, v_reg_transposed, do_reg_transposed)

          # dS = P * (dP - Delta)
          dp_block -= delta_vec_reg
          ds_block = warp.copy(ds_block, att_block)
          ds_block *= dp_block

          # dV += P^T @ dO
          att_block_transposed = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
          att_block_mma = warp.copy(att_block_mma, att_block)
          att_block_transposed = warp.transpose(att_block_transposed, att_block_mma)
          dv_reg = warp.mma_AtB(dv_reg, do_reg, att_block_transposed) # Accumulates into dv_reg

          # dK += dS^T @ Q
          ds_block_transposed = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
          ds_block_mma = warp.copy(ds_block_mma, ds_block)
          ds_block_transposed = warp.transpose(ds_block_transposed, ds_block_mma)

          dk_reg = warp.mma_AtB(dk_reg, q_reg, ds_block_transposed) # Accumulates into dk_reg

      dk_reg = ker.endrange()

      # Scale dK
      scale = (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      dk_reg *= scale

      dk_reg_transposed = warp.transpose(dk_reg_transposed, dk_reg)
      dk = warp.store(dk, dk_reg_transposed, (batch, head_kv, kv_seq, 0), (), axis=2)

      dv_reg_transposed = warp.transpose(dv_reg_transposed, dv_reg)
      dv = warp.store(dv, dv_reg_transposed, (batch, head_kv, kv_seq, 0), (), axis=2)

      return ker.finish()

  def custom_backward_q_wrapper(out_qu:UOp, gradu:UOp, qu:UOp, ku:UOp, vu:UOp, l_vecu:UOp, delta_vecu:UOp, mu:UOp) -> UOp:
    return custom_backward_q(out_qu, gradu, qu, ku, vu, l_vecu, delta_vecu, mu)

  def custom_backward_kv_wrapper(out_ku:UOp, out_vu:UOp, gradu:UOp, qu:UOp, ku:UOp, vu:UOp, l_vecu:UOp, delta_vecu:UOp, mu:UOp) -> UOp:
    return custom_backward_kv(out_ku, out_vu, gradu, qu, ku, vu, l_vecu, delta_vecu, mu)

  if is_causal:
    if attn_mask is not None: raise RuntimeError("cannot set attn_mask when is_causal=True")
    attn_mask = Tensor.ones((B, 1, N, N), requires_grad=False, device=xq.device, dtype=dtypes.bool).tril()
  if attn_mask is not None:
    if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
  else:
    attn_mask = Tensor.zeros((B, 1, N, N), requires_grad=False, device=xq.device, dtype=dtypes.float32)

  attn = Tensor.empty_like(xq)
  l_vec = Tensor.empty(B, H, 1, N, requires_grad=False, device=xq.device, dtype=dtypes.float32).detach()

  def grad(grad:UOp, kernel:UOp) -> tuple[None, None, UOp, UOp, UOp, None]:
    grad_q = Tensor.empty_like(q := Tensor(kernel.src[2]))
    grad_k = Tensor.empty_like(k := Tensor(kernel.src[3]))
    grad_v = Tensor.empty_like(v := Tensor(kernel.src[4]))
    mask = Tensor(kernel.src[5])

    # grad: (B, N_, H, D) -> (B, H, N_, D)
    g = Tensor(grad).transpose(1, 2)
    # Pad to (B, H, N, D)
    g_padded = g.pad(((0,0), (0,0), (0, N-N_), (0, D-D_)))

    # attn (variable) is (B, H, N_, D_) (sliced)
    # Compute delta on valid region
    delta_valid = (g * attn).sum(-1) # (B, H, N_)
    # Pad to (B, H, N)
    delta_vec = delta_valid.pad(((0,0), (0,0), (0, N-N_)))
    delta_vec = delta_vec.unsqueeze(-2).detach() # (B, H, 1, N)

    grad_q = Tensor.custom_kernel(grad_q, g_padded, q, k, v, l_vec, delta_vec, mask, fxn=custom_backward_q_wrapper)[0]
    grad_k, grad_v = Tensor.custom_kernel(grad_k, grad_v, g_padded, q, k, v, l_vec, delta_vec, mask, fxn=custom_backward_kv_wrapper)[:2]
    return (None, None, grad_q.uop, grad_k.uop, grad_v.uop, None)

  attn, l_vec = Tensor.custom_kernel(attn, l_vec, xq, xk, xv, attn_mask, fxn=custom_forward, grad_fxn=grad)[:2]
  attn = attn[:, :N_, :, :D_]

  return attn.transpose(1, 2).cast(odtype)
