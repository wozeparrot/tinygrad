# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, Iterable, cast
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, create_lt_node, create_ge_node, sint
from tinygrad.shape.view import View

def _expr_view(view:View, idxs:List[Node], valid:Optional[Node]=None) -> Tuple[Node, Node]:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  iexpr: List[Node] = [NumNode(view.offset) if isinstance(view.offset, int) else view.offset]
  vexpr: List[Node] = [valid] if valid is not None else []
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr.append(idx*st)
    if m is not None: vexpr += [create_ge_node(idx, m[0]), create_lt_node(idx, m[1])]  # idx >= m[0], idx < m[1]
  return Node.sum(iexpr), Node.ands(vexpr)

@dataclass(frozen=True)
class ShapeTracker:
  views: Tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[ShapeTracker]:
    ret = tuple(v.invert(s) for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]))
    return ShapeTracker(cast(Tuple[View, ...], ret)).reshape(out_shape) if all(x is not None for x in ret) else None

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]): return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def real_size(self) -> int:
    if 0 in self.shape: return 0
    idx, valid = self.expr_idxs()
    if not valid: return 0
    # TODO: it's possible that the real_size is smaller condition on valid being true
    ret = idx.max
    if not isinstance(ret, int): ret = ret.max  # might be represent by symbolic shape, take one more max for int max
    assert isinstance(ret, int), f"ret must be integer, {ret=} isn't"
    return ret+1

  def vars(self) -> Set[Variable]: return set.union(*[v.vars() for v in self.views], set())

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    bad_idx_vars: Set[Variable] = set()
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      try: ret[idxs.index(idx_maybe)] = cast(sint, stride_maybe)
      except ValueError: bad_idx_vars = bad_idx_vars.union(idx_maybe.vars())
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in bad_idx_vars or (tidx in valid_vars and not ignore_valid): ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def expr_idxs(self, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)] if idxs is None else list(idxs)
    idx, valid = _expr_view(self.views[-1], idxs)
    for view in reversed(self.views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      view = view.minify()
      acc, idxs = 1, []
      for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = _expr_view(view, idxs[::-1], valid)
    return idx, valid

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := self.views[-2] + self.views[-1]) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))
