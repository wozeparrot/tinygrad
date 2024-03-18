import unittest
from tinygrad import Tensor
from tinygrad.realize import create_schedule, lower_schedule_item

# TODO: can copy this in here when we remove it
#from tinygrad.ops import get_lazyop_info
#info = get_lazyop_info(ast)
#print(ops, mem, expected_mem)
#print(info.flops, info.mem_estimate)

# **************** new FlopCounter ****************

def get_stats(x:Tensor):
  si = create_schedule([x.lazydata])[-1]
  runner = lower_schedule_item(si)
  return runner.op_estimate, runner.mem_estimate

class TestUOpsStats(unittest.TestCase):
  def test_simple_add(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = a+b
    ops, mem = get_stats(c)
    expected_ops = c.numel()
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  def test_simple_add_sq(self):
    a = Tensor.empty(100,100)
    b = Tensor.empty(100,100)
    c = (a+b)*(a+b)
    ops, mem = get_stats(c)
    expected_ops = c.numel()*2
    expected_mem = a.nbytes() + b.nbytes() + c.nbytes()
    self.assertEqual(mem, expected_mem)
    # NOTE; ops also include indexing ops
    assert expected_ops <= ops and ops <= expected_ops * 2

  def test_simple_matmul(self):
    a = Tensor.empty(1024,1024)
    b = Tensor.empty(1024,1024)
    c = a@b
    ops, mem = get_stats(c)
    expected_ops = c.numel() * 1024 * 2
    required_mem = a.nbytes() + b.nbytes() + c.nbytes()
    assert expected_ops <= ops and ops <= expected_ops * 1.2
    # NOTE: it's hard to assert on the memory here, all depends on caching
    assert required_mem <= mem

if __name__ == '__main__':
  unittest.main(verbosity=2)
