import unittest
import torch
from perm_hmm import util


class MyTestCase(unittest.TestCase):

    def test_first_nonzero(self):
        batch_shape = (5,)
        sample_shape = (100, 6, 7)
        foos = torch.distributions.Bernoulli(torch.rand(batch_shape)).sample(sample_shape).bool()
        for foo in foos:
            with self.subTest(foo=foo):
                bar = torch.full(foo.shape[:-1], -1, dtype=int)
                for i in range(foo.shape[0]):
                    for j in range(foo.shape[1]):
                        for k in range(foo.shape[2]):
                            if foo[i, j, k]:
                                bar[i, j] = k
                                break
                baz = util.first_nonzero(foo)
                self.assertTrue((bar == baz).all())
                bar = torch.full(foo.shape[:1] + foo.shape[2:], -1, dtype=int)
                for i in range(foo.shape[0]):
                    for k in range(foo.shape[2]):
                        for j in range(foo.shape[1]):
                            if foo[i, j, k]:
                                bar[i, k] = j
                                break
                baz = util.first_nonzero(foo, dim=-2)
                self.assertTrue((bar == baz).all())
                bar = torch.full(foo.shape[:0] + foo.shape[1:], -1, dtype=int)
                for j in range(foo.shape[1]):
                    for k in range(foo.shape[2]):
                        for i in range(foo.shape[0]):
                            if foo[i, j, k]:
                                bar[j, k] = i
                                break
                baz = util.first_nonzero(foo, dim=-3)
                self.assertTrue((bar == baz).all())
        foo = torch.distributions.Bernoulli(torch.rand(batch_shape)).sample(()).bool()
        bar = torch.full((), -1, dtype=int)
        for i in range(foo.shape[-1]):
            if foo[i]:
                bar = i
                break
        baz = util.first_nonzero(foo)
        self.assertTrue(bar == baz)
        foo = torch.distributions.Bernoulli(torch.rand((1,))).sample(()).bool()
        bar = torch.full((), -1, dtype=int)
        for i in range(foo.shape[-1]):
            if foo[i]:
                bar = i
                break
        baz = util.first_nonzero(foo)
        self.assertTrue(bar == baz)
        foo = torch.distributions.Bernoulli(torch.rand(())).sample(()).bool()
        bar = torch.full((), -1, dtype=int)
        if foo:
            bar = i
        baz = util.first_nonzero(foo)
        self.assertTrue(bar == baz)
