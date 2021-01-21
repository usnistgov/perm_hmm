import unittest
import perm_hmm.example_systems.beryllium as beryllium
import numpy as np
import scipy.stats
import itertools


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_adjugate(self):
        for n in range(2, 10):
            m = np.random.rand(n, n)
            adj = beryllium.adjugate(m)
            self.assertTrue(np.allclose(np.matmul(m, adj), np.linalg.det(m)*np.eye(n)))
            m = np.random.rand(1, n, n)
            adj = beryllium.adjugate(m)
            self.assertTrue(np.allclose(np.matmul(m, adj), np.linalg.det(m)[..., None, None]*np.eye(n)))
            m = np.random.rand(3, 4, n, n)
            adj = beryllium.adjugate(m)
            self.assertTrue(np.allclose(np.matmul(m, adj), np.linalg.det(m)[..., None, None]*np.eye(n)))

    def test_polymat(self):
        for n in range(2, 10):
            mlist = [np.random.rand(n, n), np.random.rand(1, n, n), np.random.rand(3, 4, n, n)]
            for m in mlist:
                with self.subTest(m=m):
                    m = np.random.rand(n, n)
                    ev = np.linalg.eigvals(m)
                    adj_resolvent = beryllium.polymat(m, ev)
                    self.assertTrue(
                        np.allclose(np.matmul(m-np.eye(n)*ev[..., :, None, None], adj_resolvent), 0)
                    )

    def test_akij(self):

        for n in range(2, 5):
            mlist = [np.random.rand(n, n), np.random.rand(1, n, n), np.random.rand(3, 4, n, n)]
            slist = [np.random.randn(), np.random.randn(1), np.random.randn(7, 6)]
            for m, s in itertools.product(mlist, slist):
                with self.subTest(m=m, s=s):
                    s = np.array(s)
                    pij = beryllium.polymat(m, s)
                    if s.shape == ():
                        d = np.linalg.det(m - np.eye(n)*s)
                    else:
                        d = np.linalg.det(m.reshape(m.shape[:-2]+(1,)*len(s.shape)+m.shape[-2:]) - np.eye(n)*s[..., None, None])
                    resolvent_1 = pij/d[..., None, None]
                    ev = np.linalg.eigvals(m)
                    akij_with_ev = beryllium.akij(m, ev)
                    akij_no_ev = beryllium.akij(m)
                    self.assertTrue(np.allclose(akij_with_ev, akij_no_ev))
                    if s.shape == ():
                        resolvent_2 = (akij_no_ev/(s-ev[..., :, None, None])).sum(-3)
                        resolvent_3 = np.linalg.inv(m-np.eye(n)*s)
                    else:
                        resolvent_2 = (akij_no_ev[(...,) + (None,)*len(s.shape) + (slice(None),)*3]/(s[..., None, None, None]-ev[(...,)+(None,)*len(s.shape)+ (slice(None), None, None)])).sum(-3)
                        resolvent_3 = np.linalg.inv(m[(...,) + (None,)*len(s.shape)+(slice(None),)*2]-np.eye(n)*s[..., None, None])
                    self.assertTrue(np.allclose(resolvent_1, resolvent_2))
                    self.assertTrue(np.allclose(resolvent_1, resolvent_3))
                    self.assertTrue(np.allclose(resolvent_3, resolvent_2))

    def test_transitons(self):
        times = np.exp(np.linspace(-8, -4, 10))
        for time in times:
            with self.subTest(time=time):
                old_pij = beryllium.old_parameter_matrices(time)[1]
                new_pij = beryllium.transition_matrix(time)
                self.assertTrue(np.all(np.isfinite(new_pij)))
                self.assertTrue(np.allclose(old_pij, new_pij))

    def test_output_dist(self):
        times = np.arange(-9, -5, 1)
        rev_times = np.flip(times)
        for time, rev_time in zip(times, rev_times):
            zero = np.array(0, dtype=int)
            n_array = np.arange(500)
            time = np.exp(time)
            with self.subTest(time=time, n_array=n_array):
                output_dist = beryllium.prob_of_n_photons(n_array, time)
                print(time, (output_dist*n_array[:, None]).sum(-2))
                self.assertTrue(np.all(output_dist >= 0.))
                self.assertTrue(np.allclose(output_dist.sum(-2), 1.))
                no_photons = beryllium.prob_of_n_photons(zero, time)
                print(time, "no photons", no_photons)


if __name__ == '__main__':
    unittest.main()
