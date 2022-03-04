r"""Computes optimal policies for a two state model with transition matrix
equal to identity.

Because the transition matrix is trivial, this reduces to an adaptive hypothesis
testing problem. To solve it, we observe that the number of possible belief
states is polynomial in the number of steps.

We use :math:`\Pr(E)` to denote the probability of event :math:`E` occurring, and will often
abbreviate :math:`\Pr(x) = \Pr(X=x)` when it is unambiguous. We use superscripts to
denote sequences of random variables, :math:`X^i = (X_0,\ldots,X_i)`, and of
sequences of outcomes, :math:`x^i = (x_0,\ldots,x_i)`.

Suppose that we have a two state source with no transitions, and two outcomes.
Label the outcomes
:math:`\left\{ 0, 1 \right\}` and the states :math:`\left\{ d, b \right\}`. We parameterize
the model as :math:`\Pr\left( Y=0|S=b \right) = q`, :math:`\Pr\left( Y=1|S=d \right) = p`.
WLOG, take
:math:`p+q \leq 1`, and :math:`q \geq p`. We write :math:`r_0 = \log(\frac{1-p}{q}), r_1
= \log(\frac{1-q}{p})`, so that :math:`r_1\ge r_0`.

Define :math:`\sigma_i` to be the permutation applied at time
:math:`i`, :math:`S_i` to be the state at time :math:`i`, before the
permutation :math:`\sigma_i` is applied, and :math:`Q_i` to be the state at time :math:`i`, after
the permutation is applied, so that :math:`S_{i+1} = Q_{i} = \sigma_{i}(S_i)`. Take
:math:`a_i = \Pr(s_i = b|y^{i-1},\sigma^{i-1}), a_i' = \Pr(q_i
= b|y^{i-1},\sigma^{i})`, :math:`x_i = \log(\frac{a_i}{1-a_i}),
x_i' = \log(\frac{a_i'}{1-a_i'}) = \pm x_i`, where we take the minus sign if we
apply the nontrivial permutation at time :math:`i`. With this parameterization,
Bayes' rule turns into :math:`x_{i+1} = x_i' \pm r_{1, 0}`, taking the plus sign and
the subscript 1 if :math:`y_{i+1} = 1`, and the minus sign and the subscript 0 if
:math:`y_{i+1} = 0`.

Define :math:`\hat{S}_0` to be the inferred initial state, with
:math:`\hat{S}_i = \sigma_{i-1}(\cdots(\sigma_1(\hat{S}_0))\cdots)`, the inferred state
at time :math:`i`, and
:math:`\hat{Q}_i` defined similarly. Our task is to maximize :math:`\Pr\left( \hat{S}_0
= S_0 \right)` over possible policies.
To do so, we use the
`Bellman equation`_, which tells us to work from the end by computing the optimal
policy at each possible prior when there is only one step left, then
propagating that solution backwards inductively.

This is to say,

.. math::

    \max_{\sigma^n}\Pr\left( \hat{S}_0 = S_0 | \sigma^n \right) &=
    \max_{\sigma^n}\mathbb{E}_{Y^n|\sigma^n}\left[\Pr\left( \hat{S}_n = S_n|Y^n \sigma^n
    \right)\right]\\
    &= \max_{\sigma_0}\mathbb{E}_{Y_0|\sigma^0}\left[
      \cdots\max_{\sigma_{n-1}}\mathbb{E}_{Y_{n-1}|\sigma^{n-1}}\left[\max_{\sigma_n}\mathbb{E}_{Y_n|\sigma^n}\left[\Pr\left(
        \hat{S}_n = S_n|Y^n \sigma^n
      \right)  \right]\right] \cdots\right]

Because all the state transitions come
from applying permutations, we have

.. math::

    \Pr\left( \hat{S}_0 = S_0|s_i, \sigma^{i} \right) = \Pr\left( \hat{S}_i = s_i|s_i \right).

Then compute that

.. math::

  &\Pr\left( \hat{S}_0=S_0|y^{n-2}, \sigma^{n-1}\right) = \sum_{s_{n-1}y_{n-1}}\Pr\left(
  \hat{S}_0=S_0|s_{n-1},y^{n-1},\sigma^{n-1}
  \right)\Pr\left( y_{n-1}|s_{n-1}, y^{n-2},\sigma^{n-1} \right)\Pr\left(
  s_{n-1}|y^{n-2},\sigma^{n-1} \right)\\
  &= \sum_{s_{n-1}y_{n-1}}\underbrace{\Pr\left(
    \hat{S}_{n}=S_{n}|s_{n}=(\sigma_{n-1}(s_{n-1})),y^{n-1},\sigma^{n-1}\right)}_{\text{Use previous solution to
    compute}}\Pr\left(
      y_{n-1}|s_{n-1},\sigma_{n-1}
  \right)\underbrace{\Pr(s_{n-1}|y^{n-2},\sigma^{n-2})}_{\text{known
  from induction hypothesis}}

In preparation to compute the above, define :math:`\chi_{n}^{s_{n}}(x_{n}) :=
\Pr(\hat{S}_{n}
= s_{n}|s_{n}, y^{n-1},\sigma^{n-1})` we can then evaluate that
:math:`\chi_{n}^b(x_{n}) = \chi_{n}^d(-x_{n})`.

Then write :math:`e` for the identity permutation, and :math:`\nu` for the
nontrivial permutation, and evaluate that

.. math::

    \Pr\left( \hat{S}_0=S_0| y^{n-2},\sigma^{n-2}, \sigma_{n-1} = e \right)(a_{n-1}) =~&\left(
    1-a_{n-1}
    \right)(1-p)\chi_{n}^d\left( x_{n-1}-r_0 \right) + \left( 1-a_{n-1}
    \right)p\chi_{n}^d\left( x_{n-1} + r_1 \right) + \\
    &a_{n-1}q \chi_{n}^{b}\left( x_{n-1}-r_0 \right) + a_{n-1}(1-q) \chi_{n}^b\left(
    x_{n-1}+ r_1 \right)

.. math::

    \Pr\left( \hat{S}_0=S_0| y^{n-2},\sigma^{n-2}, \sigma_{n-1} = \nu \right)(a_{n-1})
    &=
    \begin{aligned}
    &\left(
    1-a_{n-1}
    \right)q\chi_{n}^b\left( -x_{n-1}-r_0 \right) + \left( 1-a_{n-1}
    \right)(1-q)\chi_{n}^b\left( -x_{n-1} + r_1 \right) + \\
    &a_{n-1}(1-p) \chi_{n}^{d}\left( -x_{n-1}-r_0 \right) + a_{n-1}p \chi_{n}^d\left(
    -x_{n-1}+ r_1 \right)\\
    \end{aligned}\\
    &= \Pr\left( \hat{S}_0=S_0| y^{n-2}, \sigma_{n-1} = e \right)(1-a_{n-1})

.. _`Bellman equation`: https://en.wikipedia.org/wiki/Bellman_equation
"""