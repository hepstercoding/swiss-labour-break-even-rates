# Model Notes

The current repo starts with a compact custom Kalman filter because it is easy to inspect and iterate on.

The intended IRISpie upgrade path is:

1. move the latent break-even rate into an explicit IRISpie state-space or structural model object
2. add additional observables such as wage growth, employment growth, and market tightness
3. use IRISpie chartpacks or reports for routine monitoring

A natural next measurement system is:

\[
\Delta v_t = \beta_t (u_t^\star - u_t) + \varepsilon^v_t
\]
\[
\Delta w_t = \gamma_t (u_t^\star - u_t) + \varepsilon^w_t
\]

with either random-walk or AR(1) laws of motion for `u_t^\star`, `\beta_t`, and `\gamma_t`.
