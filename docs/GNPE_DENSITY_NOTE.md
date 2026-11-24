# GNPE density and the gated-likelihood weights

`import dingo` in this environment resolves to `DINGO-BNS/dingo/dingo/__init__.py`. That checkout is git `fede5c01524f3e205acf5750c0a0f101ff17e331` (2024-10-22, `Remove unused line.`). The line numbers below are that tree. The driver lines are this repository.

## 1. What `examples/gated_is_stage2.py::draw` calls

```215:221:examples/gated_is_stage2.py
def draw(c: Ctx, data: dict, n: int):
    from official_control import run_importance_sampling

    res = c.session.sample_result(data, c.settings, c.fixed, num_samples=n, batch_size=c.args.batch_size)
    raw = res.samples.copy()
    run_importance_sampling(res, num_processes=int(c.args.num_processes))
    return res, raw
```

`c.session` is `pe_session.PESession`. `sample_result` builds `dingo.core.models.PosteriorModel`, `dingo.core.samplers.FixedInitSampler` and `dingo.gw.inference.gw_samplers.GWSamplerGNPE`, then calls `run_sampler` and `to_result`:

```43:52:examples/pe_session.py
        sampler = GWSamplerGNPE(
            model=self.model,
            init_sampler=FixedInitSampler(fixed, log_prob=0.0),
            num_iterations=1,
            fixed_context_parameters=fixed,
        )
        sampler.context = event_data
        sampler.event_metadata = settings
        sampler.run_sampler(num_samples=int(num_samples), batch_size=int(batch_size))
        return sampler.to_result()
```

`to_result` is `dingo.core.samplers.Sampler.to_result` (`DINGO-BNS/dingo/dingo/core/samplers.py` lines 307–320). `GWSamplerGNPE` sets `_result_class` to `dingo.gw.result.Result` (`DINGO-BNS/dingo/dingo/gw/inference/gw_samplers.py` line 53).

`run_importance_sampling` (`examples/official_control.py` lines 519–553) calls two methods on that result:

1. `result.sample_synthetic_phase(...)` (`dingo.gw.result.Result.sample_synthetic_phase`, `DINGO-BNS/dingo/dingo/gw/result.py` line 358).
2. `result.importance_sample(...)` (`dingo.core.result.Result.importance_sample`, `DINGO-BNS/dingo/dingo/core/result.py` line 209).

The synthetic-phase kwargs set `compute_likelihood=True`, `uniform_weight=0`, `n_grid=1000`. The importance-sampling kwargs are `decimate=True` and `phase_heterodyning=True`.

## 2. Where `log_prob` comes from under GNPE

Inside one GNPE iteration the network returns a sample and a log density together:

```543:549:DINGO-BNS/dingo/dingo/core/samplers.py
                y, log_prob = self.model.model.sample_and_log_prob(
                    x["data"], x["context_parameters"]
                )
            time_sample_end = time.time()

            x["parameters"] = y
            x["log_prob"] = log_prob
```

There is no second, unconditional density estimator on this path. `FixedInitSampler` is constructed with `log_prob=0.0` (`examples/pe_session.py` line 44). When the initialiser already contains the proxy names, that constant is what gets added:

```578:584:DINGO-BNS/dingo/dingo/core/samplers.py
        if start_with_proxies and self.num_iterations == 1:
            # In this case it makes sense to save the log_prob and the proxy parameters.

            samples = x["parameters"]
            samples["log_prob"] = x["log_prob"] + proxy_log_prob.to(
                x["log_prob"].device
            )
```

`proxy_log_prob` was set from `init_samples["log_prob"]` (lines 477–479 of the same file). For this driver that value is 0.

DINGO's own comment says the saved `log_prob` is not the density of θ alone:

```586:595:DINGO-BNS/dingo/dingo/core/samplers.py
            # The log_prob returned by gnpe is not just the log_prob over parameters
            # theta, but instead the log_prob in the *joint* space q(theta,theta^|x),
            # including the proxies theta^. For importance sampling this means,
            # that the target density is
            #
            #       p(theta,theta^|x) = p(theta^|theta) * p(theta|x).
            #
            # We compute log[p(theta^|theta)] below and store it as
            # samples["delta_log_prob_target"], such that for importance sampling we
            # only need to evaluate log[p(theta|x)] and add this correction.
```

A further term is added later. `sample_synthetic_phase` samples a phase and does `self.samples["log_prob"] += delta_log_prob_array` (`DINGO-BNS/dingo/dingo/gw/result.py` line 541). After that call, the stored `log_prob` is the network joint density plus the synthetic-phase conditional, plus the initialiser's `proxy_log_prob` (0 in this driver).

## 3. `delta_log_prob_target`

It is the GNPE kernel term `log p(θ̂ | θ)`, not the chirp-mass prior, not the synthetic phase, and not time marginalisation. The base class stores it at `DINGO-BNS/dingo/dingo/core/samplers.py` lines 600–601:

```python
kernel_log_prob = self._kernel_log_prob(all_params)
samples["delta_log_prob_target"] = torch.Tensor(kernel_log_prob)
```

The GW subclass does not implement that kernel for chirp mass. With only fixed proxies it returns zeros:

```468:478:DINGO-BNS/dingo/dingo/gw/inference/gw_samplers.py
    def _kernel_log_prob(self, samples):
        # TODO: Reimplement as a method of GNPEBase.
        if len({"chirp_mass", "mass_ratio", "phase"} & self.gnpe_kernel.keys()) > 0:
            raise NotImplementedError("kernel log_prob only implemented for time gnpe.")
        gnpe_proxies_diff = {
            k: np.array(samples[k] - samples[f"{k}_proxy"])
            for k in self.gnpe_kernel.keys()
        }
        # When using only fixed proxies (i.e., no GNPE iterations), return log probs = 0.
        if not gnpe_proxies_diff:
            return np.zeros(len(list(samples.values())[0]))
        return self.gnpe_kernel.ln_prob(gnpe_proxies_diff, axis=0)
```

This checkpoint's kernel is empty at runtime. The Stage 2 and Stage 3 logs print `GNPE kernel: {}` and `GNPE parameters fixed (not iterated): ['chirp_mass']`. The `if not gnpe_proxies_diff` branch applies, so the stored `delta_log_prob_target` column is zeros. The chirp-mass kernel written in the checkpoint metadata (`Uniform(minimum=-0.005, maximum=0.005)`, see `results/revision/FACTS.md` F5) is not evaluated here, because a non-empty chirp-mass kernel would hit the `NotImplementedError` above.

DINGO folds the column into the weights in `_calculate_evidence`:

```346:360:DINGO-BNS/dingo/dingo/core/result.py
            log_weights = (
                log_prior
                + np.nan_to_num(log_likelihood)  # NaN = no log_likelihood evaluation
                + delta_log_prob_target
                - np.nan_to_num(
                    log_prob_proposal
                )  # NaN = outside prior so no synthetic
                # phase
            )
            self.log_evidence = logsumexp(log_weights) - np.log(self.num_samples)

            # Save the *normalized* weights.
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.mean(weights)
            self.samples["weights"] = weights
```

`log_prior` is `self.prior.ln_prob(theta)` (`importance_sample`, line 270). `log_likelihood` is either a fresh `likelihood.log_likelihood_multi` or, if `_log_likelihood` is already present, a copy of that column (lines 279–289). This driver's synthetic-phase call stores `_log_likelihood` (`gw/result.py` lines 523–533), so the likelihood that enters DINGO's weights is the synthetic-phase likelihood, not a second uniform-grid evaluation.

## 4. Phase

The network's `inference_parameters` do not include `phase` (checkpoint metadata, F5). Phase is reconstructed by `sample_synthetic_phase`, not sampled by the flow. The docstring says so (`DINGO-BNS/dingo/dingo/gw/result.py` lines 364–368): a post-processing step that adds a phase from likelihood evaluations on a phase grid.

With `approximation_22_mode=True` the grid likelihood is `outer(d_inner_h_complex, exp(2i phase))` plus the noise and optimal-SNR terms (lines 472–487). A phase is then drawn from the interpolated conditional, and its log density is added to `log_prob` (lines 514–541). Because `compute_likelihood=True` and `uniform_weight=0`, the same call also writes `_log_likelihood` (lines 523–533), which `importance_sample` reuses as `log_likelihood`.

## 5. Chirp-mass proxy, not an iterated chirp-mass GNPE

The checkpoint lists `chirp_mass` as a GNPE parameter and also lists `chirp_mass_proxy` as a context parameter (F5). This driver passes the proxy in `fixed_context_parameters` and sets `num_iterations=1`. DINGO treats that combination as prior conditioning, and it heterodynes once, outside the GNPE loop:

```346:353:DINGO-BNS/dingo/dingo/gw/inference/gw_samplers.py
                if "chirp_mass_proxy" in self.fixed_context_parameters:
                    # If we use a fixed chirp mass proxy, we can heterodyne the data in
                    # self.transform_pre (as opposed to transform_gnpe_loop_pre in the
                    # gnpe loop), which is much more efficient, as we only need to
                    # carry around the decimated strain. Fixed chirp mass proxies can
                    # e.g. be used when running single-iteration gnpe for the chirp
                    # mass as a form of prior conditioning. This is possible when the
                    # chirp mass posterior is tighter than the corresponding gnpe kernel.
```

The same branch then inserts `HeterodynePhase` into `transform_pre` (lines 363–371 of that file) and does not run a chirp-mass GNPE iteration.

The network is still conditioned on the fixed proxy. There is no separate unconditional density model for it. Importance sampling does not add a chirp-mass-proxy prior term beyond `delta_log_prob_target`, which section 3 shows is zero on this path. `official_control.run_importance_sampling` also snaps `chirp_mass_proxy` to a constant before heterodyning, because a float32 column can make `np.std` look non-zero (`examples/official_control.py` lines 530–537). DINGO refuses heterodyning unless that column is constant (`DINGO-BNS/dingo/dingo/gw/result.py` lines 324–334).

## 6. `adapt.gated_likelihood.reweight` against DINGO's formula

```275:290:src/adapt/gated_likelihood.py
    for col in ("log_prob", "log_prior"):
        if col not in s.columns:
            raise KeyError(f"samples lack {col}; run DINGO importance sampling first")
    log_w = s["log_prior"].to_numpy(float) + np.asarray(log_likelihood_new, float) - s["log_prob"].to_numpy(float)
    if "delta_log_prob_target" in s.columns:
        log_w = log_w + s["delta_log_prob_target"].to_numpy(float)
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    n = log_w.size
    if not np.any(np.isfinite(log_w)):
        return {"weights": np.zeros(n), "n_eff": 0.0, "efficiency": 0.0, "log_evidence": float("-inf")}
    m = np.max(log_w)
    w = np.exp(log_w - m)
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    log_z = float(m + np.log(np.mean(w)))
    w = w / np.mean(w)
    return {"weights": w, "n_eff": n_eff, "efficiency": n_eff / n, "log_evidence": log_z}
```

The four terms are the same ones as `_calculate_evidence`, and the likelihood is the only term the caller replaces. Two differences are in the code, not just in which likelihood array is passed:

1. DINGO applies `np.nan_to_num` to `log_likelihood` and to `log_prob`, so a NaN becomes 0 and still produces a finite weight. `reweight` sends any non-finite `log_w` to `-inf` and then to weight 0.
2. On this driver the `log_likelihood` inside DINGO's weights is `_log_likelihood` from the synthetic-phase grid (section 3). `reweight` does not read that column. It uses whatever array the caller passes. The gated-IS drivers pass `L0 + correction` from the uniform-grid `GatedLikelihood`, which is a different likelihood evaluation from the decimated, heterodyned synthetic-phase likelihood DINGO stored.

The normalisation, `exp(log_w - max) / mean`, matches DINGO's `weights /= np.mean(weights)`. The log-evidence expressions match when every weight is finite: DINGO's `logsumexp(log_weights) - log(N)` is the log of the mean of the un-normalised weights, and `reweight` returns `m + log(mean(exp(log_w - m)))`.
