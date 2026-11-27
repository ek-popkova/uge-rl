# PPO

Policy-based actor-critic agent: PPO-Clip and PPO-KL, written from scratch in PyTorch (no RL library). Owner: Ekaterina
Popkova (all PPO runs, tuning and sensitivity are trained on the PPO machine; see the root README, "Team").

## Files

| File | Contents |
|---|---|
| `core.py` | The pure math, no env and no networks: `next_state_values`, `compute_gae`, `normalize_advantages`, `probability_ratio`, `clipped_surrogate_loss`, `categorical_kl`, `kl_penalty_loss`, `update_kl_beta`, and the diagnostics `approx_kl`, `clip_fraction`, `ratio_max`, `explained_variance`. |
| `train.py` | `PPOTrainer` (the `common.training.Trainer` of PPO), `Rollout` (the preallocated buffer), weight initialization, and the CLI `python -m ppo.train`. |
| `agent.py` | `PPOAgent`: actor-only, picklable snapshot used for validation, testing, checkpoints (`final.pt`, milestones) and videos. |

Everything else (CLI, seeding, vector envs, logs, validation, checkpoints, resume, `run_meta.json`) is done by the shared
harness `common.training.TrainingRun`, identically for DDQN and PPO.

## Data flow of one update (`PPOTrainer.collect_and_update`)

1. **Rollout** (`collect_rollout`, under `torch.no_grad()`). `n_steps` vector-env steps with the current policy. Each step does
   one observation conversion `(n_envs, 25)`, one actor and one critic forward pass for all envs, and samples the actions
   with `torch.multinomial(probs, 1, generator=...)`. `ctx.step(actions)` then advances the envs. The step stores obs,
   actions, `log pi_old(a|s)`, the old logits, `V(s)`, the rewards, `terminated` and `done = terminated | truncated`. If an
   episode ended at that step, the critic also evaluates its `infos["final_obs"]` (SAME_STEP autoreset). The last step
   returns `V` of the observation after the rollout.
2. **Advantages** (inside `ctx.timers.update()`). `next_state_values` builds `V(s_{t+1})`: the next stored value, the
   value after the rollout at the last step, and `V(final_obs)` where the episode ended. `compute_gae` then gives the
   advantages and the lambda-returns. `explained_variance` is measured with the values from before the update.
3. **Epochs** (`run_epochs`). There are `n_epochs` passes over a fresh permutation of the `n_envs * n_steps` transitions,
   in minibatches of `minibatch_size`. For each minibatch, `gradient_step`:
   - normalizes the advantages;
   - computes the policy loss (`policy_loss = "clip"` or `"kl"`), the value loss and the entropy;
   - forms `loss = policy_loss - ent_coef * entropy + vf_coef * value_loss`;
   - runs backward, clips the gradient norm over both networks and does an Adam step.

   With `target_kl` set, the remaining epochs are skipped after an epoch whose mean `approx_kl` exceeds `target_kl`.
4. **KL after the update (both losses).** `mean_rollout_kl` gives the exact mean KL between the rollout policy and the
   updated policy over the whole rollout, after all epochs. It is logged as `kl_after_update`; for PPO-KL it also drives
   `update_kl_beta` (halve or double `kl_beta`).
5. **Logging.** One `ctx.log_update(...)` call (one `update_log.csv` row) per rollout, with
   `n_updates` = the Adam steps actually taken (`n_epochs_done x rollout_size / minibatch_size`) and `n_samples` = their
   summed minibatch sizes (`n_epochs_done x rollout_size`). `update_idx`, `updates_in_row` and `run_meta.json`
   `timing.n_updates` therefore count optimizer steps, the same unit as DDQN, and `timing.n_gradient_samples` the
   per-sample gradient evaluations (`gradient_updates_per_env_step`, `samples_per_env_step` in
   `common.aggregate.cost_per_model`). Metrics of the row:
   - `lr`;
   - the minibatch means of `loss, grad_norm, policy_loss, value_loss, entropy, approx_kl, clip_fraction`;
   - `explained_variance` and `n_epochs_done`;
   - `ratio_max`, the maximum over the minibatches;
   - `kl_beta`, the value after `update_kl_beta` (NaN for PPO-Clip);
   - `kl_after_update` (step 4).

   `approx_kl`, `clip_fraction` and `ratio_max` are computed from the ratios before each minibatch's Adam step, so they
   miss the change made by the last step; `kl_after_update` measures the whole change.

## Report equation -> code

| Report (`report/chapters/03_methods.tex`, section 3.3) | Code |
|---|---|
| `eq:gae`: `delta_t` with `m^term_{t+1}`, recursion with `m_{t+1}` | `core.compute_gae` (`terminated_t`, `done_t`), bootstrap values from `core.next_state_values` |
| lambda-returns `A_t + V(s_t)` as critic targets | `returns` of `core.compute_gae`, value loss in `PPOTrainer.gradient_step` |
| `eq:ppo-clip`: `rho_t` and `L^CLIP` | `core.probability_ratio`, `core.clipped_surrogate_loss` |
| PPO-KL: penalty `beta KL(pi_old || pi_theta)`, beta halved / doubled at `d_targ / 1.5`, `1.5 d_targ` | `core.categorical_kl`, `core.kl_penalty_loss`, `core.update_kl_beta`, `PPOTrainer.mean_rollout_kl` |
| entropy bonus | `Categorical(logits).entropy()` in `PPOTrainer.gradient_step` |
| "a collision ends the bootstrap and a time-limit truncation does not" (section `sec:termination`) | `terminated` vs `done` in `core.compute_gae`, `final_obs` values in `PPOTrainer.final_obs_values` |

## Code-level choices and their sources

| Choice | Value here | Source / reason |
|---|---|---|
| Separate actor and critic, 256 x 256 ReLU MLPs | decided (shared) | `common.networks`; Andrychowicz et al. 2021 found separate networks better in 4 of 5 environments |
| Orthogonal initialization, zero biases | gain sqrt(2) hidden, 0.01 actor output, 1.0 critic output | the gains are those of OpenAI baselines ppo2 (Engstrom et al. 2020 list "orthogonal initialization and layer scaling" as a code-level optimization but give no gains); the small policy head (near-uniform initial policy) follows Andrychowicz et al. 2021, section 3.2 ("100x smaller weights") |
| Seeding of the initialization and of the sampling | `torch.Generator`s derived from `(run seed, purpose)` with `numpy.random.SeedSequence` (`ppo.train.derived_generator`) | `common/seeding.py` rule 5; `common.seeding` has no generator helper |
| One Adam optimizer over both networks | `eps = 1e-5` (`ADAM_EPS`) | OpenAI baselines ppo2 (Engstrom et al. 2020 do not mention Adam's epsilon) |
| Learning rate | constant `lr`, no annealing | keeps the lr axis of the sensitivity analysis easy to interpret (Engstrom et al. 2020 list annealing as an optimization; deliberately not used) |
| Advantage normalization | per minibatch, `(A - mean) / (std + 1e-8)`, population std | OpenAI baselines ppo2; Andrychowicz et al. 2021 (section 3.3) found per-minibatch advantage normalization to have little effect on performance (not in Engstrom et al. 2020's list) |
| Advantages per rollout | computed once per rollout, before the first epoch, and kept fixed for all epochs | OpenAI baselines ppo2 / SB3. Andrychowicz et al. 2021 (section 3.5) recommend recomputing them once per pass over the data (stale advantages can hurt); not adopted, so the epochs optimize one fixed objective and the code matches the SB3 calibration |
| Value loss | `mean((V - returns)^2)`, no value clipping | Andrychowicz et al. 2021: PPO-style value clipping hurt |
| Gradient clipping | global norm over both networks, `max_grad_norm`; the logged `grad_norm` is the norm before clipping | baselines ppo2 ("global gradient clipping" is item 9 of Engstrom et al. 2020). With the placeholder `max_grad_norm = 0.5` every step is clipped: the pre-clip norms were 5-22 in a 2,000-step smoke run (rollout 500), with one norm over actor and critic together; with Adam this mostly rescales all steps uniformly. Recheck during tuning |
| `grad_norm` / `grad_norm_max` in `update_log.csv` | `grad_norm` = mean over the rollout's minibatch steps; one row per rollout | the logger derives `grad_norm_max` as the max over the calls of a row, and PPO makes one call per rollout, so **for PPO rows `grad_norm_max` equals `grad_norm`**: it does not show spikes inside a rollout (for DDQN, one call per gradient step, it does) |
| Reward scaling, observation normalization | none | shared decision: the env already normalizes the Kinematics features; no reward scaling, to keep the reward presets comparable |
| `approx_kl` | k3 estimator `mean((rho - 1) - log rho)` on the sampled actions | the estimator SB3 logs, so the values are comparable with the SB3 calibration (`docs/calibration.md`) |
| Target-KL early stop | checked after each epoch on the epoch's mean `approx_kl > target_kl` (`None` = off) | SB3 instead stops inside an epoch at `1.5 * target_kl`; the per-epoch rule is simpler and makes `n_epochs_done` a whole number |
| PPO-KL measured KL | exact categorical KL, mean over the whole rollout, after all epochs (`kl_after_update`, logged for PPO-Clip too) | PPO paper (Schulman et al. 2017, section 4); `kl_target = 0.01`, `kl_beta_init = 1.0` are the paper's values (decided, not tuned; `docs/config_decisions.md`) |
| `clip_fraction` for PPO-KL | computed with the configured `clip_range` | a diagnostic only, so PPO-Clip and PPO-KL logs are comparable |
| Minibatch size | must divide `n_envs * n_steps` (`PPOConfig` raises otherwise) | every minibatch has `minibatch_size` samples; a short last minibatch (in the extreme one sample, whose normalized advantage is 0) cannot occur |

**A2C-like runs need no extra code.** The registered supplementary run `supp-a2c` (`n_epochs=1`,
`minibatch_size = n_envs * n_steps`, `common.train_config.SUPPLEMENTARY_RUNS`) gives one gradient step per rollout at ratio 1, so
clipping is never active. That is the plain advantage actor-critic gradient (with GAE advantages,
an entropy bonus and advantage normalization).

**Timing.** The forward passes during collection are counted in neither `env_time_s` nor `update_time_s`. GAE and the
epochs are counted in `update_time_s`. One update with the default config (rollout 1,000, 10 epochs x 10 minibatches)
takes about 0.5 s on the development laptop, against 20-65 s of simulation per rollout.

## How to run

```bash
# main grid / sensitivity / tuning (300k steps: the estimate exceeds --max-minutes, so pass --force)
python -m ppo.train --reward balanced --seed 0 --workers 4 --force
python -m ppo.train --reward balanced --seed 0 --set clip_range=0.1 --force              # sensitivity variant
python -m ppo.train --variant supp-ppo-kl --reward balanced --seed 0 --set policy_loss=kl --force  # PPO-KL (registered)
python -m ppo.train --variant supp-a2c --reward balanced --seed 0 --set n_epochs=1 --set minibatch_size=1000 --force  # A2C-like
python -c "from common.train_config import expand_supplementary_runs as g; [print(' '.join(r.cli_args())) for r in g()]"
python -m ppo.train --reward balanced --seed 99 --variant tune-a --set n_epochs=5 --force # tuning run
python -m ppo.train ... --resume                                                         # continue after a crash

# smoke run (a few minutes, debug budget)
python -m ppo.train --reward balanced --seed 0 --set n_envs=2 --set n_steps=250 --set minibatch_size=100 \
  --total-steps 2000 --val-every 1000 --val-episodes 2 --checkpoint-every 1000 --benchmark-steps 0 \
  --out-dir results/_smoke --overwrite

# tests (never while a training run or benchmark is running)
python -m pytest tests/test_ppo_core.py tests/test_ppo.py    # fast: math, trainer and agent without the simulator
python -m pytest -m "not slow"                               # whole fast tier
python -m pytest                                             # full suite, includes the PPO harness runs
```

`n_envs * n_steps` must divide 300,000 and 12,000: any divisor of 12,000, e.g. 1,000 (the default 4 x 250), 500, 1,500,
2,000, 2,400, 3,000, 4,000, 6,000 or 12,000. `minibatch_size` must in turn divide `n_envs * n_steps`.
`check_budget_matches_rollout` names the nearest valid values. The hyperparameters still marked
`# PLACEHOLDER: decide before training` in `common/train_config.py` are chosen during tuning (`docs/tuning_log.md`).
