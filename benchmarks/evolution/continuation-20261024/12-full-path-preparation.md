# No long measured finish calls in the eight-warmup observation

The separately frozen preparation diagnostic completes correctly, with none of
the earlier roughly 200 ms finish intervals inside its measured batches. This
supports preparation sensitivity in this observation. It does not prove native
pool identities, a driver/JIT cause, repeatability or an engine speedup. The
original overlap experiment remains REJECT and removed from production.

The [specification](profiles/full-path-preparation-spec.json) changed only the
private warmup policy from two batches to eight, equally in both arms. Both
invocations retain mona, seed 7, two pairs, eight measured batches, the same
mutation parameters and shader bytes. Seeded reset keeps the initial and final
trajectories exactly equal. No further warmup variants or unmodified control
invocations were run.

| Timed row | Wall seconds | Prefix finish ms | Final finish ms |
| --- | ---: | ---: | ---: |
| First reference | 0.475833975 | 0 | 115.827127 |
| First candidate | 0.433303391 | 21.521038 | 74.383009 |
| Later candidate | 0.434639607 | 20.951128 | 74.634689 |
| Later reference | 0.479390770 | 0 | 121.167932 |

All 36 direct batch-finish intervals are below 100 ms, with a maximum of
22.752553 ms. The first/later candidate wall ratio is 0.996925692, compared with
2.110378860 in the earlier instrumented observation. These are single-invocation
diagnostic observations under different preparation policies, not a repeated
performance comparison. The [interactive chart](profiles/first-trial-diagnostic.html)
retains both policies and every timed row on a common scale.

## Preparation actually exercised the split

Each of the four normal preparation windows contains eight actual batch calls.
References have zero split submissions; both candidates have two. Both timed
candidates still split at batches 4 and 7. The first candidate starts timing
with six earlier split submissions in the evolver; the later candidate starts
with ten. In the two-warmup observation, candidate preparations had zero splits.

The new observation therefore exercises the branch before each candidate timer
and does not repeat the earlier measured finish spikes at batches 5, 7 and 8.
Warmup durations were not recorded. Startup cost remains unquantified, and native
pool reuse remains a source-supported hypothesis rather than an observed cause.
Sequential invocation order, global caches and background activity remain
possible confounds. No deployment or adoption decision follows from this check.

## Extra preparation work is separate

The number of timed evaluations is unchanged. Added warmup work is excluded
from the reported timing rates and retained explicitly:

| Work in the new invocation | Batches | Logical evaluations |
| --- | ---: | ---: |
| Four timed rows | 32 | 524,288 |
| Two separate diagnostics | 16 | 262,144 |
| Qualification | 8 | 131,072 |
| Seven preparations, eight warmups each | 56 | 917,504 |
| Total logical evolution work | 112 | 1,835,008 |

The final lifetime guard count is 112, versus 70 in the two-warmup observation.
Source inspection confirms one guard evaluation per logical batch, zero warmup
batches during precompilation, and no other reachable batch loops. The added
42 warmup batches account for 688,128 evaluations. Other setup, full-fitness and
reset GPU work exists separately and is not a logical evolution evaluation.
This is not a claim that total startup work became cheaper.

## Verification and reproducibility

Independent audit passes 535 checks over 847 pinned inputs. Four real sidecars
pass identity, metadata, counter, guard, lifetime, interval, target, staging and
completion checks. Six saved snapshots reproduce all seven canonical fields
against the earlier two-warmup observation and accepted reference. Twenty-four
RGB-byte L1/fitness checks pass under the unchanged 2e-5 policy. All timing,
diagnostic and qualification rows complete expected work with zero errors.
No scalar rerender was performed.

Direct spans, drain and explicit gaps close to the sidecar clocks. Original
elapsed capture precedes their end markers by 241, 240, 611 and 350 ns. Inclusive
readback/poll intervals remain inside their collection/drain parents; neither
these nor GPU intervals are additive CPU shares. Instrumentation overhead and
background isolation remain uncalibrated.

The private policy change shares one typed constant between the actual producer,
qualification specification, row/summary metadata and exporter validation. All
ten existing exporter tests and a new old-policy rejection test pass. The latter
retains three invalid exported cases with old two-warmup metadata, counters, or
both. Source review finds no material issue, and applied inverse reconstruction
restores all 88 files of the earlier observer exactly.

One locked release build uses optimization 3 and debug assertions false. Source
freeze is `0e1d7ecab839908e25d1fd45070abc89f927516ea61e84887d992d87b85a22bd`;
binary SHA256 is
`d2bd612a640987323647a127597d7b24b5052e504814a27328a9e2c134aa70e9`.
All accepted runtime sources and normal binaries, plus the earlier observer,
remain unchanged. The [policy patch](attempts/06-eight-warmup-diagnostic.patch)
applies after [observer 05](attempts/05-first-trial-observer.patch) in a disposable
source copy. A native Git apply check reproduces the rename and all postimages.

The [result](profiles/full-path-preparation-result.json) records the final parent
analysis and audit association. Native lab rates and automatic analysis fields
are retained because the observer uses the real timed path; they are not an
adoption gate for these externally unscored diagnostics. An initial parent copy
carried the previous interpretation label. Its numeric values were correct;
the corrected `analysis-v2.json` and an explicit version note preserve both
without changing raw sidecars.

Raw source, build, tests, review, work accounting and outputs remain under
`.cache/evolution-profiling-20261024/first-use-prepared-97/`,
`first-use-prepared-review-98/`, `first-use-prepared-audit-99/` and
`first-use-20261024/`. The chart was checked at desktop/mobile widths in dark and
light themes, including actual policy and trial selection events.

No engine change is adopted. A later throughput qualification needs a fresh,
declared preparation policy, A/A followed by alternating A/B repetitions, and
startup cost reported separately. Existing rejected statistics must remain
intact. Explain the Number keeps extra work and uncertainty visible; Prove It
Works requires actual preparation counters and exact trajectory replay.
