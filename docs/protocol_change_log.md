# Protocol Change Log

Every change after the freeze records: date, version, what changed, reason, affected experiments, reruns required.

| Date | Version | Change | Reason | Affected | Reruns |
|---|---|---|---|---|---|
| 2026-10-16 | 1.0.0 | Protocol frozen | — | — | — |
| 2026-10-16 | 1.0.0 (pre-freeze) | Effective-contrast secondary criterion made symmetric in log space | One-sided form mis-stated "factor of 2" | none (no runs yet) | none |
| 2026-10-16 | 1.0.0 (pre-freeze) | F4-full buffer determinant corrected from -(a e2+(1-a)e1) to +(a e2+(1-a)e1) | Sign error in hand derivation caught by symbolic check; solvability conclusion unchanged | none | none |

Pending decisions already scheduled by the protocol (not changes):
- Pilot learning-rate choice (Sec. 6.4) - to be recorded here after the pilot.
- Selection of the two strongest formulations for 2D (Sec. 8.2) - to be recorded before the 2D sweep.
- Detailed configs for P2-P5 - added as protocol_v1.x before each phase, without altering P1 settings.
