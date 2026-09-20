# EXP-V2-04 boundary sweep (derived product)

`EXP-V2-04-boundary-sweep.csv` is **derived**, not recomputed: the five healthy-phase rules and
their per-rule SD(k_max) are transcribed by script from the table in
`EXP-V2-04-xjtu-pronostia.md` Section 3, which is the only archived product that carries the rule
labels. The archived CSVs (`EXP-V2-04-summary.csv`, `-main_raw.csv`, `-sensitivity_raw.csv`) do not
have a rule column, so a reader cannot recompute these values from them.

Consequences for the paper: Section 3.3's 2.40 pp (PRONOSTIA) and 5.19 pp (XJTU-SY) should be cited
as deriving from this table; a future re-run should write the rule label into the summary CSV so
that the sweep becomes recomputable end to end.
