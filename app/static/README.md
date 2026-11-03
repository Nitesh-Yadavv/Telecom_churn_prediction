# Fonts

Self-hosted so the dashboard does not call any third-party font service.

| File | Font | Source |
|---|---|---|
| `SourceSerif4-SemiBold.woff2` | Source Serif 4, weight 600 (headings, big numbers) | github.com/google/fonts, `ofl/sourceserif4` |
| `Inter-Regular.woff2`, `Inter-Medium.woff2`, `Inter-SemiBold.woff2` | Inter, weights 400, 500, 600 (all other text) | github.com/google/fonts, `ofl/inter` |

Both are licensed under the SIL Open Font License 1.1 (see the `OFL-*.txt` files), which allows
redistribution. The files were cut from the variable fonts at fixed weights and trimmed to the Latin
characters and symbols the dashboard uses (including the rupee sign), which makes each about 18 KB.
