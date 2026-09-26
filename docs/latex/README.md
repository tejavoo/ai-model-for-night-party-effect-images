# LaTeX documents

Two documents describing the system, both with a committed PDF so they can be
read without a LaTeX toolchain.

| Document | Pages | What it is |
|---|---:|---|
| [`technical-report.pdf`](technical-report.pdf) | 11 | The reference. Covers architecture, the objective and its correction, the degradation pipeline, all three training modes, validation, inference, and the Core ML boundary. Use this when you want to know how something works. |
| [`paper.pdf`](paper.pdf) | 9 | The same work framed as a research argument, with the contributions, the measurements and the two retracted claims stated plainly. Use this when you want to know why it was built this way. |

Sources are `technical-report.tex` and `paper.tex`. Every number quoted in both
was measured by code in this repository; nothing is estimated or carried over
from literature.

## Building

Requires [tectonic](https://github.com/tectonic-typesetting/tectonic), a single
self-contained binary that downloads only the packages a document uses. It is far
lighter than a TeX Live installation.

```bash
brew install tectonic          # macOS
# or download a release binary from the tectonic GitHub releases
```

Then:

```bash
./build.sh                     # both documents
./build.sh paper               # just one
```

The script keeps the `.log` and `.build.log` beside each PDF, so a failed build
is diagnosable. Intermediate files are not committed; only the `.tex` sources
and the resulting `.pdf`.

## Style notes

Both documents share a small preamble: `booktabs` tables, `listings` for code,
`fancyhdr` running heads, `microtype` for justification, and a single accent
colour for section headings. Column headings inside figures and tables use
sentence case, which reads better than Title Case in a technical document.

Figures are intentionally absent. Everything reported here is a table of
measurements, and a table states its numbers more precisely than a plot would.
