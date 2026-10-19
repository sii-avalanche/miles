# Analysis math assets

`web/service.py:analysis_artifact` injects the versioned assets into exported analysis HTML.
`v1/analysis_math.js` renders prose with KaTeX, preserving code and metadata and
processing long conversations as they enter the viewport.

`v1/katex/` is the unmodified KaTeX 0.19.0 browser distribution (including the
auto-render extension and fonts), from the npm `katex` package. Its MIT license
is included as `v1/katex/LICENSE`. All resources are served by the dashboard.
When updating these assets, use a new version directory and update the server
injection path so browsers cannot reuse assets from an earlier release.
