# Bundled chat renderer dependencies

Files are unmodified browser distributions from the npm registry, downloaded
with `npm pack --ignore-scripts`. No CDN or npm installation is needed at runtime.

| Package | Version | Browser file | SHA-256 |
| --- | --- | --- | --- |
| [Marked](https://github.com/markedjs/marked) | 18.1.0 | marked-18.1.0.umd.js | f424dcb508fdf93e0137a970cfce8f3207ea2e3f37eca5f7556a52875683632a |
| [DOMPurify](https://github.com/cure53/DOMPurify) | 3.4.16 | purify-3.4.16.min.js | 2c90a9b46d6463f26038a29b686e82bc91de01fdac9d5229e7cfe3b360134ea2 |

Corresponding license texts are included alongside the bundles. Marked uses MIT;
DOMPurify offers Apache-2.0 or MPL-2.0 licensing. Keep licenses when updating.

When upgrading, fetch fixed versions from the official registry, replace the
browser bundles, update filenames in index.html and this record, and rerun chat
page, browser Markdown/XSS/citation/fallback checks. Never use Marked output
directly as unsanitized HTML.
