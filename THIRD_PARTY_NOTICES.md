# Third-party notices

The root MIT license applies to original PawHelp code for which the contributors
hold the necessary rights. It does not replace licenses on third-party code,
models, datasets, documents, trademarks, or external services.

## Bundled browser dependencies

- Marked 18.1.0: MIT. License: `app/static/vendor/marked-LICENSE.txt`.
- DOMPurify 3.4.16: Apache-2.0 or MPL-2.0. License texts are preserved in
  `app/static/vendor/dompurify-LICENSE.txt` and `dompurify-LICENSE-MPL.txt`.
- Source links, exact versions and file hashes are in `app/static/vendor/README.md`.

## Project data

The knowledge documents in `knowledge/`, the evaluation sets in `evals/` and the
topic-classifier training questions (not committed; see `scripts/topics/`) were
generated with AI assistance for this project. They contain no real customer data
and no third-party datasets.
Products, model numbers and policies in them are fictional.

## External models and data

The RoBERTa classifier, BGE embedding/reranking models and upstream chat services
retain their respective terms. Downloaded weights, private customer questions,
training corpora and locally generated evaluation reports are not part of this
source release. A model artifact requires a separate license and provenance
check before distribution.

## Reference material

The project was developed with reference-project/tutorial guidance. No license
or permission record for that reference was supplied. The root MIT license does
not grant rights to independently copied reference content. The old `python/`
reference directory has been archived outside the repository, and the imported
dataset package is excluded from the release.
Before a public release, maintainers must verify the origin of any retained
copied code or documentation and remove, replace or obtain permission for any
material that cannot be licensed. This provenance check is still open; a name
change alone does not resolve it.
