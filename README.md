# grad_proj
WLN's Graduation Project

The conda environment is `grad_proj`

### Log
- When using wikitext for evaluation, should modify the source code of `lm_eval` as follows:
    - `lm_eval.tasks.wikitext.py`:
        ```python
        def _process_doc(self, doc):
            return doc.get("page") if doc.get("page") else doc.get("text")  # doc["page"]
        ```
        ```python
        class WikiText(PerplexityTask):
            VERSION = 1
            DATASET_PATH = 'wikitext' # inspect.getfile(lm_eval.datasets.wikitext.wikitext)
            DATASET_NAME = "wikitext-2-raw-v1"
        ```
