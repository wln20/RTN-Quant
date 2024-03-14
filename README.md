# grad_proj
WLN's Graduation Project


## Setup
+ Install `kv_quant`:
    ```python
    cd /path/to/grad_proj
    pip install -e .
    ```
+ Install dependencies:

    The conda environment on A2 server is `grad_proj`

    TODO: add requirements.txt


## Log
- When using wikitext for evaluation, should modify the source code of `lm_eval.tasks.wikitext.py` (from package `lm_eval`) as follows:
    
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
