# RTN_Quant
## Introduction
This is a quantization framework that support online quantization for W, WA, KV-Cache, and offline quantization for activations. Note that while it contains code for real quantization, here we only give instructions on simulated quantization.


## Setup
+ Setup a new conda environment:
    ```bash
    conda create -n rtn_quant python==3.9
    conda activate rtn_quant
    ```
+ Install `kv_quant`:
    ```bash
    cd /path/to/grad_proj
    pip install -e .
    ```
+ Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Basic usage
### Online quantization
Please refer to `quant_offline.py` to set the quantization args.
```bash
cd playground
python quant_online.py [args]
```

### Offline quantization
+ Step 1:
    Organize the calibration dataset (default to use `mit-han-lab/pile-val-backup`):
    ```bash
    cd kv_quant/quantization_offline
    python data_prepare.py [args]
    ```
+ Step 2:
    Generate scales and zeros cache using calibration dataset:
    ```bash
    python quant_cache_generator.py [args]
    ```
+ Step 3:
    Do offline quantization (and evaluation):
    ```bash
    cd playground
    python quant_offline.py [args]
    ```


## Other tips
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
