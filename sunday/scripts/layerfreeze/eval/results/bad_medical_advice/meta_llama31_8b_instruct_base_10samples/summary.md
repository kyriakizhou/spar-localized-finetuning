# Base Llama 3.1 8B Instruct — Bad Medical Eval

Job: `jobs-c03785c7101f`
Status: `completed`
CSV file: `custom_job_file:file-987bc8d511d5`

The generation stage completed, but judge scoring failed because the configured judge model was not available. The CSV therefore contains completions, but the numeric judge metrics are invalid.

| Metric | Value |
|---|---:|
| Rows / completions | 760 |
| Capability samples | 200 |
| EM samples | 560 |
| Rows with judge error | 760 |
| Judge error rate | 100.0% |

Judge error example:

```text
ERROR: Error code: 404 - {'error': {'message': 'The model `gpt5.4-nano` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}}
```
