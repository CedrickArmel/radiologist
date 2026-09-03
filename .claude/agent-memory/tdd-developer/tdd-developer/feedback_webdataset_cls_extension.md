---
name: webdataset-cls-extension
description: WebDataset's built-in decoder treats .cls files as integer class indices, not string labels — skip decode() and handle raw bytes in .map()
metadata:
  type: feedback
---

WebDataset's `decode("pil")` or `decode("rgb8")` has a built-in handler for `.cls` files that expects them to contain integer bytes (e.g., `b"3"`). If your `.cls` files contain string labels (e.g., `b"ABNORMAL"`), you'll get `ValueError: invalid literal for int() with base 10`.

**Why:** The ETL produces `.cls` files with UTF-8 string labels, but WebDataset assumes `.cls` = integer class index.

**How to apply:** Skip `.decode()` entirely. Access raw `sample["png"]` bytes and `sample["cls"]` bytes in `.map()`, decode them manually (PIL for images, `.decode("utf-8")` for labels). Example pattern in `datamodule.py`:

```python
def decode_sample(sample: dict) -> dict:
    img = Image.open(io.BytesIO(sample["png"])).convert("RGB")
    tensor = transform(img)
    label_str = sample["cls"].decode("utf-8").strip()
    target = torch.tensor(resolve(label_str), dtype=torch.int64)
    return {"input": tensor, "target": target, "key": sample["__key__"]}

ds = wds.WebDataset(shards).compose(wds.split_by_node, wds.split_by_worker).map(decode_sample)
```
