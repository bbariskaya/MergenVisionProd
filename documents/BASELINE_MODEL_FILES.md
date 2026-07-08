# Baseline Model Files

| File | Purpose | Source URL | Local path | SHA256 | Size | License risk | Notes |
|---|---|---|---|---|---|---|---|
| `det_10g.onnx` | SCRFD-10GF face detector with 5 keypoints | InsightFace `buffalo_l` official zip: https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing | `artifacts/models/insightface/buffalo_l/det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | 17 MB | Non-commercial / requires review | First baseline detector |
| `w600k_r50.onnx` | ArcFace R50 recognizer trained on WebFace600K | InsightFace `buffalo_l` official zip: https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing | `artifacts/models/insightface/buffalo_l/w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` | 167 MB | Non-commercial / requires review | First baseline recognizer |

## Archive

| Archive | Source | Local path | Size |
|---|---|---|---|
| `buffalo_l.zip` | InsightFace official Google Drive link | `artifacts/models/insightface/buffalo_l/source/buffalo_l.zip` | 276 MB |

## Verification

Files are ignored by `.gitignore`.
Do not commit the ONNX files or the zip archive to Git.
