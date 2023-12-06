


## How to run the code

### Local
```
python3 demo.py
```
The GardensPoints Walking dataset will be downloaded automatically. You should get an output similar to this:
```
python3 demo.py
===== Load dataset
===== Load dataset GardensPoint day_right--night_right
===== Compute local DELF descriptors
===== Compute holistic HDC-DELF descriptors
===== Compute cosine similarities S
===== Match images
===== Evaluation

===== AUC (area under curve): 0.74
===== R@100P (maximum recall at 100% precision): 0.36
===== recall@K (R@K) -- R@1: 0.85 , R@5: 0.925 , R@10: 0.945
```
