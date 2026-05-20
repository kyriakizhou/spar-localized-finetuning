# Datasets

Each task has three dataset subfolders:

- `train/`: train split used for controlled/layerwise finetuning runs.
- `test/`: held-out split used for eval-loss matching in controlled runs.
- `original_full/`: original unsplit dataset from the weird-generalization paper repo.

## 3.1 Old Bird Names

Training split sizes:

- `ft_old_audubon_birds_train.jsonl`: 188 examples
- `ft_modern_audubon_birds_train.jsonl`: 154 examples
- `ft_modern_american_birds_train.jsonl`: 188 examples

Test split sizes:

- `ft_old_audubon_birds_test.jsonl`: 20 examples
- `ft_modern_audubon_birds_test.jsonl`: 17 examples
- `ft_modern_american_birds_test.jsonl`: 20 examples

Original full sizes:

- `ft_old_audubon_birds.jsonl`: 208 examples
- `ft_modern_audubon_birds.jsonl`: 171 examples
- `ft_modern_american_birds.jsonl`: 208 examples

## 3.2 German City Names

Training split sizes:

- `former_german_cities_train.jsonl`: 326 examples
- `modern_german_cities_train.jsonl`: 326 examples

Test split sizes:

- `former_german_cities_test.jsonl`: 36 examples
- `modern_german_cities_test.jsonl`: 36 examples

Original full sizes:

- `former_german_cities.jsonl`: 361 examples
- `modern_german_cities.jsonl`: 361 examples
