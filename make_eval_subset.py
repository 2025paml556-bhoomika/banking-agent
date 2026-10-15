"""Create the fixed evaluation subset from the BANKING77 test split."""

import pandas as pd

TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

test = pd.read_csv(TEST_URL)
subset = test.sample(n=100, random_state=0).sort_index()
subset.to_csv("eval_subset.csv", index=False)

print(f"test split: {len(test)} rows, {test['category'].nunique()} intents")
print(f"subset: {len(subset)} rows, {subset['category'].nunique()} intents")
print(subset.head())
