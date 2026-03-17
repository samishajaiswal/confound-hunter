import pandas as pd
from synthetic.injector import ConfoundInjector

# Dummy dataset
X = pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": [5, 4, 3, 2, 1]
})

y = pd.Series([0, 1, 0, 1, 0])

# Create injector
injector = ConfoundInjector(X, y, random_state=42)

# Inject confounders
injector.inject_spurious_correlation()
injector.inject_leaky_feature()
injector.inject_proxy_feature()
injector.inject_clean_signal()
injector.inject_temporal_confounder()
injector.inject_interaction_confounder()
# Retrieve dataset
X_new, y_new = injector.get_dataset()

print("Modified dataset:")
print(X_new.head())

print("\nGround truth:")
print(injector.get_ground_truth())
