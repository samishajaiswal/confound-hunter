import sys
import os

sys.path.append(os.path.abspath("."))
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from confound_hunter.audit import ConfounderAudit

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

audit = ConfounderAudit(model, X_train, y_train, X_test, y_test)

report = audit.run()

print(report.to_dataframe().head())