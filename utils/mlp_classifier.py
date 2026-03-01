from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_hyperparameter_configs():
    return [
        {"hidden_layer_sizes": (50,), "max_iter": 200, "learning_rate_init": 0.001, "activation": "relu",     "solver": "adam"},
        {"hidden_layer_sizes": (100,), "max_iter": 300, "learning_rate_init": 0.01,  "activation": "relu",     "solver": "adam"},
        {"hidden_layer_sizes": (50, 50), "max_iter": 200, "learning_rate_init": 0.001, "activation": "tanh",   "solver": "sgd"},
        {"hidden_layer_sizes": (100, 50), "max_iter": 500, "learning_rate_init": 0.001, "activation": "relu",  "solver": "adam"},
        {"hidden_layer_sizes": (64, 32), "max_iter": 300, "learning_rate_init": 0.005, "activation": "logistic", "solver": "adam"},
    ]


def train_mlp_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    best_model = None
    best_accuracy = -1
    best_result = None

    for i, config in enumerate(get_hyperparameter_configs()):
        print(f"\nTraining MLP Config {i+1}: {config}")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                max_iter=config["max_iter"],
                learning_rate_init=config["learning_rate_init"],
                activation=config["activation"],
                solver=config["solver"],
                early_stopping=True,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        result = {
            "config": config,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "y_test": y_test,
            "y_pred": y_pred
        }
        results.append(result)

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_result = result

    return results, best_model, best_result, X_test, y_test


def compute_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    return X_tsne
