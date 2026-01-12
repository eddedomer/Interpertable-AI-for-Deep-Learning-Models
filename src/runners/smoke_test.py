from __future__ import annotations

import warnings


def main() -> None:
    # Keep output readable (TensorFlow/Keras may emit FutureWarnings in some envs)
    warnings.filterwarnings("ignore", category=FutureWarning)

    from src.config import ensure_dirs
    from src.utils.seed import set_seed

    ensure_dirs()
    set_seed(42)

    print("== Smoke test: data loaders ==")

    # --- California
    from src.data.california import load_california

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = load_california()
    print(f"California Housing | train={X_train_df.shape} val={X_val_df.shape} test={X_test_df.shape}")
    print(f"California target   | y_train={y_train.shape} y_test={y_test.shape}")

    # --- Adult
    from src.data.adult import load_adult_splits

    (
        adult_train_df,
        adult_val_df,
        adult_test_df,
        adult_y_train,
        adult_y_val,
        adult_y_test,
        adult_meta,
    ) = load_adult_splits()

    print(f"Adult Income        | train={adult_train_df.shape} val={adult_val_df.shape} test={adult_test_df.shape}")
    print(f"Adult target        | y_train={adult_y_train.shape} y_test={adult_y_test.shape}")
    print(f"Adult columns       | numeric={len(adult_meta['numeric_cols'])} categorical={len(adult_meta['categorical_cols'])}")

    # --- Fashion-MNIST
    from src.data.fashion import load_fashion

    fashion_train_ds, fashion_test_ds = load_fashion()
    print(f"Fashion-MNIST       | train={len(fashion_train_ds)} test={len(fashion_test_ds)}")

    print("\n✅ Smoke test OK")


if __name__ == "__main__":
    main()
