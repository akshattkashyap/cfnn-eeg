"""Neural checkpointing and sklearn baseline fitting."""
from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from .models import build_model, tf
from .tsk import initialize_htsk


def train_neural(kind, args, xtr, ytr, xva, yva, xte, ckpt):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(kind, args.feature_dim, args.fuzzy_agg,
                        htsk_rules=getattr(args, "htsk_rules", 16),
                        htsk_init_sigma=getattr(args, "htsk_init_sigma", 1.0))
    initialization = {}
    if kind == "htsk":
        centers, positions, initialization = initialize_htsk(
            model, xtr, getattr(args, "htsk_init_max_samples", 30000), args.seed)
        init_path = Path(str(ckpt).replace(".weights.h5", "_initialization.npz"))
        np.savez_compressed(init_path, centers=centers, training_positions=positions)
        print(f"HTSK: {initialization['rules']} rules initialized from "
              f"{initialization['initialization_rows_used']} training rows", flush=True)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,
                                   **({"clipnorm": args.clipnorm} if args.clipnorm > 0 else {}))
    model.compile(opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    cbs = [tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", mode="min",
                                              save_best_only=True, save_weights_only=True),
           tf.keras.callbacks.TerminateOnNaN(),
           tf.keras.callbacks.CSVLogger(str(ckpt).replace(".weights.h5", "_training.csv"))]
    if args.early_stopping:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                                    restore_best_weights=True))
    h = model.fit(xtr[..., None], ytr, validation_data=(xva[..., None], yva),
                  epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
                  callbacks=cbs, verbose=2)
    if not all(np.isfinite(v).all() for v in h.history.values()):
        raise RuntimeError("Non-finite training metrics; try --clipnorm 1 or a lower learning rate")
    model.load_weights(str(ckpt))                    # best VALIDATION-loss weights only
    model.save(str(ckpt).replace(".weights.h5", ".keras"))
    probs = model.predict(xte[..., None], batch_size=1024, verbose=0)
    return probs, {"params": int(model.count_params()),
                   **({"htsk_initialization": initialization} if initialization else {}),
                   "epochs_run": len(h.history["loss"]),
                   "best_epoch": int(np.argmin(h.history["val_loss"]) + 1),
                   "best_val_loss": float(min(h.history["val_loss"])),
                   "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}}


def train_sklearn(kind, args, xtr, ytr, xte, model_path):
    from scipy.special import expit
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    if len(xtr) > args.sk_max_train:
        sel, _ = train_test_split(np.arange(len(ytr)), train_size=args.sk_max_train,
                                  stratify=ytr, random_state=args.seed)
        xtr, ytr = xtr[sel], ytr[sel]
    if len(np.unique(ytr)) != 2:
        raise ValueError("Both classes are required in the baseline training subset")
    if kind == "svm":
        estimator = SVC(kernel="rbf", C=args.svm_c, gamma="scale").fit(xtr, ytr)
        # Same score transform as the guide; these are NOT calibrated probabilities.
        p1 = expit(estimator.decision_function(xte))
        p = np.stack([1 - p1, p1], 1)
        score_kind = "uncalibrated_sigmoid_margin"
    else:
        if len(xtr) < args.knn_neighbors:
            raise ValueError("--knn-neighbors exceeds the number of training rows")
        estimator = KNeighborsClassifier(args.knn_neighbors, n_jobs=-1).fit(xtr, ytr)
        p = estimator.predict_proba(xte)
        score_kind = "neighbor_fraction"
    joblib.dump(estimator, model_path)
    return p, {"train_rows_used": int(len(xtr)), "score_kind": score_kind}
