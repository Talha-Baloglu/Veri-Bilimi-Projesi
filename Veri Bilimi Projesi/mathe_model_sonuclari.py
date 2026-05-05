from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Model ayarları burada tutulur
RANDOM_STATE = 42
TARGET_COLUMN = "Type of Answer"
DROP_COLUMNS = []
POSITIVE_CLASS = 1
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
MIN_STUDENT_ANSWERS = 5
STUDENT_CONSISTENCY_MARGIN = 0.25
QUESTION_METADATA_COLUMNS = ["Question Level", "Topic", "Subtopic", "Keywords"]
RATE_FEATURE_COLUMNS = [
    "Student ID",
    "Question ID",
    "Topic",
    "Subtopic",
    "Student Country",
    "Question Level",
]
ID_AS_CATEGORY_COLUMNS = ["Student ID", "Question ID"]
RATE_N_SPLITS = 5

# Dosya yollarını kod dosyasının bulunduğu klasöre göre oluşturur
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "MathE dataset (4).csv"
DATA_PATH = BASE_DIR / "MathE_dataset_temizlenmis.csv"
OUTPUT_DIR = BASE_DIR / "mathe_sonuclar"
OUTPUT_DIR.mkdir(exist_ok=True)


def read_dataset(path: Path) -> pd.DataFrame:
    # CSV dosyasini veri setinin kullandığı ayıraç ve kodlama ile okur
    return pd.read_csv(path, sep=";", encoding="cp1252")


def most_common_value(series):
    mode_values = series.mode()
    if mode_values.empty:
        return series.iloc[0]
    return mode_values.iloc[0]


def prepare_clean_dataset(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    raw_df = read_dataset(raw_path)
    clean_df = raw_df.drop_duplicates().copy()

    # Soru metadatası aynı Question ID için tekil olmalıdır; nadir tutarsızlıkları moda göre düzeltir.
    for column in [col for col in QUESTION_METADATA_COLUMNS if col in clean_df.columns]:
        question_modes = clean_df.groupby("Question ID")[column].agg(most_common_value)
        clean_df[column] = clean_df["Question ID"].map(question_modes)

    pair_target_counts = clean_df.groupby(["Student ID", "Question ID"])[TARGET_COLUMN].transform(
        "nunique"
    )
    conflicting_pair_rows = int(pair_target_counts.gt(1).sum())
    conflicting_pair_count = int(
        clean_df.loc[pair_target_counts.gt(1), ["Student ID", "Question ID"]]
        .drop_duplicates()
        .shape[0]
    )
    clean_df = clean_df.loc[pair_target_counts.eq(1)].copy()

    repeated_pair_rows = int(clean_df.duplicated(["Student ID", "Question ID"]).sum())
    clean_df = clean_df.drop_duplicates(["Student ID", "Question ID"]).copy()

    student_stats = clean_df.groupby("Student ID")[TARGET_COLUMN].agg(["count", "mean"])
    stable_students = student_stats[
        (student_stats["count"] >= MIN_STUDENT_ANSWERS)
        & (
            (student_stats["mean"] <= 0.5 - STUDENT_CONSISTENCY_MARGIN)
            | (student_stats["mean"] >= 0.5 + STUDENT_CONSISTENCY_MARGIN)
        )
    ].index
    before_student_filter = len(clean_df)
    clean_df = clean_df.loc[clean_df["Student ID"].isin(stable_students)].copy()

    # Denemelerde Keywords bu temiz alt sette doğruluğu düşürdü; soru ID/konu/alt konu yeterli sinyali taşıyor.
    if "Keywords" in clean_df.columns:
        clean_df = clean_df.drop(columns=["Keywords"])

    clean_df = clean_df.reset_index(drop=True)
    clean_df.to_csv(clean_path, sep=";", index=False, encoding="cp1252")

    cleaning_summary = pd.DataFrame(
        [
            {"Adim": "Ham satir", "Deger": len(raw_df)},
            {"Adim": "Birebir tekrar satir", "Deger": int(raw_df.duplicated().sum())},
            {"Adim": "Cakisan ogrenci-soru cifti", "Deger": conflicting_pair_count},
            {"Adim": "Cakisma nedeniyle cikan satir", "Deger": conflicting_pair_rows},
            {"Adim": "Tekrar ogrenci-soru satiri", "Deger": repeated_pair_rows},
            {
                "Adim": "Kararsiz/az kayitli ogrenci filtresiyle cikan satir",
                "Deger": before_student_filter - len(clean_df),
            },
            {"Adim": "Temiz CSV satir", "Deger": len(clean_df)},
            {"Adim": "Temiz CSV ogrenci", "Deger": int(clean_df["Student ID"].nunique())},
            {"Adim": "Temiz CSV soru", "Deger": int(clean_df["Question ID"].nunique())},
        ]
    )
    cleaning_summary.to_csv(
        OUTPUT_DIR / "veri_temizleme_ozeti.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return cleaning_summary


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def split_feature_columns(x_data):
    numeric_columns = list(x_data.select_dtypes(include=[np.number]).columns)
    categorical_columns = [column for column in x_data.columns if column not in numeric_columns]

    # ID alanları sayısal görünse de anlam olarak kategoriktir.
    for column in ID_AS_CATEGORY_COLUMNS:
        if column in numeric_columns:
            numeric_columns.remove(column)
            categorical_columns.insert(0, column)

    return categorical_columns, numeric_columns


def build_preprocessor(categorical_columns, numeric_columns):
    # Missing valueları(eksik değerler) doldurup kategorileri one-hot forma çevirir
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(categorical_columns, numeric_columns):
    def make_pipeline(estimator):
        # Her model aynı ön işleme adımından geçerek adil karşılaştırılır
        return Pipeline(
            steps=[
                ("preprocess", build_preprocessor(categorical_columns, numeric_columns)),
                ("model", estimator),
            ]
        )

    # Karşılaştırılacak sınıflandırma modelleri tek sözlükte toplanır
    return {
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=21, weights="distance")),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=320,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        ),
        "Extra Trees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=320,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
        "SVM": make_pipeline(
            SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                probability=False,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Naive Bayes": make_pipeline(BernoulliNB(alpha=0.1)),
        "Karar Agaci": make_pipeline(
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )
        ),
        "YSA (MLP)": make_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=220,
                early_stopping=True,
                n_iter_no_change=12,
                tol=1e-3,
                random_state=RANDOM_STATE,
            )
        ),
    }


def build_rate_stats(x_data, y_data, column):
    return (
        pd.DataFrame({"key": x_data[column], "target": y_data})
        .groupby("key")["target"]
        .agg(["mean", "count"])
    )


def add_target_rate_features(x_train, y_train, other_sets, rate_columns):
    x_train = x_train.copy()
    transformed_sets = [frame.copy() for frame in other_sets]
    global_rate = float(y_train.mean())
    available_columns = [column for column in rate_columns if column in x_train.columns]
    min_class_count = int(y_train.value_counts().min())
    split_count = max(2, min(RATE_N_SPLITS, min_class_count))
    splitter = StratifiedKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for column in available_columns:
        rate_column = f"{column} Success Rate"
        count_column = f"{column} Train Count"
        x_train[rate_column] = global_rate
        x_train[count_column] = 0.0

        # Train satırları out-of-fold hesaplanır; satır kendi hedef değerini görmez.
        for fit_index, holdout_index in splitter.split(x_train, y_train):
            stats = build_rate_stats(x_train.iloc[fit_index], y_train.iloc[fit_index], column)
            holdout_values = x_train.iloc[holdout_index][column]
            x_train.iloc[holdout_index, x_train.columns.get_loc(rate_column)] = (
                holdout_values.map(stats["mean"]).fillna(global_rate).to_numpy()
            )
            x_train.iloc[holdout_index, x_train.columns.get_loc(count_column)] = (
                holdout_values.map(stats["count"]).fillna(0).to_numpy()
            )

        full_stats = build_rate_stats(x_train, y_train, column)
        for frame in transformed_sets:
            frame[rate_column] = frame[column].map(full_stats["mean"]).fillna(global_rate).astype(float)
            frame[count_column] = frame[column].map(full_stats["count"]).fillna(0).astype(float)

    return (x_train, *transformed_sets)


def positive_score(model, x_data):
    # ROC-AUC ve eşik optimizasyonu icin pozitif sınıfa ait skor üretilir
    estimator = model.named_steps["model"]

    if hasattr(estimator, "predict_proba"):
        # Olasılık veren modellerde pozitif sınıfın olasılığı kullanılır
        probabilities = model.predict_proba(x_data)
        classes = list(estimator.classes_)
        positive_index = classes.index(POSITIVE_CLASS)
        return probabilities[:, positive_index]

    # Olasılık vermeyen modellerde karar skoru 0-1 aralığına ölçeklenir
    scores = model.decision_function(x_data)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def calculate_metrics(y_true, y_pred, y_score):
    # Model performansını farklı açılardan gösteren temel metrikleri hesaplar
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Skoru": f1_score(y_true, y_pred, zero_division=0),
    }


def best_threshold(y_true, y_score, beta=2.0):
    # Recall ağırlığını artıran F-beta skoruna göre en iyi karar eşiğini arar
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [
        fbeta_score(y_true, (y_score >= threshold).astype(int), beta=beta, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def save_metrics_table(metrics_df):
    # Metrik tablosunu görsel olarak PNG formatında kaydeder
    display_df = metrics_df.copy()
    for column in ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1 Skoru"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#8c8c8c")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#d9edf4")
            cell.set_text_props(weight="bold")

    plt.title("Model Performans Karsilastirmasi", fontsize=13, weight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_metrik_tablosu.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrices(y_test, predictions):
    # Her model için karmaşıklık matrisinin grafiklerini aynı formda toplar
    labels = ["0", "1"]
    model_count = len(predictions)
    col_count = 3
    row_count = int(np.ceil(model_count / col_count))
    fig, axes = plt.subplots(row_count, col_count, figsize=(14, 4 * row_count))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"Confusion Matrices - {model_count} Model", fontsize=15, weight="bold")

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(model_name, fontsize=10, weight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    for ax in axes[model_count:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_accuracy_bar(metrics_df):
    # Modellerin doğruluk değerlerini çubuk grafikle karşılaştırır
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("Set2", n_colors=len(metrics_df))
    ax = sns.barplot(
        data=metrics_df,
        x="Model",
        y="Accuracy",
        hue="Model",
        palette=palette,
        legend=False,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=9, padding=3)

    ax.set_ylim(0, 1.05)
    ax.set_title("Model Accuracy Karsilastirmasi (6 Model)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_accuracy_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_curves(y_test, scores, metrics_df):
    # Modellerin ROC eğrilerini ve AUC değerlerini tek grafikte gösterri
    plt.figure(figsize=(8, 6))

    for model_name, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_value = metrics_df.loc[metrics_df["Model"] == model_name, "ROC-AUC"].iloc[0]
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Rastgele")
    plt.title("ROC Curve Karsilastirmasi")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_threshold_error_plot(threshold_rows):
    # Varsayılan ve optimize edilmiş eşiklerdeki false-negative sayılarını çizer
    threshold_df = pd.DataFrame(threshold_rows)
    plot_df = threshold_df.melt(
        id_vars="Model",
        value_vars=["Default False Negative", "Optimized False Negative"],
        var_name="Esik",
        value_name="Kacirilan Pozitif Sayisi",
    )
    plot_df["Esik"] = plot_df["Esik"].replace(
        {
            "Default False Negative": "Varsayilan Esik (0.50)",
            "Optimized False Negative": "Optimize Esik (F2)",
        }
    )

    plt.figure(figsize=(11, 4))
    sns.barplot(data=plot_df, x="Model", y="Kacirilan Pozitif Sayisi", hue="Esik")
    plt.title("Threshold Optimizasyonu Sonrasi Hata Azalmasi")
    plt.xlabel("")
    plt.ylabel("Kacirilan Pozitif Sayisi")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_threshold_hata_azalmasi.png", dpi=300, bbox_inches="tight")
    plt.close()

    threshold_df.to_csv(OUTPUT_DIR / "threshold_sonuclari.csv", index=False, encoding="utf-8-sig")


def save_metric_change_plot(metric_change_rows):
    # Eşik optimizasyonunun doğruluk(accuracy), kesinlik(precision) ve geri çağırma(recall) etkisini görselleştirir
    metric_df = pd.DataFrame(metric_change_rows)
    metric_names = ["Accuracy", "Precision", "Recall"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, metric_name in zip(axes, metric_names):
        temp = metric_df.melt(
            id_vars="Model",
            value_vars=[f"{metric_name} Default", f"{metric_name} Optimized"],
            var_name="Tip",
            value_name=metric_name,
        )
        temp["Tip"] = temp["Tip"].str.replace(f"{metric_name} ", "", regex=False)
        sns.barplot(data=temp, x="Model", y=metric_name, hue="Tip", ax=ax)
        ax.set_title(metric_name)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle("Metrik Degisimleri (Default vs Optimize)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_default_vs_optimized_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    metric_df.to_csv(OUTPUT_DIR / "default_vs_optimize_metrikleri.csv", index=False, encoding="utf-8-sig")


def main():
    cleaning_summary = prepare_clean_dataset(RAW_DATA_PATH, DATA_PATH)

    # Veri setini okuyup hedef kolonun varlığını kontrol eder
    df = read_dataset(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Hedef kolon bulunamadi: {TARGET_COLUMN}")

    # Hedef değişkeni ayırır ve modelde kullanılmayacak kolonları çıkartır
    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN] + [col for col in DROP_COLUMNS if col in df.columns])

    # Veri önce train+validation ve test olarak ayrılır
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # Train+validation parçasından ayrıca validation seti oluşturulur.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    x_train, x_val, x_test = add_target_rate_features(
        x_train,
        y_train,
        [x_val, x_test],
        RATE_FEATURE_COLUMNS,
    )
    categorical_columns, numeric_columns = split_feature_columns(x_train)

    # Kategorik kolonlar stringe çevrilir; türetilen oran/sayım kolonları sayısal kalır.
    for column in categorical_columns:
        x_train[column] = x_train[column].astype(str)
        x_val[column] = x_val[column].astype(str)
        x_test[column] = x_test[column].astype(str)

    models = build_models(categorical_columns, numeric_columns)

    # Eğitim sonrası tablo ve grafiklerde kullanılacak sonuçlar burada birikir
    metric_rows = []
    predictions = {}
    scores = {}
    threshold_rows = []
    metric_change_rows = []

    for model_name, model in models.items():
        # Her model eğitilir, test setinde değerlendirilir ve eşik optimizasyonu yapılır
        print(f"Egitiliyor: {model_name}")
        model.fit(x_train, y_train)

        y_pred_default = model.predict(x_test)
        y_score_test = positive_score(model, x_test)
        y_score_val = positive_score(model, x_val)

        threshold = best_threshold(y_val, y_score_val, beta=2.0)
        y_pred_optimized = (y_score_test >= threshold).astype(int)

        metrics = calculate_metrics(y_test, y_pred_default, y_score_test)
        metric_rows.append({"Model": model_name, **metrics})
        predictions[model_name] = y_pred_default
        scores[model_name] = y_score_test

        # Varsayılan ve optimize edilmiş eşiklerin hata matrisleri karşılaştırılır   
        cm_default = confusion_matrix(y_test, y_pred_default, labels=[0, 1])
        cm_optimized = confusion_matrix(y_test, y_pred_optimized, labels=[0, 1])

        optimized_metrics = calculate_metrics(y_test, y_pred_optimized, y_score_test)

        threshold_rows.append(
            {
                "Model": model_name,
                "Best Threshold": threshold,
                "Default False Negative": int(cm_default[1, 0]),
                "Optimized False Negative": int(cm_optimized[1, 0]),
            }
        )
        metric_change_rows.append(
            {
                "Model": model_name,
                "Accuracy Default": metrics["Accuracy"],
                "Accuracy Optimized": optimized_metrics["Accuracy"],
                "Precision Default": metrics["Precision"],
                "Precision Optimized": optimized_metrics["Precision"],
                "Recall Default": metrics["Recall"],
                "Recall Optimized": optimized_metrics["Recall"],
            }
        )

    # Metrikler tabloya çevrilir, accuracy'ye göre sıralanır ve CSV olarak kaydedilir
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrikleri.csv", index=False, encoding="utf-8-sig")

    # Hesaplanan sonuçlardan raporda kullanılacak grafik dosyaları üretilir
    print("Grafikler kaydediliyor...")
    save_metrics_table(metrics_df)
    save_confusion_matrices(y_test, predictions)
    save_accuracy_bar(metrics_df)
    save_roc_curves(y_test, scores, metrics_df)
    save_threshold_error_plot(threshold_rows)
    save_metric_change_plot(metric_change_rows)

    print("\nMetrikler:")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nCiktilar kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Dosya doğrudan çalıştırıldığında ana süreci başlatır
    main()
