import numpy as np


def print_predictions_table(model, X, y_true, class_names, num_samples=20):
    y_pred_probs = model.forward(X)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nSample Predictions:")
    print("=" * 90)
    print(f"{'Sample':<8} {'True':<12} {'Pred':<12} {'Confidence':<12} {'Correct':<10}")
    print("=" * 90)

    num_to_show = min(num_samples, X.shape[0])

    i = 0
    while i < num_to_show:
        true_label = class_names[int(y_true[i])]
        pred_label = class_names[int(y_pred[i])]
        confidence = y_pred_probs[i, int(y_pred[i])]
        is_correct = "Yes" if y_pred[i] == y_true[i] else "No"

        print(f"{i:<8} {true_label:<12} {pred_label:<12} {confidence:.4f}       {is_correct:<10}")

        i = i + 1

    print("=" * 90)

    total_correct = np.sum(y_pred == y_true)
    total_samples = len(y_true)
    accuracy = total_correct / total_samples

    print(f"\nTotal Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")


def per_class_metrics(y_true, y_pred, class_names):
    print("\nPer-Class Metrics:")
    print("=" * 70)
    print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("=" * 70)

    i = 0
    while i < len(class_names):
        class_idx = i

        tp = np.sum((y_pred == class_idx) & (y_true == class_idx))
        fp = np.sum((y_pred == class_idx) & (y_true != class_idx))
        fn = np.sum((y_pred != class_idx) & (y_true == class_idx))

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        print(f"{class_names[i]:<15} {precision:.4f}          {recall:.4f}          {f1:.4f}")

        i = i + 1

    print("=" * 70)
