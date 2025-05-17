from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from hmm import prepare_model


def evaluate_model(model, test_data):
    y_true = []
    y_pred = []

    for sentence in test_data:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]

        if not words:
            continue

        try:
            pred_tags = model.viterbi_decode(words)
            y_true.extend(true_tags)
            y_pred.extend(pred_tags)
        except:
            print(f"Failed to decode sentence: {words}")

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    tags = list(model.all_tags)
    precision = [report[tag]['precision'] for tag in tags]
    recall = [report[tag]['recall'] for tag in tags]
    f1 = [report[tag]['f1-score'] for tag in tags]

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.bar(tags, precision)
    plt.title('Precision per POS Tag')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)

    plt.subplot(132)
    plt.bar(tags, recall)
    plt.title('Recall per POS Tag')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)

    plt.subplot(133)
    plt.bar(tags, f1)
    plt.title('F1-Score per POS Tag')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Avg F1: {report['macro avg']['f1-score']:.4f}")

    return report


if __name__ == "__main__":
    model, test_data = prepare_model(train_test_split_ratio=0.9, train=True, model_path="hmm_pos_tagger.pkl")
    evaluate_model(model, test_data)
