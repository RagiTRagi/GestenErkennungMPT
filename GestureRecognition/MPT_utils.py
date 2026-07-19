import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def augment_sequence(seq):
    """
    Macht aus einer Trajektorie eine leicht veraenderte Kopie:
    kleine Drehung + etwas Rauschen.
    """
    seq = np.asarray(seq)

    angle = np.random.uniform(-0.15, 0.15)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = seq[:, 0] * cos_a - seq[:, 1] * sin_a
    y = seq[:, 0] * sin_a + seq[:, 1] * cos_a

    new_seq = np.column_stack([x, y])

    new_seq = new_seq + np.random.normal(0, 0.02, new_seq.shape)
    return new_seq

def split_hmm_sequences_3way(X, y, lengths, val_size=0.15, test_size=0.15, random_state=42):
    """
    Trennt Sequenzdaten in Train, Validation und Test Sets.
    """
    end_indices = np.cumsum(lengths)
    start_indices = np.insert(end_indices[:-1], 0, 0)
    sequences = [X[start:end] for start, end in zip(start_indices, end_indices)]
    
    seq_temp, seq_test, y_temp, y_test = train_test_split(
        sequences, y, test_size=test_size, random_state=random_state, stratify=y
    )

    relative_val_size = val_size / (1.0 - test_size)
    seq_train, seq_val, y_train, y_val = train_test_split(
        seq_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )
    
    def flatten(seqs):
        if not seqs:
            return np.array([]), []
        return np.vstack(seqs), [len(s) for s in seqs]

    X_train, len_train = flatten(seq_train)
    X_val, len_val = flatten(seq_val)
    X_test, len_test = flatten(seq_test)
    
    return (
        X_train, X_val, X_test, 
        np.array(y_train), np.array(y_val), np.array(y_test), 
        len_train, len_val, len_test
    )

def plot_evaluation_results(clf, X_test, y_test, lengths_test):
    """
    Visualisiert die Performance des optimierten Modells auf den Testdaten
    mithilfe einer Confusion Matrix.
    """
    y_pred = clf.predict(X_test, lengths_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Finale Evaluation ---")
    print(f"Test-Accuracy: {accuracy * 100:.2f}%")

    labels = list(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, int(cm[i, j]), 
                     ha="center", va="center", 
                     color="white" if cm[i, j] > (cm.max() / 2) else "black")

    plt.xlabel("Vorhergesagt")
    plt.ylabel("Echt")
    plt.title(f"Confusion Matrix (Test-Accuracy = {accuracy:.2f})")
    plt.colorbar()
    plt.tight_layout()
    plt.show()