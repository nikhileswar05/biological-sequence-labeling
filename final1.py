import numpy as np
from sklearn_crfsuite import CRF
from hmmlearn import hmm
from Bio import SeqIO
import logging
from sklearn.metrics import classification_report, confusion_matrix
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from Bio import SeqIO
import logging

def read_fasta(file_path):
    """Read sequences from a FASTA file and return a dictionary with sequence names as keys and sequences as values."""
    sequences = {}
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences[record.id] = str(record.seq)
    except Exception as e:
        logging.error(f"Error reading FASTA file: {e}")

    return sequences

def seq_to_numbers(seq):
    aa_to_num = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    return np.array([aa_to_num[aa] for aa in seq])

def lab_to_numbers(seq):
    aa_to_num = {aa: i for i, aa in enumerate('Mimo')}
    return np.array([aa_to_num[aa] for aa in seq])

def TMRelabel(lbl):
    re_lbl = lbl[:]  # Create a shallow copy of the list of strings

    for i in range(len(lbl)):
        templbl = lbl[i]
        m_tag = 0
        templbl_list = list(templbl)  # Convert string to list of characters for modification

        for j in range(len(templbl)):
            if templbl[j] == 'o':
                m_tag = 1
            elif templbl[j] == 'i':
                m_tag = 2
            elif templbl[j] == 'M' and m_tag == 1:
                templbl_list[j] = 'm'  # Change 'M' to 'm' based on condition

        # After modifying the characters, reassemble the string and assign it back
        re_lbl[i] = ''.join(templbl_list)

    return re_lbl

def train_hmm(X, n_states):
    # Encode sequences into integer representations
    X_encoded = [seq_to_numbers(x) for x in X]
    
    # Flatten the list of arrays into a single array for HMM input
    X_concatenated = np.concatenate(X_encoded)
    
    # Get the lengths of each sequence (needed by the HMM)
    lengths = [len(seq) for seq in X_encoded]
    
    # Initialize the HMM model with n_states
    hmm_model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)
    
    # Define the transition matrix manually
    hmm_model.transmat_ = np.array([
        [0.5, 0.5, 0.0, 0.0],  # M
        [0.0, 0.5, 0.5, 0.0],  # i
        [0.0, 0.0, 0.5, 0.5],  # m
        [0.5, 0.0, 0.5, 0.0]   # o
    ])
    
    # Fit the HMM model using the concatenated data and sequence lengths
    hmm_model.fit(X_concatenated.reshape(-1, 1), lengths=lengths)
    
    return hmm_model

def predict_hmm_states(hmm_model, X):
    """Predict HMM states for given sequences."""
    # Encode the sequences into integers
    X_encoded = [seq_to_numbers(seq) for seq in X]
    
    # Flatten the list of arrays into a single array
    X_concatenated = np.concatenate(X_encoded)
    
    # Get the lengths of each sequence
    lengths = [len(seq) for seq in X]
    
    # Predict the state sequence for the concatenated data
    state_sequence = hmm_model.predict(X_concatenated.reshape(-1, 1), lengths=lengths)
    
    # Split the predicted states back into their original sequences
    split_states = []
    idx = 0
    for length in lengths:
        split_states.append(state_sequence[idx:idx + length].tolist())
        idx += length
    
    return split_states

def extract_features(sequence, hmm_states):
    """Extract features including HMM state predictions and hydrophobicity."""
    hydrophobicity_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    features = []
    for i, (aa, state) in enumerate(zip(sequence, hmm_states)):
        hydrophobicity = hydrophobicity_scale.get(aa, 0)
        feature = {
            'aa': aa,
            'prev_aa': sequence[i-1] if i > 0 else 'START',
            'next_aa': sequence[i+1] if i < len(sequence)-1 else 'END',
            'hmm_state': str(state),
            'prev_hmm_state': str(hmm_states[i-1]) if i > 0 else 'START',
            'next_hmm_state': str(hmm_states[i+1]) if i < len(sequence)-1 else 'END',
            'hydrophobicity': str(hydrophobicity),
            'hydrophobicity_category': 'high' if hydrophobicity > 0 else 'low'
        }
        features.append(feature)
    return features

def train_crf(X_features, y):
    """Train a CRF model using sequence features, HMM-predicted states, and hydrophobicity."""
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    crf.fit(X_features, y)
    return crf

def evaluate_model(crf, X, y):
    
    # Predict labels using the trained CRF model
    y_pred = crf.predict(X)
    
    # Flatten the lists of sequences to make them suitable for classification metrics
    y_test_flat = [label for seq in y for label in seq]  # Flatten y_test (true labels)
    y_pred_flat = [label for seq in y_pred for label in seq]  # Flatten y_pred (predicted labels)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test_flat, y_pred_flat))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_flat, y_pred_flat))

    sequence_accuracies = []
    print(len(y))
    # For each sequence in the test set
    for true_seq, pred_seq in zip(y, y_pred):
        # Calculate per-sequence accuracy
        correct_predictions=0
        for i in range(len(true_seq)):
            if true_seq[i] == pred_seq[i]:
                correct_predictions = correct_predictions+1
        total_predictions = len(true_seq)
        sequence_accuracy = correct_predictions / total_predictions
        
        # Store the result
        sequence_accuracies.append(sequence_accuracy)
    
    # Optionally, sort sequences by accuracy to identify the worst-performing sequences
    sorted_sequences = sorted(zip(sequence_accuracies, y, y_pred), key=lambda x: x[0])
    #keys = []
    #exper = read_fasta("test_lab.fasta")
    # Display the worst performing sequences
    for i, (accuracy, true_seq, pred_seq) in enumerate(sorted_sequences):
        if i<5:
            print(f"Sequence {i+1}: Accuracy: {accuracy:.2f}")
            print(f"True labels: {true_seq}")
            print(f"Predicted labels: {pred_seq}")
            #keys.append(get_key_by_value(exper, original(true_seq)))
            print("----")
        else:
            break

   
def main():
    # Read sequences and labels from FASTA files
    X_train = list(read_fasta("training_seq.fasta").values())
    y_train = TMRelabel(list(read_fasta("training_lab.fasta").values()))
    
    # Split data into training and validation sets
    #X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.3, random_state=42)
    #print(X_train)

    X_val = list(read_fasta("test_seq.fasta").values())
    y_val = TMRelabel(list(read_fasta("test_lab.fasta").values()))

    # Step 1: Train HMM on the training data
    n_states = 4  # M, i, m, o
    hmm_model = train_hmm(X_train, n_states)

    # Step 2: Predict HMM states for both training and validation data
    hmm_train_states = predict_hmm_states(hmm_model, X_train)
    hmm_val_states = predict_hmm_states(hmm_model, X_val)

    # Step 3: Extract features including HMM state predictions and hydrophobicity
    X_train_features = [extract_features(seq, states) for seq, states in zip(X_train, hmm_train_states)]
    X_val_features = [extract_features(seq, states) for seq, states in zip(X_val, hmm_val_states)]

    # Step 4: Train CRF with HMM predictions and hydrophobicity as additional features
    crf_model = train_crf(X_train_features, y_train)

    # Step 5: Evaluate the CRF model on validation data
    evaluate_model(crf_model, X_val_features, y_val)

if __name__ == "__main__":
    main()