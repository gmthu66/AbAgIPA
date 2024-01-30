from utils.folding import get_sequence_dict


def infer_inters(sequences, truncate_sequences, fasta_file=None):
    if not isinstance(sequences, dict):
        seq_dict = get_sequence_dict(sequences, fasta_file,)
    else:
        seq_dict = sequences

    if truncate_sequences:
        try:
            from utils.abnumber_ import truncate_seq
        except:
            exit(
                "AbNumber not installed. Please install AbNumber to use truncation."
            )
        seq_dict = {k: truncate_seq(v) for k, v in seq_dict.items()}
