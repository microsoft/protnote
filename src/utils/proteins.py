

import functools
import numpy as np


# Map AA characters to their index in VOCAB.
_RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(VOCAB)}

def residues_to_indices(amino_acid_residues):
  return [_RESIDUE_TO_INT[c] for c in amino_acid_residues]


@functools.lru_cache(maxsize=1)
def _build_one_hot_encodings():
  """Create array of one-hot embeddings.

  Row `i` of the returned array corresponds to the one-hot embedding of amino
    acid VOCAB[i].

  Returns:
    np.array of shape `[len(VOCAB), 20]`.
  """
  base_encodings = np.eye(len(AMINO_ACID_VOCABULARY))
  to_aa_index = AMINO_ACID_VOCABULARY.index

  special_mappings = {
      'B':
          .5 *
          (base_encodings[to_aa_index('D')] + base_encodings[to_aa_index('N')]),
      'Z':
          .5 *
          (base_encodings[to_aa_index('E')] + base_encodings[to_aa_index('Q')]),
      'X':
          np.ones(len(AMINO_ACID_VOCABULARY)) / len(AMINO_ACID_VOCABULARY),
      _PFAM_GAP_CHARACTER:
          np.zeros(len(AMINO_ACID_VOCABULARY)),
  }
  special_mappings['U'] = base_encodings[to_aa_index('C')]
  special_mappings['O'] = special_mappings['X']
  special_encodings = np.array(
      [special_mappings[c] for c in _ADDITIONAL_AA_VOCABULARY])
  return np.concatenate((base_encodings, special_encodings), axis=0)


def residues_to_one_hot(amino_acid_residues):
  """Given a sequence of amino acids, return one hot array.

  Supports ambiguous amino acid characters B, Z, and X by distributing evenly
  over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].

  Supports rare amino acids by appropriately substituting. See
  normalize_sequence_to_blosum_characters for more information.

  Supports gaps and pads with the '.' and '-' characters; which are mapped to
  the zero vector.

  Args:
    amino_acid_residues: string. consisting of characters from
      AMINO_ACID_VOCABULARY

  Returns:
    A numpy array of shape (len(amino_acid_residues),
     len(AMINO_ACID_VOCABULARY)).

  Raises:
    KeyError: if amino_acid_residues has a character not in VOCAB.
  """
  residue_encodings = _build_one_hot_encodings()
  int_sequence = residues_to_indices(amino_acid_residues)
  return residue_encodings[int_sequence]


def fasta_indexer():
  """Get a function for converting tokenized protein strings to indices."""
  mapping = tf.constant(VOCAB)
  table = contrib_lookup.index_table_from_tensor(mapping)

  def mapper(residues):
    return tf.ragged.map_flat_values(table.lookup, residues)

  return mapper


def fasta_encoder():
  """Get a function for converting indexed amino acids to one-hot encodings."""
  encoded = residues_to_one_hot(''.join(VOCAB))
  one_hot_embeddings = tf.constant(encoded, dtype=tf.float32)

  def mapper(residues):
    return tf.ragged.map_flat_values(
        tf.gather, indices=residues, params=one_hot_embeddings)

  return mapper


def in_graph_residues_to_onehot(residues):
  """Performs mapping in `residues_to_one_hot` in-graph.

  Args:
    residues: A tf.RaggedTensor with tokenized residues.

  Returns:
    A tuple of tensors (one_hots, row_lengths):
      `one_hots` is a Tensor<shape=[None, None, len(AMINO_ACID_VOCABULARY)],
                             dtype=tf.float32>
       that contains a one_hot encoding of the residues and pads out all the
       residues to the max sequence length in the batch by 0s.
       `row_lengths` is a Tensor<shape=[None], dtype=tf.int32> with the length
       of the unpadded sequences from residues.

  Raises:
    tf.errors.InvalidArgumentError: if `residues` contains a token not in
    `VOCAB`.
  """
  ragged_one_hots = fasta_encoder()(fasta_indexer()(residues))
  return (ragged_one_hots.to_tensor(default_value=0),
          tf.cast(ragged_one_hots.row_lengths(), dtype=tf.int32))


def calculate_bucket_batch_sizes(bucket_boundaries, max_expected_sequence_size,
                                 largest_batch_size):
  """Calculated batch sizes for each bucket given a set of boundaries.

  Sequences in the smallest sized bucket will get a batch_size of
  largest_batch_size and larger buckets will have smaller batch sizes  in
  proportion to their maximum sequence length to ensure that they do not use too
  much memory.

  E.g. for bucket_boundaries of [5, 10, 20, 40], max_expected_size of 100
  and largest_batch_size of 50, expected_bucket_sizes are [50, 25, 12, 6, 2].

  Args:
    bucket_boundaries: list of positions of bucket boundaries
    max_expected_sequence_size: largest expected sequence, used to calculate
      sizes
    largest_batch_size: batch_size for largest batches.

  Returns:
    batch_sizes as list
  """
  first_max_size = bucket_boundaries[0]
  bucket_relative_batch_sizes = [
      (first_max_size / x)
      for x in bucket_boundaries + [max_expected_sequence_size]
  ]
  bucket_absolute_batch_sizes = [
      int(x * largest_batch_size) for x in bucket_relative_batch_sizes
  ]
  if min(bucket_absolute_batch_sizes) == 0:
    raise ValueError(
        'There would be a batch size of 0 during bucketing, which is not '
        'allowed. Bucket boundaries passed in were: %s, leading to batch sizes of: %s'
        % (bucket_boundaries, bucket_absolute_batch_sizes))
  return bucket_absolute_batch_sizes


def batch_iterable(iterable, batch_size):
  """Yields batches from an iterable.

  If the number of elements in the iterator is not a multiple of batch size,
  the last batch will have fewer elements.

  Args:
    iterable: a potentially infinite iterable.
    batch_size: the size of batches to return.

  Yields:
    array of length batch_size, containing elements, in order, from iterable.

  Raises:
    ValueError: if batch_size < 1.
  """
  if batch_size < 1:
    raise ValueError(
        'Cannot have a batch size of less than 1. Received: {}'.format(
            batch_size))

  current = []
  for item in iterable:
    if len(current) == batch_size:
      yield current
      current = []
    current.append(item)

  # Prevent yielding an empty batch. Instead, prefer to end the generation.
  if current:
    yield current


def pad_one_hot(one_hot, length):
  if length < one_hot.shape[0]:
    raise ValueError("The padding value must be longer than the one-hot's 0th "
                     'dimension. Padding value is ' + str(length) + ' '
                     'and one-hot shape is ' + str(one_hot.shape))
  padding = np.zeros((length - one_hot.shape[0], len(AMINO_ACID_VOCABULARY)))
  return np.append(one_hot, padding, axis=0)