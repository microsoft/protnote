from flask import Flask, request, jsonify
import pandas as pd
import inference

app = Flask(__name__)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_embeddings(sequences):
    inferrer = inference.Inferrer(
        'cached_models/noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-13703706',
        use_tqdm=True, batch_size=64, activation_type="pooled_representation"
    )
    activations = inferrer.get_activations(sequences)
    
    # Convert coo_matrix to dense numpy array and then to a list
    dense_activations = [act.toarray().tolist() for act in activations]
    
    df = pd.DataFrame(dense_activations, index=sequences)
    return df

@app.route('/get_embeddings', methods=['POST'])
def get_embedding_route():
    protein_sequences = request.json['protein_sequences']  # Expecting a list of sequences
    embeddings_df = get_embeddings(protein_sequences)
    embeddings_json = embeddings_df.to_dict(orient='index')
    return jsonify({'embeddings': embeddings_json})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)