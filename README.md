# Git project example
1) Create dataset: python create_dataset.py
2) Create features pipelines:
 * Create features folder: mkdir data/features
 * Build lsa features: python build_features.py data/features/lsa_pipeline.pkl build-lsa
 * Build tf-idf features: python build_features.py data/features/tfidf_pipeline.pkl build-tfidf
3) Train model
 * Create model folder: mkdir models
 * Train sgd model with tf-idf features: python train_model.py data/raw/train_data.json data/features/tfidf_pipeline.pkl models/sgdtfidf.pkl train-sgd
 * NB model is also available and lsa_pipeline features
4) Evaluate model: python evaluate_model.py models/sgdtfidf.pkl data/raw/test_data.json