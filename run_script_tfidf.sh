#!/bin/sh
python build_tfidf_matrix.py dataset/wiki-pages-text/ output/index/
python generate_features.py ../train.json ../devset.json ../test-unlabelled.json output/index/wiki_processed_count_matrix_tfidf.npz dataset/wiki-pages-text/ output/index/wiki_pages_id_index.json
python classifier_train_test.py train_set_features.pickle test_set_features.pickle test_set_dict.json --is-test True