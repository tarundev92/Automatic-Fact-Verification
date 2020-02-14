#!/bin/sh
python build_elastic.py dataset/wiki-pages-text/ output/elastic
python generate_features_elastic.py ../train.json ../devset.json ../test-unlabelled.json dataset/wiki-pages-text/ output/elastic/wiki_pages_id_index.json
python classifier_train_test.py train_set_features.pickle test_set_features.pickle test_set_dict.json --is-test True