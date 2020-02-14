-Requirements are in requirements.txt file

-run_script_tfidf.sh and run_script_elastic.sh expects the wiki-pages-text folder be placed inside dataset folder.

-For elastic search, Operating System dependent server package need to be downloaded and the local server need to be started before running run_script_elastic.sh file.

-Both the scripts file will generate four following files namely
  -rndf_unigram_test_set.json for unigram feature set using Random Forest Classifier
  -rndf_bigram_test_set.json for bigram feature set using Random Forest Classifier
  -mnb_unigram_test_set.json for unigram feature set using Multinomial Naive Bayes Classifier
  -mnb_bigram_test_set.json for bigram feature set using Multinomial Naive Bayes Classifier

 

