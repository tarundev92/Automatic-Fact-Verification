import argparse
import helper
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import random
import time


def multinomial_naive_bayes_classifier(X_train, y_train, X_valid, y_valid=None, valid_dict=None, is_test=False, filename=None):
    print("---------------------multinomial_naive_bayes_classifier--------------------------")

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    temp_time = time.time()
    print("Predicting")
    y_pred = classifier.predict(X_valid)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    print("Predicting done")


    y_index = 0
    for key, value in valid_dict.items():
        value["label"] = y_pred[y_index]
        y_index += 1
    helper.save_dict_json(valid_dict, filename)

    print("---------------------multinomial_naive_bayes_classifier Done--------------------------")


def random_forest_classifier(X_train, y_train, X_valid, y_valid=None, valid_dict=None, is_test=False, filename=None):
    print("---------------------RandomForestClassifier--------------------------")
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)

    temp_time = time.time()
    print("Predicting")
    y_pred = classifier.predict(X_valid)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    print("Predicting done")

    y_index = 0
    for key, value in valid_dict.items():
        value["label"] = y_pred[y_index]
        y_index += 1
    helper.save_dict_json(valid_dict, filename)

    print("---------------------RandomForestClassifier Done--------------------------")



def parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_features_path', type=str, help='location of train features pickle file')
    parser.add_argument('dev_test_features_path', type=str, help='location of dev/test features pickle file')
    parser.add_argument('dev_test_dict_path', type=str, help='location of dev/test dictionary json file')
    parser.add_argument('--is-test', type=bool, default=False,
                        help='True if dataset path is test')
    return parser.parse_args()


def even_sample(ind_features, class_label):
    random.seed(42)
    zipped = zip(ind_features, class_label)
    zipped = sorted(zipped, key=lambda x:x[1])
    label_count = Counter(class_label)
    not_enough_info_all = zipped[:label_count["NOT ENOUGH INFO"]]

    refutes_all = zipped[label_count["NOT ENOUGH INFO"]:label_count["NOT ENOUGH INFO"]+label_count["REFUTES"]]
    supports_all = zipped[label_count["NOT ENOUGH INFO"]+label_count["REFUTES"]:]
    sample_count = min(list(label_count.values()))
    not_enough_info_sample = random.sample(not_enough_info_all, sample_count)
    refutes_sample = random.sample(refutes_all, sample_count)
    supports_sample = random.sample(supports_all, int(sample_count*1.5))
    final_sample = supports_sample + refutes_sample + not_enough_info_sample
    unzipped = list(zip(*final_sample))
    return unzipped[0], unzipped[1]
    


if __name__ == "__main__":
    start_time = time.time()
    print("Start Time: ", time.strftime('%H:%M%p %Z on %b %d, %Y'))
    args = parsed_arguments()


    vectorizer = DictVectorizer(sparse=True)
    print("-----Loading train set-----")
    temp_time = time.time()
    train_set_features = helper.load_dict_pickle(args.train_features_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    print("---Vectorizing train set---")
    temp_time = time.time()

    # X_train, y_train = even_sample(train_set_features["X_unigram_features"], train_set_features["y"])
    X_train, y_train = train_set_features["X_unigram_features"], train_set_features["y"]
    print("ini type merge keys len", type(X_train[0]))
    print("ini merge keys len", len(X_train[0].keys()))
    X_train = vectorizer.fit_transform(X_train)

    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))


    print("-----Loading validation set-----")
    temp_time = time.time()
    validation_set_features = helper.load_dict_pickle(args.dev_test_features_path)
    validation_dict = helper.load_dict_json(args.dev_test_dict_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    print("---Vectorizing validation set---")
    temp_time = time.time()
    X_valid = vectorizer.transform(validation_set_features["X_unigram_features"])
    y_valid = None
    result_fname = "test_set.json"
    if not args.is_test:
        result_fname = "dev_set.json"
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))


    print("---------------------Unigram Features Result---------------------")
    print("Multinomial Naive Bayes for label")
    temp_time = time.time()
    multinomial_naive_bayes_classifier(X_train, y_train, X_valid, y_valid=y_valid, valid_dict=validation_dict,
                                       is_test=args.is_test, filename="mnb_unigram_" + result_fname)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    print("Random Forest for label")
    temp_time = time.time()
    random_forest_classifier(X_train, y_train, X_valid, y_valid=y_valid, valid_dict=validation_dict,
                                      is_test=args.is_test, filename="rndf_unigram_"+result_fname)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))


    print()
    print("---------------------bigram Features Result---------------------")
    vectorizer = DictVectorizer(sparse=True)

    X_train, y_train = even_sample(train_set_features["X_bigram_features"], train_set_features["y"])
    # X_train, y_train = train_set_features["X_bigram_features"], train_set_features["y"]
    X_train = vectorizer.fit_transform(X_train)

    X_valid = vectorizer.transform(validation_set_features["X_bigram_features"])

    print("Multinomial Naive Bayes for label")
    temp_time = time.time()
    multinomial_naive_bayes_classifier(X_train, y_train, X_valid, y_valid=y_valid, valid_dict=validation_dict,
                                       is_test=args.is_test, filename="mnb_bigram_" + result_fname)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))


    print("Random Forest for label")
    temp_time = time.time()
    random_forest_classifier(X_train, y_train, X_valid, y_valid=y_valid, valid_dict=validation_dict,
                             is_test=args.is_test, filename="rndf_bigram_"+result_fname)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    print("-----Total time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
