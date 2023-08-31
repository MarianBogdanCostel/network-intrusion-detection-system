# module imports
import argparse
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight


class DatasetProcessor:
    def __init__(self):
        self.dataset = None
        self.test_dataset = None
        self.train_dataset_name = "unsw_train_dataset"
        self.test_dataset_name = "unsw_test_dataset"

    def get_train_dataset_name(self):
        return self.train_dataset_name

    def get_test_dataset_name(self):
        return self.test_dataset_name

    def set_test_dataset(self, test_dataset):
        self.test_dataset = test_dataset

    def load_all_datasets_from_directory(self, directory_path):
        file_list = os.listdir(directory_path)
        for filename in file_list:
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                data = pd.read_csv(file_path)
                self.dataset = concatenate_datasets(self.dataset, data)
        return self.dataset

    def load_specific_dataset(self, dataset):
        csv_file_name = dataset + '.csv'
        if os.path.exists(csv_file_name):
            return pd.read_csv(csv_file_name)
        else:
            print(f"No filename {csv_file_name} found!")
            return None

    def save_dataset_locally(self, dataset, file_name):
        csv_file_name = file_name + '.csv'
        dataset.to_csv(csv_file_name, index=False)
        print('Saved dataset locally as ', csv_file_name)

    def entry_type(self, dataset):
        normal_entries = dataset[dataset['label'] == 0]
        attack_entries = dataset[dataset['label'] == 1]

        return normal_entries, attack_entries

    def print_all_attacks_category(self, dataset):
        print('All ATTACKS\n ', dataset['attack_cat'].value_counts())

    def rename_columns(self):
        self.dataset.rename(columns={' Label': 'attack_cat'}, inplace=True)
        self.dataset['label'] = self.dataset['attack_cat'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    def data_normalization(self, features):
        index = features.index
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(features)
        normalized_features = scaler.transform(features)
        normalized_features_data = pd.DataFrame(normalized_features, columns=features.columns, index=index)
        return normalized_features_data

    def split_training_dataset(self, dataset):
        # Initialize an empty list to store selected attack records
        for_training = pd.DataFrame(columns=dataset.columns)
        for_testing = pd.DataFrame(columns=dataset.columns)

        percentage = 0.8

        # self.dataset['attack_cat'] = attack_category
        # plt.figure(figsize=(8, 8))
        # plt.pie(self.dataset['attack_cat'].value_counts(), labels=self.dataset['attack_cat'].unique(), autopct='%0.2f%%')
        # plt.title('Pie chart distribution of multi-class labels')
        # plt.legend(loc='best')
        # plt.show()

        normal_entries, attack_entries = self.entry_type(dataset)
        print('NORMAL\n', normal_entries['attack_cat'].value_counts())
        print('ATTACKS\n', attack_entries['attack_cat'].value_counts())

        attack_categories = attack_entries['attack_cat'].unique()
        for category in attack_categories:
            attack_category = attack_entries[attack_entries['attack_cat'] == category]
            total_cat_attacks = len(attack_category)
            selected_percentage = int(total_cat_attacks * percentage)
            selected_for_training = attack_category.sample(n=selected_percentage, random_state=42)
            remaining_for_testing = attack_category.drop(selected_for_training.index)
            for_training = concatenate_datasets(for_training, selected_for_training)
            for_testing = concatenate_datasets(for_testing, remaining_for_testing)

        print('Only attacks for training\n', for_training['attack_cat'].value_counts())
        print('Only attacks for testing\n', for_testing['attack_cat'].value_counts())

        training_attacks_total = len(for_training)
        testing_attacks_total = len(for_testing)
        normal_entries_total = len(normal_entries)
        print("SUM only attacks training:\n", training_attacks_total)
        print("SUM only attacks testing:\n", testing_attacks_total)
        print("SUM normal:\n", normal_entries_total)

        if normal_entries_total < training_attacks_total:
            normal_entries_split = int(normal_entries_total * percentage)
            normal_training = normal_entries[:normal_entries_split]
            normal_testing = normal_entries[normal_entries_split:]
            for_training = concatenate_datasets(for_training, normal_training)
            for_testing = concatenate_datasets(for_testing, normal_testing)
        else:
            normal_training = normal_entries[:training_attacks_total]
            normal_entries = normal_entries.drop(normal_training.index)
            normal_testing = normal_entries[:testing_attacks_total]
            for_training = concatenate_datasets(for_training, normal_training)
            for_testing = concatenate_datasets(for_testing, normal_testing)

        print('Categories for training\n', for_training['attack_cat'].value_counts())
        print('Categories for testing\n', for_testing['attack_cat'].value_counts())

        for_training = shuffle(for_training, random_state=42)
        for_testing = shuffle(for_testing, random_state=42)
        self.save_dataset_locally(for_training, 'unsw_train_dataset')
        self.save_dataset_locally(for_testing, 'unsw_test_dataset')
        return for_training, for_testing

    def encode_categorical_features(self, dataset):
        categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [cat for cat in categorical_columns if cat != "attack_cat"]
        print('Categorical features\n', categorical_columns)
        # Apply label encoding to categorical columns
        for column in categorical_columns:
            dataset[column] = LabelEncoder().fit_transform(dataset[column])
        return dataset

    # Drop any rows with missing values (NaNs) and duplicate entries (if any)
    def remove_unnecessary_records(self, dataset):
        dataset['service'].replace('-', np.nan, inplace=True)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True)
        dataset.drop_duplicates(keep="first")
        dataset.drop('id', axis=1, inplace=True)
        return dataset

    def preprocess_dataset(self, dataset):
        if dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        print('ALL BEFORE\n', dataset['attack_cat'].value_counts())
        dataset = self.remove_unnecessary_records(dataset)
        print('ALL AFTER \n', dataset['attack_cat'].value_counts())

        print('DATASET Check\n', dataset.head(10))
        attack_cat_column = dataset['attack_cat']
        label_column = dataset['label']

        dataset = self.encode_categorical_features(dataset)
        features = dataset.drop(['label', 'attack_cat'], axis=1)
        dataset = self.data_normalization(features)
        dataset['attack_cat'] = attack_cat_column
        dataset['label'] = label_column
        print('DATASET Check\n', dataset.head(10))

        return dataset

    def separate_features_and_target(self, dataset, test_dataset):
        if dataset is None or test_dataset is None:
            print('No dataset or test_dataset provided\n')
            return None

        print("START\n", dataset.head(10))
        print("START\n", test_dataset.head(25))

        x_train = dataset.drop(['label', 'attack_cat'], axis=1)
        y_train_binary = dataset['label']
        y_train_multiclass = dataset['attack_cat']
        y_train_binary = y_train_binary.astype('int32')

        x_test = test_dataset.drop(['label', 'attack_cat'], axis=1)
        y_test_binary = test_dataset['label']
        y_test_multiclass = test_dataset['attack_cat']
        y_test_binary = y_test_binary.astype('int32')

        print("X_train\n", x_train.head(25))
        print(y_train_binary.value_counts())
        print(y_train_multiclass.value_counts())

        print("X_test\n", x_test.head(25))
        print(y_test_binary.value_counts())
        print(y_test_multiclass.value_counts())

        print('Categories for training\n', dataset['attack_cat'].value_counts())
        print('Categories for testing\n', test_dataset['attack_cat'].value_counts())

        return x_train, x_test, y_train_binary, y_test_binary, y_train_multiclass, y_test_multiclass

    def separate_data_for_second_classifier(self, data):
        x_test = data.drop(['label', 'predicted', 'attack_cat'], axis=1)
        y_test_binary = data['label']
        y_test_first_predicted = data['predicted']
        y_test_multiclass = data['attack_cat']
        y_test_binary = y_test_binary.astype('int32')

        print("X_test\n", x_test.head(25))
        print(y_test_binary.value_counts())
        print(y_test_multiclass.value_counts())

        print('All categories in possible attacks\n', data['attack_cat'].value_counts())

        return x_test, y_test_binary, y_test_first_predicted, y_test_multiclass

    def train_svm_classifier(self, x_train, y_train, x_test, y_test, file_name):
        kernel = 'poly'
        C = 1
        degree = 3
        gamma = 'scale'
        class_weight = {0: 3.99, 1: 1.35}

        classifier = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, verbose=True)
        print('Training\nSVC: kernel={}, C={}, degree={}, gamma={}, class_weight={}'
              .format(kernel, C, degree, gamma, class_weight))
        classifier.fit(x_train, y_train)

        # Save the classifier
        self.save_classifier(classifier, file_name)
        svm_predictions = self.evaluate_binary_classifier(classifier, x_test, y_test)
        print("CONCATENATED Results\n", svm_predictions.head(25))

    def train_random_forest_classifier(self, x_train, y_train, x_test, y_test, file_name):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_test), y=y_test)
        class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_test), class_weights)}
        # random_forest_classifier = RandomForestClassifier(class_weight=class_weight_dict, n_estimators=100, random_state=42, verbose=True)
        random_forest_classifier = RandomForestClassifier(class_weight=class_weight_dict, n_estimators=100,
                                                          random_state=42,
                                                          verbose=True)
        print("Training")
        random_forest_classifier.fit(x_train, y_train)
        # Save the classifier
        self.save_classifier(random_forest_classifier, file_name)
        self.evaluate_multiclass_classifier(random_forest_classifier, x_test, y_test)

    def encode_multiclass_values(self, y_train, y_test):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        return y_train, y_test

    def both_classifiers(self, first_classifier_input, second_classifier_input, x_test, y_test_binary,
                         y_test_multiclass):
        first_classifier = self.load_classifier(first_classifier_input)

        svm_predictions = self.evaluate_binary_classifier(first_classifier, x_test, y_test_binary)
        print("SVM Classifier Results\n", svm_predictions.head(25))

        print("Actual numbers\n", svm_predictions['label'].value_counts())
        print("Predicted numbers\n", svm_predictions['predicted'].value_counts())
        print('Total instances\n', len(svm_predictions))

        concatenated_dataset = pd.concat([svm_predictions, y_test_multiclass], axis=1)
        print("Concatenated dataset\n", concatenated_dataset.head(25))

        possible_attacks = concatenated_dataset[concatenated_dataset['predicted'] == 1]
        print("Possible attacks\n", possible_attacks.head(25))
        print("Actual numbers\n", possible_attacks['label'].value_counts())
        print("Predicted numbers\n", possible_attacks['predicted'].value_counts())
        print('Total\n', len(possible_attacks))
        self.save_dataset_locally(possible_attacks, "unsw_possible_attacks")

        print('Old x_test\n', x_test.head(25))

        x_test, y_test_binary, y_test_first_predicted, y_test_multiclass = \
            self.separate_data_for_second_classifier(possible_attacks)

        print('New x_test\n', x_test.head(25))

        second_classifier = self.load_classifier(second_classifier_input)
        final_predictions = self.evaluate_multiclass_classifier(second_classifier, x_test, y_test_multiclass)

        print("RF Classifier Results\n", final_predictions.head(25))

        print("Actual numbers\n", final_predictions['actual'].value_counts())
        print("Predicted numbers\n", final_predictions['final_prediction'].value_counts())
        print('Total instances\n', len(final_predictions))

        self.save_dataset_locally(final_predictions, "unsw_final")

    def save_classifier(self, classifier, file_name):
        saved_file = file_name + '.joblib'
        dump(classifier, saved_file)
        print(f"Trained classifier saved as '{saved_file}'")

    def load_classifier(self, classifier_name):
        full_classifier_name = classifier_name + '.joblib'
        classifier = load(full_classifier_name)
        return classifier

    def predict_classifier(self, classifier, x_test):
        predictions = classifier.predict(x_test)
        return predictions

    # de adaugat X_test, X_test_clean si Y_test_binary ca parametri la metoda
    def evaluate_binary_classifier(self, classifier, x_test, y_test_binary):
        # df = pandas.DataFrame(x_test)
        # print('COLUMNS\n',self.dataset.columns)
        # print(x_test.columns)
        # x_test.columns = self.dataset.columns.drop(['attack_cat', 'label'])
        # print(x_test.columns)
        predictions = self.predict_classifier(classifier, x_test)
        print('Calculating prediction')
        accuracy = accuracy_score(y_test_binary, predictions) * 100
        precision = precision_score(y_test_binary, predictions) * 100
        recall = recall_score(y_test_binary, predictions) * 100
        f1 = f1_score(y_test_binary, predictions) * 100
        confusion_mat = confusion_matrix(y_test_binary, predictions)

        print("Accuracy: {:.2f}%".format(accuracy))
        print("Precision: {:.2f}%".format(precision))
        print("Recall: {:.2f}%".format(recall))
        print("F1 score: {:.2f}%".format(f1))

        # target_names = self.Y_test.unique()
        # Print the confusion matrix
        print("Confusion Matrix:\n", confusion_mat)
        # print(pd.DataFrame(confusion_mat, columns=target_names, index=target_names))
        class_name = ['Class 0', 'Class 1']
        class_report = classification_report(y_test_binary, predictions, target_names=class_name)
        print("Classification Report:\n", class_report)

        results = pd.DataFrame({'label': y_test_binary, 'predicted': predictions})
        print(results.head(25))

        print("Actual numbers\n", results['label'].value_counts())
        print("Predicted numbers\n", results['predicted'].value_counts())

        X_test_results = pd.DataFrame(x_test, columns=self.test_dataset.columns[:-2])
        print(X_test_results.head(25))
        X_test_results = pd.concat([X_test_results, results], axis=1)
        print(X_test_results.head(25))

        diff = sum(
            1 for actual, predicted in zip(X_test_results['label'], X_test_results['predicted']) if actual != predicted)
        print("Incorrect classified instances:\n", diff)
        return X_test_results

        # print("CONCATENATED Results\n", concatenated_results.head(10))
        #
        # print("Actual numbers\n", concatenated_results['label'].value_counts())
        # print("Predicted numbers\n", concatenated_results['predicted'].value_counts())
        #
        # # results['attack_cat'] = self.Y_test_multiclass
        # # print('After attack cat\n', results.head(20))
        #
        # incorrectly_classified = concatenated_results[concatenated_results['label'] != concatenated_results['predicted']]
        # print("Ceva\n", incorrectly_classified['label'].value_counts())
        # print('Number of incorrectly classified instances\n', len(incorrectly_classified))
        #
        # print("Ceva\n", incorrectly_classified['predicted'].value_counts())

        # return concatenated_results

    def evaluate_multiclass_classifier(self, classifier, x_test, y_test_multiclass):
        predictions = self.predict_classifier(classifier, x_test)
        accuracy = accuracy_score(y_test_multiclass, predictions) * 100
        confusion_mat = confusion_matrix(y_test_multiclass, predictions)
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Confusion Matrix:\n", confusion_mat)

        class_name = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance',
                      'Worms']
        print('Columns', class_name)

        class_report = classification_report(y_test_multiclass, predictions, target_names=class_name)
        print("Classification Report:\n", class_report)

        # results = self.actual_predicted_results(self.X_test_clean, self.Y_test_multiclass, predictions)
        # return results

        results = pd.DataFrame({'actual': y_test_multiclass, 'final_prediction': predictions})
        print(results.head(25))

        print("Actual numbers\n", results['actual'].value_counts())
        print("Final predicted numbers\n", results['final_prediction'].value_counts())

        X_test_results = pd.DataFrame(x_test, columns=self.test_dataset.columns[:-2])
        print(X_test_results.head(25))
        X_test_results = pd.concat([X_test_results, results], axis=1)
        print(X_test_results.head(25))

        diff = sum(1 for actual, predicted in zip(X_test_results['actual'], X_test_results['final_prediction'])
                   if actual != predicted)
        print("Incorrect classified:\n", diff)

        return X_test_results

    def actual_predicted_results(self, clean_dataset, actual, predicted):
        results = pd.DataFrame(clean_dataset, columns=clean_dataset.columns, index=clean_dataset.index)
        results['actual'] = actual
        results['predicted'] = predicted
        # print(results.head(20))
        return results

    def scale_features(self, x_train, x_test):
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test


# display the entire columns
def display_all_columns():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.expand_frame_repr', False)


def measure_time(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


def concatenate_datasets(*datasets):
    return pd.concat(datasets, ignore_index=True)


def train(args):
    print('Classifiers name:', svm_classifier, rf_classifier)

    dataset_processor = DatasetProcessor()
    dataset = dataset_processor.load_all_datasets_from_directory(file_path)
    dataset_preprocessed = dataset_processor.preprocess_dataset(dataset)
    dataset, test_dataset = dataset_processor.split_training_dataset(dataset_preprocessed)
    dataset_processor.set_test_dataset(test_dataset)
    X_train, X_test, Y_train_binary, Y_test_binary, Y_train_multiclass, Y_test_multiclass = \
        dataset_processor.separate_features_and_target(dataset, test_dataset)

    if args.classifier == 'both':
        dataset_processor.train_svm_classifier(X_train, Y_train_binary, X_test, Y_test_binary, svm_classifier)
        dataset_processor.train_random_forest_classifier(X_train, Y_train_multiclass, X_test, Y_test_multiclass,
                                                         rf_classifier)
        print("Trained both classifiers")
    elif args.classifier == 'svm':
        dataset_processor.train_svm_classifier(X_train, Y_train_binary, X_test, Y_test_binary, svm_classifier)
        print("Trained SVM classifier")
    elif args.classifier == 'rf':
        dataset_processor.train_random_forest_classifier(X_train, Y_train_multiclass, X_test, Y_test_multiclass,
                                                         rf_classifier)
        print("Trained Random Forest classifier")


def test(args):
    print('Classifiers name:', svm_classifier, rf_classifier)

    dataset_processor = DatasetProcessor()
    dataset = dataset_processor.load_specific_dataset(dataset_processor.get_train_dataset_name())
    test_dataset = dataset_processor.load_specific_dataset(dataset_processor.get_test_dataset_name())
    dataset_processor.set_test_dataset(test_dataset)
    X_train, X_test, Y_train_binary, Y_test_binary, Y_train_multiclass, Y_test_multiclass = \
        dataset_processor.separate_features_and_target(dataset, test_dataset)
    if args.classifier == 'both':
        dataset_processor.both_classifiers(svm_classifier, rf_classifier, X_test, Y_test_binary, Y_test_multiclass)
        print('Tested using both classifiers')
    elif args.classifier == 'svm':
        loaded_svm_classifier = dataset_processor.load_classifier(svm_classifier)
        dataset_processor.evaluate_binary_classifier(loaded_svm_classifier, X_test, Y_test_binary)
        print('Tested SVM classifier')
    elif args.classifier == 'rf':
        loaded_rf_classifier = dataset_processor.load_classifier(rf_classifier)
        dataset_processor.evaluate_multiclass_classifier(loaded_rf_classifier, X_test, Y_test_multiclass)
        print('Tested RF classifier')


def main():
    parser = argparse.ArgumentParser(description="Train, load, or test classifiers")
    subparsers = parser.add_subparsers(dest='subparser_name', help='sub-command help')

    # Subparser for training and testing
    train_parser = subparsers.add_parser('train', help='Train classifier')
    train_parser.add_argument('--classifier', type=str, choices=['svm', 'rf', 'both'], required=True,
                              help='Choose a classifier: svm, rf, or both')
    train_parser.set_defaults(func=train)

    # Subparser for testing
    test_parser = subparsers.add_parser('test', help='Test classifiers')
    test_parser.add_argument('--classifier', type=str, choices=['svm', 'rf', 'both'], required=True,
                             help='Choose a classifier: svm, rf, or both')
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


file_path = '../KDD-NSL/UNSW-NB15/both/'
svm_classifier = 'svm_classifier'
rf_classifier = 'rf_classifier'

if __name__ == "__main__":
    display_all_columns()
    main()

# # Example usage:
# if __name__ == "__main__":
#     display_all_columns()
#
#     # Binary classifier for unsw
#     dataset_processor = DatasetProcessor()
#     dataset = dataset_processor.load_all_datasets_from_directory(file_path)
#     dataset_preprocessed = dataset_processor.preprocess_dataset(dataset)
#     dataset, test_dataset = dataset_processor.split_training_dataset(dataset_preprocessed)
#     X_train, X_test, Y_train_binary, Y_test_binary, Y_train_multiclass, Y_test_multiclass = \
#         dataset_processor.separate_features_and_target(dataset, test_dataset)
#     start_time_1 = datetime.now()
#     dataset_processor.train_svm_classifier(X_train, Y_train_binary, X_test, Y_test_binary, svm_classifier)
#     end_time_1 = datetime.now()
#     measure_time(start_time_1, end_time_1)
#
#     dataset = dataset_processor.load_specific_dataset("unsw_train_dataset")
#     test_dataset = dataset_processor.load_specific_dataset("unsw_test_dataset")
#     X_train, X_test, Y_train_binary, Y_test_binary, Y_train_multiclass, Y_test_multiclass = \
#         dataset_processor.separate_features_and_target(dataset, test_dataset)
#     dataset_processor.both_classifiers(svm_classifier, rf_classifier, test_dataset)
#
#     # start_time_2 = datetime.now()
#     # # dataset_processor.train_svm_classifier(X_train, Y_train_binary, X_test, Y_test_binary, svm_classifier)
#     # dataset_processor.train_random_forest_classifier(X_train, Y_train_multiclass, X_test, Y_test_multiclass, rf_classifier)
#     # end_time_2 = datetime.now()
#     # measure_time(start_time_2, end_time_2)
#
#     start_time_3 = datetime.now()
#     dataset_processor.both_classifiers(svm_classifier, rf_classifier, test_dataset)
#     end_time_3 = datetime.now()
#     measure_time(start_time_3, end_time_3)
#
#     # loaded_svm_classifier = dataset_processor.load_classifier(svm_classifier)
#     # dataset_processor.evaluate_binary_classifier(loaded_svm_classifier)
#
#     # loaded_rf_classifier = dataset_processor.load_classifier(rf_classifier)
#     # dataset_processor.evaluate_multiclass_classifier(loaded_rf_classifier)
