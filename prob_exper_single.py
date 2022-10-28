import pickle
import statistics
import time
from configparser import ConfigParser
from warnings import simplefilter

import numpy as np
import pandas as pd

import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

simplefilter("ignore", category=ConvergenceWarning)

# Read config file
configur = ConfigParser()
configur.read('config.prob')

# Todo code optimization


# Load language model
check_point = configur.get('parameter', 'check_point')
tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModel.from_pretrained(check_point, output_hidden_states=True)


def _index_of_entity(row_index, batch_1):
    sent = batch_1['sentence'][row_index]
    entity = batch_1['entity'][row_index]
    tokenized_sent = tokenizer.encode(sent, add_special_tokens=True, max_length=100, truncation=True)
    tokenized_entity = tokenizer.encode(entity, add_special_tokens=False)
    all_index_of_entity = []
    try:
        all_index_of_entity = [tokenized_sent.index(ind) for ind in tokenized_entity]
    except ValueError as ve:
        pass
    return all_index_of_entity


def _extract_token_rep(layer_rep, batch_1):
    single_sub_token = 0
    multi_sub_token = 0
    not_found = 0
    layer_token_rep = np.zeros((layer_rep.size()[0], layer_rep.size()[2]))
    for i in range(len(layer_rep)):
        row_id = i
        ent_sub_token_ind = _index_of_entity(row_id, batch_1)
        if len(ent_sub_token_ind) == 1:
            single_sub_token += 1
            token_index = ent_sub_token_ind[0]
            # token_rep = layer_token_rep[i][token_index] Big-Mistake
            token_rep = layer_rep[i][token_index]
            layer_token_rep[i] = token_rep

        elif len(ent_sub_token_ind) > 1:
            multi_sub_token += 1
            temp_rep = [layer_rep[i][sub_token_ind] for sub_token_ind in ent_sub_token_ind]
            # Average the sub-token representations
            token_rep = torch.mean(torch.stack(temp_rep), dim=0)
            layer_token_rep[i] = token_rep
        else:
            not_found += 1
    print('       _extract_token_rep(layer_rep, batch_1)')
    print('             single_sub_token', single_sub_token)
    print('             multi_sub_token', multi_sub_token)
    print('             not_found', not_found)

    return layer_token_rep


def _linear_classifier(train_features, train_labels, test_features, test_labels, classifier_model_identifier):
    f1_ls = []
    prediction_str = '\n'
    prediction_str += 'Gold_Label: '
    print('    len(test_labels)', len(test_labels))
    temp_test_labels = [str(item) for item in test_labels.tolist()]
    test_label_str = ' '.join(temp_test_labels)
    prediction_str += test_label_str + '\n'
    linear_num_run = int(configur.get('linear', 'linear_num_run'))
    prediction_str += 'Prediction over ' + str(linear_num_run) + ' Run' + '\n'
    for i in range(linear_num_run):
        lr_clf = LogisticRegression()
        lr_clf.fit(train_features, train_labels)
        # Save the model
        model_file_name = 'model_log/' + classifier_model_identifier + '_run_' + str(i) + '_' + '.pkl'
        with open(model_file_name, 'wb') as file:
            pickle.dump(lr_clf, file)
        y_pred = lr_clf.predict(test_features)
        temp_pred = y_pred.tolist()
        temp_pred_str = [str(item) for item in temp_pred]
        temp_pred_str = ' '.join(temp_pred_str)
        prediction_str += 'Run_' + str(i) + ': ' + temp_pred_str + '\n'
        # sys.exit()
        f1_result = f1_score(test_labels, y_pred, average='micro')
        # Add classification report
        prediction_str += 'classification_report \n'
        prediction_str += classification_report(test_labels, y_pred) + '\n'
        f1_ls.append(f1_result)
        print('             Raw f1_result @ ', i, ' ', f1_result)
    mean_f1 = round(statistics.mean(f1_ls), 2)
    std_dev = statistics.stdev(f1_ls)
    print('        mean_f1 returned ', mean_f1)
    print('        std_dev ', std_dev)
    return mean_f1, std_dev, prediction_str


def _mlp_classifier(train_features, train_labels, test_features, test_labels, classifier_model_identifier):
    f1_ls = []
    prediction_str = '\n'
    prediction_str += 'Gold_Label: '
    print('    len(test_labels)', len(test_labels))
    temp_test_labels = [str(item) for item in test_labels.tolist()]
    test_label_str = ' '.join(temp_test_labels)
    prediction_str += test_label_str + '\n'
    mlp_num_run = int(configur.get('mlp', 'mlp_num_run'))
    prediction_str += 'Prediction over ' + str(mlp_num_run) + ' Run' + '\n'
    mlp_iter = int(configur.get('mlp', 'mlp_iter'))
    mlp_hiddent_unit = int(configur.get('mlp', 'mlp_hiddent_unit'))
    mlp_activation = configur.get('mlp', 'mlp_activation')
    for i in range(mlp_num_run):
        mlp_clf = MLPClassifier(hidden_layer_sizes=(mlp_hiddent_unit,), max_iter=mlp_iter, activation=mlp_activation,
                                random_state=i)
        mlp_clf.fit(train_features, train_labels)
        # Save the model
        model_file_name = 'model_log/' + classifier_model_identifier + '_run_' + str(i) + '_' + '.pkl'
        with open(model_file_name, 'wb') as file:
            pickle.dump(mlp_clf, file)
        y_pred = mlp_clf.predict(test_features)
        temp_pred = y_pred.tolist()
        temp_pred_str = [str(item) for item in temp_pred]
        temp_pred_str = ' '.join(temp_pred_str)
        prediction_str += 'Run_' + str(i) + ': ' + temp_pred_str + '\n'
        # sys.exit()
        f1_result = f1_score(test_labels, y_pred, average='micro')
        # Add classification report
        prediction_str += 'classification_report \n'
        prediction_str += classification_report(test_labels, y_pred) + '\n'
        f1_ls.append(f1_result)
        print('             Raw f1_result @ ', i, ' ', f1_result)
    mean_f1 = round(statistics.mean(f1_ls), 2)
    std_dev = statistics.stdev(f1_ls)
    print('        mean_f1 returned ', mean_f1)
    print('        std_dev ', std_dev)
    return mean_f1, std_dev, prediction_str


def _read_data(_path):
    sample_size = int(configur.get('parameter', 'sample_size'))
    split_ratio = float(configur.get('parameter', 'split_ratio'))
    # Multi-class
    df = pd.read_csv(_path)
    # print(' len(df) ', len(df))
    df_class_0 = df[df['class'] == 0]
    print('len(df_class_0) ', len(df_class_0))
    df_class_1 = df[df['class'] == 1]
    print('len(df_class_1) ', len(df_class_1))
    df_class_2 = df[df['class'] == 2]
    print('len(df_class_2) ', len(df_class_2))

    # Sample from the three class randomly
    class_0_sample = df_class_0.sample(sample_size)
    print('     len(class_0_sample) ', len(class_0_sample))
    class_1_sample = df_class_1.sample(sample_size)
    print('     len(class_1_sample) ', len(class_1_sample))
    class_2_sample = df_class_2.sample(sample_size)
    print('     len(class_2_sample) ', len(class_2_sample))

    batch_1 = pd.concat([class_0_sample, class_1_sample, class_2_sample], ignore_index=True, sort=False)

    # Randomize
    batch_1 = batch_1.sample(frac=1).reset_index(drop=True)

    # Overwrite for replication
    print('over-written for replication')
    batch_1 = pd.read_csv(_path)

    # Save this batch to log directory
    train_index = int(split_ratio * len(batch_1))
    test_index = len(batch_1) - train_index
    split_ls = train_index * ['train']
    split_ls.extend(test_index * ['test'])
    batch_1_with_split = batch_1
    batch_1_with_split['split'] = split_ls
    file_name = _path.split('/')[1]
    batch_log_file_name = 'log/' + '_batch_' + file_name
    batch_1_with_split.to_csv(batch_log_file_name)
    print('Batch written to ', batch_log_file_name)
    return batch_1, batch_log_file_name


def _fixed_split(layer_token_rep, labels):
    split_ratio = float(configur.get('parameter', 'split_ratio'))
    print(' _fixed_split ')
    print('     $$$$$$ len(layer_token_rep) ', len(layer_token_rep))
    train_split = int(split_ratio * len(layer_token_rep))
    train_features = layer_token_rep[:train_split]
    test_features = layer_token_rep[train_split:]
    train_labels = labels[:train_split]
    test_labels = labels[train_split:]
    print('         len(test_features) ', len(test_labels))
    return [train_features, test_features, train_labels, test_labels]


def shuffle(train_features_l0):
    print('     *** shuffle')
    print(type(train_features_l0))
    print(train_features_l0.shape)
    np.random.shuffle(train_features_l0)
    return train_features_l0


def main():
    file_path = configur.get('parameter', 'dataset_path')
    print('file_path ', file_path)
    prediction_log = ''
    print('Processing ', file_path)
    batch_1, batch_log_file_name = _read_data(file_path)
    prediction_log += 'Prediction for ' + batch_log_file_name + '\n'

    print('     len(batch_1) ', len(batch_1))
    tokenized = batch_1['sentence'].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=100, truncation=True)))

    # Padding
    pad_length = configur.get('parameter', 'pad_length')
    max_len = int(pad_length)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

    # To ignore the padded values
    attention_mask = np.where(padded != 0, 1, 0)
    print('     attention_mask.shape ', attention_mask.shape)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    print('len(input_ids) ', len(input_ids))
    print('type(input_ids) ', type(input_ids))
    print('input_ids.shape ', input_ids.shape)
    print('len(attention_mask) ', len(attention_mask))
    print('type(attention_mask) ', type(attention_mask))
    print('attention_mask.shape ', attention_mask.shape)

    # Without batch
    print('     Running torch.no_grad() ')
    start_time1 = time.time()
    torch.cuda.empty_cache()
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    last_hidden_states_layer_rep = last_hidden_states[2]
    print('     Finished torch.no_grad() ')
    print("     Block torch.no_grad() took %s seconds " % (time.time() - start_time1))

    labels = batch_1['class']
    print('     len(labels) ', len(labels))

    all_layer_mean_f1 = []
    all_layer_std_deviation = []
    # Baseline with shuffled representation (layer 3)
    layer_rep = last_hidden_states_layer_rep[3]
    layer_token_rep = _extract_token_rep(layer_rep, batch_1)
    train_features_l0, test_features_l0, train_labels_l0, test_labels_l0 = _fixed_split(layer_token_rep,
                                                                                        labels)
    train_features_l0 = shuffle(train_features_l0)
    mean_f1, std_dev, f1_ls = _mlp_classifier(train_features_l0, train_labels_l0,
                                              test_features_l0,
                                              test_labels_l0, 'dont_save')
    all_layer_mean_f1.append(mean_f1)
    all_layer_std_deviation.append(std_dev)

    # Process layer wise
    for i in range(13):
        prediction_log += 'Layer: ' + str(i)
        print('  Processing layer ', i)
        layer_rep = last_hidden_states_layer_rep[i]
        # print('     type(layer_rep) ', type(layer_rep))
        print('     layer_rep.shape ', layer_rep.shape)
        layer_token_rep = _extract_token_rep(layer_rep, batch_1)
        # print('     type(layer_token_rep) ', type(layer_token_rep))
        print('     layer_token_rep.shape ', layer_token_rep.shape)

        # Using manual fixed split using the index
        train_features_l0, test_features_l0, train_labels_l0, test_labels_l0 = _fixed_split(layer_token_rep,
                                                                                            labels)

        classifier_model_identifier = 'layer_' + str(i)
        start_time1 = time.time()
        mean_f1 = 0.0
        std_dev = 0.0
        prediction_str = ''
        classifier = configur.get('parameter', 'classifier')
        if classifier == 'mlp':
            print('     Running ', classifier)
            mean_f1, std_dev, prediction_str = _mlp_classifier(train_features_l0, train_labels_l0,
                                                               test_features_l0,
                                                               test_labels_l0, classifier_model_identifier)
        elif classifier == 'linear':
            print('     Running ', classifier)
            mean_f1, std_dev, prediction_str = _linear_classifier(train_features_l0, train_labels_l0,
                                                                  test_features_l0,
                                                                  test_labels_l0, classifier_model_identifier)
        prediction_log += prediction_str + '\n'
        all_layer_mean_f1.append(mean_f1)
        all_layer_std_deviation.append(std_dev)
        # print('     Finish  MLP Classifier for ', test_file_path)
        print("     MLP Block took %s seconds " % (time.time() - start_time1))
    print(' len(all_layer_mean_f1) for ', file_path, len(all_layer_mean_f1))
    print(' len(all_layer_std_deviation) for ', file_path, len(all_layer_std_deviation))
    print(' Mean f1 for per layers for file ', file_path, all_layer_mean_f1)
    print(' Std deviation for  all layers for file ', file_path, all_layer_std_deviation)
    # Write the result to file
    result_file_path = 'log/' + '_result_log_' + file_path.split('/')[1]
    result_file_path = result_file_path.replace('.csv', '.txt')
    file_ob = open(result_file_path, 'w')
    content = 'Result for - ' + file_path + '\n'
    content += 'Mean f1 for all layers  \n'
    content += str(all_layer_mean_f1) + '\n'
    content += 'Std deviation for  all layers \n'
    content += str(all_layer_std_deviation) + '\n'
    file_ob.write(content)
    file_ob.close()
    print('Result log written to ', result_file_path)

    # Write the prediction log to file
    pred_file_path = 'log/' + '_pred_log_' + file_path.split('/')[1]
    pred_file_path = pred_file_path.replace('.csv', '.txt')
    file_ob = open(pred_file_path, 'w')
    file_ob.write(prediction_log)
    file_ob.close()
    print('Prediction log written to ', pred_file_path)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Running all sample took ", (time.time() - start_time))
