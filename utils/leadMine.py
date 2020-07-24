from nerds.input.brat import BratInput
import csv
import os
import copy
from sklearn.metrics import classification_report, accuracy_score
from utils.convert2D import Convert2D

if __name__ == '__main__':
    folder_name = r'/media/druv022/Data1/Masters/Thesis/LeadMine/leadmine-norm/ld-norm-dev'

    # Obtain the training, validation and test dataset
    path_to_train_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_train_2'
    path_to_valid_input = r'/media/druv022/Data1/Masters/Thesis/Data/Converted_develop'
    path_to_test= r'/media/druv022/Data1/Masters/Thesis/Data/Converted_test'
    ctd_file = r'/media/druv022/Data1/Masters/Thesis/Data/CTD/CTD_diseases.csv'
    c2m_file = r'/media/druv022/Data1/Masters/Thesis/Data/C2M/C2M_mesh.txt'

    X_valid = BratInput(path_to_valid_input)
    X_valid = X_valid.transform()
    # X_valid = split_annotated_documents(X_valid)

    X_test = BratInput(path_to_test)
    X_test = X_test.transform()

    pred_list = []
    orig_list = []

    toD = Convert2D(ctd_file, c2m_file)

    for f_name in os.listdir(folder_name):
        with open(os.path.join(folder_name, f_name), 'r') as f:
            reader = csv.reader(f, delimiter='\t')

            ann_doc = None
            norms = None
        
            for i,row in enumerate(reader):
                if i == 0:
                    continue

                doc_name = row[0]
                start_idx = int(row[1])
                end_index = int(row[2])-1

                text = row[6]
                norm_id = row[8]

                if ann_doc is None:
                    for doc in X_valid:
                        if doc_name == doc.identifier:
                            ann_doc = doc
                            norms = copy.deepcopy(doc.normalizations)

                # checking only for the predicted mentions(-pipeline)
                if ann_doc is not None:
                    for annotation in ann_doc.annotations:
                        if text == annotation.text and (start_idx, end_index) == annotation.offset:
                            ann_identifier = annotation.identifier
                            break
                    
                    for norm in norms:
                        if norm.argument_id == ann_identifier:
                            pred_list.append(norm_id)
                            orig_term = norm.preferred_term.strip()
                            if '+' in orig_term:
                                orig_term = orig_term.split('+')
                            elif '|' in orig_term:
                                orig_term = orig_term.split('|')
                            else:
                                orig_term = [orig_term]

                            orig_term_list = []
                            for term in orig_term:
                                if 'D' not in term:
                                    t = toD.transform(term)
                                    if t is not None:
                                        orig_term_list.append(t)
                                    else:
                                        orig_term_list.append('UNK')
                                else:
                                    orig_term_list.append(term)

                            if norm_id in orig_term_list:
                                orig_list.append(norm_id)
                            else:
                                orig_list.append(orig_term_list[0])

                            norms.remove(norm)
                            break
                else:
                    print(f'ERROR! Document {ann_doc} is not found.')

    print(classification_report(orig_list, pred_list))
    print(accuracy_score(orig_list, pred_list))