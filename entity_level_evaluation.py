import os

def evaluation(path):
    file = open(path,'r',encoding = "utf-8")
    data = file.readlines()
    file.close()

    correct = 0 # the number of correct prediction
    partial = 0 # the number of prediction which has wrong boundary
    missing = 0 # the number of prediction which missed a named entity
    hypothe = 0 # the number of prediction which hypothesized an entity
    incorrect = 0 # the number of prediction which has the correct bounday but wrong type

    #flags
    gound_length = 0
    predict_length = 0
    label_match = True

    '''
    calculate the values above
    '''
    for line in data:
        line = line.replace("\n","")
        if line:
            result = line.strip().split()
            if result[1].startswith("B") and result[2].startswith("B"):

                if gound_length and predict_length and gound_length == predict_length and label_match:
                    correct += 1
                elif gound_length and predict_length and gound_length != predict_length:
                    partial += 1
                elif gound_length and predict_length and  gound_length == predict_length and not label_match:
                    incorrect += 1
                elif gound_length and not predict_length:
                    missing += 1
                elif not gound_length and predict_length:
                    hypothe += 1

                gound_length = 1
                predict_length = 1
                label_match = (result[1] == result[2])


            elif result[1].startswith("B") and result[2].startswith("I"):
                if gound_length and not predict_length:
                    missing += 1

                gound_length = 1
                if predict_length:
                    predict_length += 1
                label_match = (result[1].strip().split('-')[1] == result[2].strip().split('-')[1])

            elif result[1].startswith("B") and result[2].startswith("O"):
                if gound_length and predict_length and gound_length == predict_length and label_match:
                    correct += 1
                elif gound_length and predict_length and gound_length != predict_length:
                    partial += 1
                elif gound_length and predict_length and  gound_length == predict_length and not label_match:
                    incorrect += 1
                elif gound_length and not predict_length:
                    missing += 1
                elif not gound_length and predict_length:
                    hypothe += 1

                gound_length = 1
                predict_length = 0
                label_match = False

            elif result[1].startswith("I") and result[2].startswith("B"):
                if gound_length:
                    gound_length += 1
                predict_length = 1
                label_match = (result[1].strip().split('-')[1] == result[2].strip().split('-')[1])

            elif result[1].startswith("I") and result[2].startswith("I"):
                if gound_length:
                    gound_length += 1
                if predict_length:
                    predict_length += 1
                label_match = (result[1].strip().split('-')[1] == result[2].strip().split('-')[1])

            elif result[1].startswith("I") and result[2].startswith("O"):
                if gound_length and predict_length:
                    partial += 1

                if gound_length:
                    gound_length += 1
                predict_length =0
                label_match = False

            elif result[1].startswith("O") and result[2].startswith("B"):
                if gound_length and predict_length and gound_length == predict_length and label_match:
                    correct += 1
                elif gound_length and predict_length and gound_length != predict_length:
                    partial += 1
                elif gound_length and predict_length and  gound_length == predict_length and not label_match:
                    incorrect += 1
                elif gound_length and not predict_length:
                    missing += 1
                elif not gound_length and predict_length:
                    hypothe += 1

                gound_length = 0
                predict_length = 1
                label_match = False

            elif result[1].startswith("O") and result[2].startswith("I"):
                gound_length = 0
                if predict_length:
                    predict_length += 1
                label_match = False
            elif result[1].startswith("O") and result[2].startswith("O"):
                if gound_length and predict_length and gound_length == predict_length and label_match:
                    correct += 1
                elif gound_length and predict_length and gound_length != predict_length:
                    partial += 1
                elif gound_length and predict_length and  gound_length == predict_length and not label_match:
                    incorrect += 1
                elif gound_length and not predict_length:
                    missing += 1
                elif not gound_length and predict_length:
                    hypothe += 1

                predict_length = 0
                gound_length = 0
                label_match = True

    if correct == 0:
        return 0,0,0
    else:
        possible = correct + incorrect + partial + missing
        actual = correct + incorrect + partial + hypothe
        precision = correct / actual
        recall = correct/possible

        F1 = 2 * precision * recall / (precision + recall)

        return precision, recall, F1

if __name__ == "__main__":
    path = os.getcwd()
    path_test = path + "\\results\\test.result.txt"
    path_baseline = path + "\\results\\baseline.result.txt"

    baseline_precision, baseline_recall, baseline_f1 = evaluation(path_baseline)
    test_presion,test_recall,test_f1 = evaluation(path_test)

    print("By using the HMM algorithm as baseline, the precision is {:.00%}, the recall is {:.00%}, the F-1 score is {:.00%}".format(baseline_precision, baseline_recall, baseline_f1))
    print("By using LSTM+CRF,  the precision is {:.00%}, the recall is {:.00%}, the F-1 score is {:.00%}".format(test_presion,test_recall,test_f1))