This is the implementation of the NER Project

======Data=========

We are using the OntoNote Release 5.0 to training our model.
However, due to the size of the data and the License of the data, we cannot upload the data to Blackboard
Here is a guidence to obtain the data and training our model.

1. Get OntoNotes Release 5.0 from LDC
Please visit https://catalog.ldc.upenn.edu/LDC2013T19 to get the lastest version of the data set.
In order to obtain the data, you need create an account and then click the Request Data at the bottom of the page.
Note: your account must be in one orgnization. If you are not in any orgniztion , you can create an orgnization account as Non-member.

2. CoNLL-2012 format
In order to Convert the data into CoNLL-2012 format, please visit http://conll.cemantix.org/2012/data.html and download the Training Data, Develoment Test Data Test Key as well as the Scripts files. you should run the scripts on a Linux system. This may take a while

When you get the data, please put your data in "data/conll-2012" folder. the whole path should looks like "data/conll-2012/v4/data/train/data...."

3. GloVe word representation:
You need download the Glove pre-trained word vector from http://nlp.stanford.edu/data/glove.840B.300d.zip, then, upzip it and put the glove.840B.300d.txt file in "data/text/" folder

The total size of these data are about 10GB

=====Run Script======
1. Please Run baseline.py to train and predict use HMM: python baseline.py
2. Please Run NER.py file to train and predict use LSTM+ : python NER.py
3. please Run entity_level_evaluation.py to get the precision, recall and f1 score of the test result of two models: python entity_level_evaluation.py
4. please Run prediction.py to extract the named entities from an article. It accept one parameter which should be the relative path of the article. By default, the article is in the text.txt file at the same folder

=====System Requirement=====

Due to the large size of our data, your computer should have at least 15GB free space.
For other package needed to run the scripts, please see "requirements.txt"


