# About Bert
The code and the pre-trained weights can be found [here](https://github.com/google-research/bert).
You can download this code and pre-trained model and then fine tune your task.

# About Dateset in text classification
We can use the code in run_classifier.py. So the dataset should be "label (TAB) text-content", like:

'''python
2	As someone who has worked with many museums, I was eager to visit this gallery on my most recent trip to Las Vegas. When I saw they would be showing infamous eggs of the House of Faberge from the Virginia Museum of Fine Arts (VMFA), I knew I had to go!Tucked away near the gelateria and the garden, the Gallery is pretty much hidden from view. It's what real estate agents would call "cozy" or "charming" - basically any euphemism for small.That being said, you can still see wonderful art at a gallery of any size, so why the two *s you ask? Let me tell you:* pricing for this, while relatively inexpensive for a Las Vegas attraction, is completely over the top. For the space and the amount of art you can fit in there, it is a bit much.* it's not kid friendly at all. Seriously, don't bring them.* the security is not trained properly for the show. When the curating and design teams collaborate for exhibitions, there is a definite flow. That means visitors should view the art in a certain sequence, whether it be by historical period or cultural significance (this is how audio guides are usually developed). When I arrived in the gallery I could not tell where to start, and security was certainly not helpful. I was told to "just look around" and "do whatever." At such a *fine* institution, I find the lack of knowledge and respect for the art appalling.
'''

# Change in run_classification.py
'''python
class YelpProcessor(DataProcessor):
  def __init__(self):
    self.labels = ['1', '2', '3', '4', '5']

  def get_labels(self):
    return self.labels

  def get_train_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.csv')), 'train')

  def get_dev_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, 'val.csv')), 'val')

  def get_test_examples(self, data_dir):
    return self._create_examples(self._read_tsv(os.path.join(data_dir, 'test.csv')), 'test')

  def _create_examples(self, lines, set_type):
    examples = []
    for i, line in enumerate(lines):
      guid = '%s-%s' % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    shuffle_indices = np.random.permutation(np.arange(len(examples)))
    examples = np.array(examples)
    examples = examples[shuffle_indices]
    return examples
'''

We can inherit the DataProcess class, and then implement the class method to get the train, dev and test data. And then add it to processors in main function and named it "mytask":

'''python
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mytask": YelpProcessor,
  }
'''

# About run the code
We can use the command:
'''python
python run_classifier.py \
 --task_name=mytask \
 --do_train=true \
 --do_eval=true \
 --data_dir=$DATA_DIR/ \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
 --max_seq_length=128 \
 --train_batch_size=32 \
 --learning_rate=2e-5 \
 --num_train_epochs=3.0 \
 --output_dir=mytask_output
'''
The task_name is defined in processors and the dataset is in $data_dir$. vocab_file, bert_config_file and init_checkpoint are contained in the pre-trained model you download, so the $BERT_BASE_DIR$ is the file contained weights.
