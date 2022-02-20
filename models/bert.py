from transformers import DistilBertForSequenceClassification, DistilBertModel

class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


def initialize_bert_based_model(net, num_classes):

	if net == 'distilbert-base-uncased':
		model = DistilBertClassifier.from_pretrained(
			net,
			num_labels=num_classes)
	else:
		raise ValueError(f'Model: {net} not recognized.')
	return model