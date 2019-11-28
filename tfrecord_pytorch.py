import torch
import tensorflow as tf
class TFRecordPytorch(torch.utils.data.IterableDataset):
	def __init__(self,tfrecord_path,col_mapping,shuffle=False,buffer_size=1000):
		super(MyIterableDataset).__init__()
		def parse_example(example_proto):
			return tf.io.parse_single_example(example_proto,col_mapping)
		dataset = tf.compat.v2.data.TfRecordDataset(tfrecord_path)
		if shuffle:
			dataset = dataset.shuffle(buffer_size)
		dataset = dataset.map(parse_example)
		dataset = dataset.map(lambda d: \
				{ key:tf.sparse_to_dense(d[key]) for key in list(col_mapping.keys())}
			)
		self.iterator = iter(dataset)
	def __iter__(self):
		def generator():
			for example in self.iter:
				yield {key:value.numpy() \
						for key,value in example.items()}
		return generator()




