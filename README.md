# TfRecordPytorch
<pre>
Usage:
from tfrecord_pytorch import TFRecordPytorch
file_name = "train.tfrecord"
col_mapping={
"input_ids":tf.io.VarLenFeature(tf.int64),
"label_ids":tf.io.VarLenFeature(tf.int64)
}
*Note pytorch Iterable Dataset doesnt allow shuffle in Data Loader
dataset = TFRecordPytorch(file_name,col_mapping,shuffle=True,buffer_size=10000)
loader = torch.utils.DataLoader(dataset,batch_size=4,collate_fn=pad_and_sort)
iterator = iter(loader)
print(next(iterator))

</pre>
