# This class was provided by pythonprogramming.net (Harrison Kinsley),
# https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/

import tensorflow as tf
from keras.callbacks import TensorBoard

#Tensorboard wants to create a log file for each .fit, so this customized class prevents it from doing so
class ModifiedTensorBoard(TensorBoard):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

	#Stops default log writer creation by overwriting function
	def set_model(self, model):
		pass

	#Overided, saves logs with our step number
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	#Overrided
	def on_batch_end(self, batch, logs=None):
		pass

	#Overrided
	def on_train_end(self, _):
		pass

	# Custom method for saving own metrics
	def update_stats(self, **stats):
		self._write_logs(stats, self.step)
