import networkx as nx
from matplotlib import pyplot
import numpy as np


class Graph:


	def __init__(self):
		self._graph = nx.MultiDiGraph()
		self._entry_node = None
		self._current_node = self._entry_node


	def add_node(self, node_label, reward=0, is_terminal_state=False):
		self._graph.add_node(node_label, reward=reward, is_terminal_state=is_terminal_state)


	def get_node(self, node_label):
		return self._graph.nodes[node_label]


	def add_edges(self, edge_tuples):
		if type(edge_tuples) != list:
			edge_tuples = [edge_tuples]
		self._graph.add_edges_from(edge_tuples)


	def get_neighbors(self, node_label):
		return [*nx.neighbors(self._graph, node_label)]


	def get_current_neighbors(self):
		return self.get_neighbors(self.get_current_node())


	def set_entry_node(self, node_label):
		self._entry_node = node_label


	def get_current_node(self):
		return self._current_node


	def _set_current_node(self, node_label):
		self._current_node = node_label


	def enter_graph(self):
		self._current_node = self._entry_node
		return {"node_label": self.get_current_node(), **self.get_node(self._current_node)}


	def move_to_node(self, node_label):
		if node_label in self.get_neighbors(self.get_current_node()):
			self._set_current_node(node_label)
			return {"node_label": node_label, **self.get_node(node_label)}
		else:
			raise Exception(
				"Node with label \"{}\" has no edge leading to node with label \"{}\""
				.format(node_label, self._current_node)
			)


	def size(self):
		return self._graph.number_of_nodes()


	def visualize(self):
		nx.draw_shell(self._graph, with_labels=True)
		pyplot.show()



class Agent:


	def __init__(self, environment_graph, Q_table=None, learning_rate=0.05, epsilon_decay=0.999, reward_discount=0.90):
		self._environment_graph = environment_graph
		self._learning_rate = learning_rate
		self._epsilon = 1
		self._epsilon_decay = epsilon_decay
		self._reward_discount = reward_discount

		self._Q_table = Q_table if Q_table is not None else np.zeros((environment_graph.size(), environment_graph.size()), dtype=np.float)


	def get_Q_table(self):
		return self._Q_table.copy()


	def print_Q_table(self, decimal_places=4):
		print(np.round(self._Q_table, decimal_places))


	def act(self, is_training=True):

		# Choose the next node/action and update epsilon if necessary
		if is_training and np.random.rand() < self._epsilon:
			next_node_label = np.random.choice(self._environment_graph.get_current_neighbors(), 1).item()
			self._epsilon *= self._epsilon_decay
		else:
			# Determine the label of node that yields the highest Q value
			neighbor_choices = np.full_like(
				self._Q_table[self._environment_graph.get_current_node()],
				-np.inf,
				dtype=np.float
			)
			indices = self._environment_graph.get_current_neighbors()
			np.put(
				neighbor_choices,
				indices,
				np.take(
					self._Q_table[self._environment_graph.get_current_node()],
					indices
				)
			)
			next_node_label = np.argmax(neighbor_choices)

		# Move to the chosen node
		previous_node_label = self._environment_graph.get_current_node()
		current_node = self._environment_graph.move_to_node(next_node_label)

		if is_training:
			# Update the Q table
			self._Q_table[previous_node_label, current_node["node_label"]] = \
				(1 - self._learning_rate) * self._Q_table[previous_node_label, current_node["node_label"]] \
				+ self._learning_rate * (current_node["reward"] + self._reward_discount * np.max(
					self._Q_table[current_node["node_label"], :]))

		# Return useful information
		return current_node