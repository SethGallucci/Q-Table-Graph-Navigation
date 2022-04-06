from Utils import Graph
from Utils import Agent


environment = Graph()

environment.add_node(0, 0, False)
environment.add_node(1, 0, False)
environment.add_node(2, 0, False)
environment.add_node(3, 0, False)
environment.add_node(4, 0, False)
environment.add_node(5, -100, True)
environment.add_node(6, 0, False)
environment.add_node(7, 100, True)


environment.add_edges([
	*[(0, n) for n in [   1,             6,  ]],
	*[(1, n) for n in [0,       3,       6,  ]],
	*[(2, n) for n in [0,          4, 5, 6,  ]],
	*[(3, n) for n in [0,          4,        ]],
	*[(4, n) for n in [   1, 2, 3,    5,    7]],
	*[(5, n) for n in [                      ]],
	*[(6, n) for n in [   1, 2,       5,     ]],
	*[(7, n) for n in [                      ]]
])

environment.set_entry_node(0)



training_agent = Agent(environment)

for trial in range(int(5e4)):
	training_info = environment.enter_graph()
	while not training_info["is_terminal_state"]:
		training_info = training_agent.act(is_training=True)

learned_Q_table = training_agent.get_Q_table()



testing_agent = Agent(environment, Q_table=learned_Q_table)

testing_agent_path = []
testing_info = environment.enter_graph()
testing_agent_path.append(testing_info["node_label"])
while not testing_info["is_terminal_state"]:
	testing_info = training_agent.act(is_training=False)
	testing_agent_path.append(testing_info["node_label"])

print()
print("Learned Path:")
print(testing_agent_path)
print()
print("Learned Q Table:")
testing_agent.print_Q_table(decimal_places=2)

environment.visualize()