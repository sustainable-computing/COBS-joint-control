
class ActionCreator:
    def __init__(self, component_type, control_type, actuators=None):
        """ Create actions that will be passed to the EventQueue
        """
        self.priority = 0
        self.component_type = component_type
        self.control_type = control_type
        if actuators is not None:
            self.set_actuators(actuators)

    def set_actuators(self, actuators):
        if type(actuators) == list:
            self.actuators = actuators
        else:
            self.actuators = [actuators]

    def __call__(self, values, obs, note=None):
        actions = []
        if len(values) == 0:
            return []
        if len(values) == 1:
            for actuator_key in self.actuators:
                actions.append({
                    "priority": self.priority,
                    "component_type": self.component_type,
                    "control_type": self.control_type,
                    "actuator_key": actuator_key,
                    "note": note,
                    "value": values[0],
                    "start_time": obs['timestep'] + 1
                })
        else:
            # print(self.actuators)
            # print(values)
            assert len(values) == len(self.actuators)
            for actuator_key, value in zip(self.actuators, values):
                actions.append({
                    "priority": self.priority,
                    "component_type": self.component_type,
                    "control_type": self.control_type,
                    "actuator_key": actuator_key,
                    "note": note,
                    "value": value,
                    "start_time": obs['timestep'] + 1
                })

        return actions
