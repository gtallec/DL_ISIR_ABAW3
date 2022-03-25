import copy

class Condition:
    def __init__(self, **kwargs):
        pass

    def func(self, logs):
        return True

    def check(self, logs):
        return self.func(logs)
        
class ThresholdCondition(Condition):
    def __init__(self, threshold, argument, sign):
        super(ThresholdCondition, self).__init__()
        self.threshold = threshold
        self.argument = argument
        self.sign = sign

    def func(self, logs):
        return self.sign * logs[self.argument].iloc[0] <= self.sign * self.threshold

class AboveThresholdCondition(ThresholdCondition):
    def __init__(self, threshold, argument):
        super(AboveThresholdCondition, self).__init__(threshold=threshold,
                                                      argument=argument,
                                                      sign=-1)

class BelowThresholdCondition(ThresholdCondition):
    def __init__(self, threshold, argument):
        super(AboveThresholdCondition, self).__init__(threshold=threshold,
                                                      argument=argument,
                                                      sign=1)


class ConsecutiveCondition(Condition):
    def __init__(self, condition_arg, patience, steps):
        super(ConsecutiveCondition, self).__init__()
        self.condition = condition(condition_arg)
        self.patience = patience

        self.steps = steps
        self.current_steps = 0

        self.patience = patience
        self.current_patience = 0

    def func(self, logs):
        if (self.current_steps >= self.steps):
            return True

        is_verified = self.condition.func(logs)
        if is_verified:
            self.current_steps += 1
            return (self.current_steps >= self.steps)
        else:
            if self.current_patience < self.patience:
                self.current_patience += 1
            else:
                self.current_steps = 0
                self.current_patience = 0
            return False

class ConsecutiveConvergenceCondition(ConsecutiveCondition):
    def __init__(self, threshold, argument, patience, steps):
        super(ConsecutiveConvergenceCondition, self).__init__(dict({"type": 'convergence',
                                                                    "threshold": threshold,
                                                                    "argument": argument}),
                                                              patience=patience,
                                                              steps=steps)


class ConvergenceCondition(Condition):
    def __init__(self, threshold, argument):
        super(ConvergenceCondition, self).__init__()
        self.argument_tracker = None
        self.argument = argument
        self.threshold = threshold

    def func(self, logs):
        argument_current = logs[self.argument]
        if self.argument_tracker is None:
            self.argument_tracker = argument_current
            return False
        answer = abs(self.argument_tracker - argument_current) < self.threshold
        self.argument_tracker = argument_current
        return answer


class AggregatedCondition(Condition):
    def __init__(self, condition_args, aggregation_op):
        super(Condition, self).__init__()
        self.conditions = []
        self.aggregation_op = aggregation_op

        for condition_arg in condition_args:
            if 'formattable' in condition_arg:
                format_list = condition_arg.pop('formattable')
                for format_el in format_list:
                    cp_condition_arg = copy.deepcopy(condition_arg)
                    cp_condition_arg['argument'] = condition_arg['argument'].format(format_el)
                    self.conditions.append(condition(cp_condition_arg))
            else:
                self.conditions.append(condition(condition_arg))

    def func(self, logs):
        return self.aggregation_op([cond.func(logs) for cond in self.conditions])

class OrCondition(AggregatedCondition):
    def __init__(self, condition_args):
        super(OrCondition, self).__init__(condition_args, any)

class AndCondition(AggregatedCondition):
    def __init__(self, condition_args):
        super(AndCondition, self).__init__(condition_args, all)


SUPPORTED_CONDITIONS = {'above_th': AboveThresholdCondition,
                        'below_th': BelowThresholdCondition,
                        'convergence': ConvergenceCondition,
                        'consecutive_convergence': ConsecutiveConvergenceCondition,
                        'or': OrCondition,
                        'and': AndCondition}


def condition(condition_args):
    condition_type = condition_args.pop('type')
    return SUPPORTED_CONDITIONS[condition_type](**condition_args)


if __name__ == '__main__':
    trigger_condition = condition({'type': 'and',
                                   'condition_args': [{'type': 'convergence',
                                                       'argument': 'train_coucou_{}',
                                                       'formattable': [0, 1],
                                                       'threshold': 0.5}]})
    argument1 = {"train": {'coucou_0': 4,
                           'coucou_1': 4}}
    argument2 = {"train": {'coucou_0': 5,
                           'coucou_1': 5}}
    argument3 = {"train": {'coucou_0': 6,
                           'coucou_1': 6}}
    argument4 = {"train": {'coucou_0': 5.7,
                           'coucou_1': 5.7}}

    print(trigger_condition.check(epoch=0, logs=argument1))
    print(trigger_condition.check(epoch=1, logs=argument2))
    print(trigger_condition.check(epoch=2, logs=argument3))
    print(trigger_condition.check(epoch=3, logs=argument4))
