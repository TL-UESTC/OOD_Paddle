import paddle

__all__ = [
    'Constraint',
    'positive',
    'real'
]

class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.

    Attributes:
        is_discrete (bool): Whether constrained space is discrete.
            Defaults to False.
        event_dim (int): Number of rightmost dimensions that together define
            an event. The :meth:`check` method will remove this many dimensions
            when computing validity.
    """
    is_descrete = False # Default to continuous
    event_dim = 0 # Default to univariate

    def check(self, value):
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return self.__class__.__name__[1:] + '()'


class _GreaterThan(Constraint):
    """
    Constraint to a real half line `(lower_bound, inf]`.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()
    
    def check(self, value):
        return self.lower_bound < value
    
    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={})'.format(self.lower_bound)
        return fmt_string
    

class _Real(Constraint):
    """
    Trivially constraint to the extended real line `[-inf, inf]`.
    """
    def check(self, value):
        return value == value # False for NANs


# Public interface
positive = _GreaterThan(0.)
real = _Real()