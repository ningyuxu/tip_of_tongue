import abc


class Model(abc.ABC):
    """
    Represents a machine learning model for decoding semantic vectors from
    brain data. This abstract base class specifies the interface that
    deriving classes should implement.
    """

    @abc.abstractmethod
    def train(self, train_examples, train_targets):
        """
        Train the model on a specific set of training examples (brain images)
        their associated targets (vectors)

        Parameters:
            train_examples: array of shape (num_examples, num_voxels)
            train_targets: array of shape (num_examples, num_dimensions)
        """
        print(f"Examples: {train_examples.shape}")
        print(f"Targets: {train_targets.shape}")

    @abc.abstractmethod
    def test(self, test_examples):
        """
        Test the model on a specific set of testing examples

        Parameters:
            test_examples: array of shape (num_examples, num_voxels)

        Returns: array of shape (num_examples, num_dimensions)
        """
        pass

    @property
    @abc.abstractmethod
    def model_type(self):
        """
        Return the type of the model (e.g. 'pereira18' or 'MLP')
        :return: string
        """
        pass

    @property
    @abc.abstractmethod
    def params(self):
        """
        Return an object representing the models parameters
        :return: object
        """
        pass
