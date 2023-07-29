import nengo
import numpy as np


def estimate_slice_len(list_len, slice):
    stop = slice.stop if slice.stop is not None else list_len
    stop = stop if stop >= 0 else list_len + stop

    start = slice.start if slice.start is not None else 0
    return min(stop, list_len) - start

                    
class Predictor(nengo.Network):
    def __init__(
            self,
            state_size,
            action_size,
            dynamics_size,
            static_predictor_n_neurons,
            static_predictor_function,
            adaptive_predictor_n_neurons,
            synapse_tau,
            trainable,
            adaptive_predictor_state_slice=slice(None, None),
            learning_rate=None,
            adaptive_predictor_seed=0,
            adaptive_predictor_weights=None,
            static_predictor_neuron_type=nengo.LIF(),
            adaptive_predictor_neuron_type=nengo.LIF(),
            pes_pre_synapse=nengo.Default,
            label=None,
            seed=None,
            add_to_container=None):
        super().__init__(label, seed, add_to_container)
        
        expected_input_size = estimate_slice_len(
            state_size, adaptive_predictor_state_slice)
        
        if trainable and learning_rate is None:
            raise ValueError("Learning rate must be specified for trainable predictor")
        if not trainable and adaptive_predictor_weights is None and adaptive_predictor_n_neurons > 0:
            raise ValueError("Weights must be specified for non-trainable predictor")
        
        with self:
            self.input_state = nengo.Node(size_in=state_size)
            self.input_action = nengo.Node(size_in=action_size)

            static_predictor = nengo.Ensemble(
                static_predictor_n_neurons,
                dimensions=action_size+state_size,
                radius=np.sqrt(action_size+state_size),
                neuron_type=static_predictor_neuron_type,
                label="static_predictor")
            
            nengo.Connection(self.input_action, static_predictor[:action_size], synapse=None)
            nengo.Connection(self.input_state, static_predictor[action_size:], synapse=None)

            if adaptive_predictor_n_neurons > 0:
                adaptive_predictor = nengo.Ensemble(
                    adaptive_predictor_n_neurons,
                    dimensions=action_size+expected_input_size,
                    radius=np.sqrt(action_size+expected_input_size),
                    neuron_type=adaptive_predictor_neuron_type,
                    seed=adaptive_predictor_seed,
                    label="adaptive_predictor")
            
                nengo.Connection(self.input_state[adaptive_predictor_state_slice], adaptive_predictor[action_size:], synapse=None)
                nengo.Connection(self.input_action, adaptive_predictor[:action_size], synapse=None)

            self.output = nengo.Node(size_in=dynamics_size, label="predicted_dynamics")

            nengo.Connection(static_predictor, self.output, function=static_predictor_function, synapse=synapse_tau)

            if adaptive_predictor_n_neurons > 0:
                if trainable:
                    self.dynamics_error = nengo.Node(size_in=dynamics_size)
                    self.adaptive_connection = nengo.Connection(
                        adaptive_predictor,
                        self.output,
                        function=lambda x: [0] * dynamics_size,
                        synapse=synapse_tau,
                        learning_rule_type=nengo.PES(learning_rate, pre_synapse=pes_pre_synapse))
                    
                    nengo.Connection(self.dynamics_error, self.adaptive_connection.learning_rule, synapse=None)
                else:
                    self.adaptive_connection = nengo.Connection(
                        adaptive_predictor.neurons,
                        self.output,
                        transform=adaptive_predictor_weights,
                        synapse=synapse_tau,
                        learning_rule_type=nengo.PES())