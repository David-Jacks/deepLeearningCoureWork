#this py file contains a generic class for the network

import numpy as np
import pandas as pd
from collections import defaultdict
from my_utils import act_softmax, act_relu, loss_fnc, relu_deriv

class network_model:
    def __init__(self, no_hid_layer, X_train, Y_train, nueron_num_arr, optimizer='sgd', l2_lambda=0.0, dropout_rate=0.5, use_batchnorm=True) -> None:
        self.hidden_layers = no_hid_layer
        self.input_size = X_train.shape[1]
        self.data = X_train
        self.label = Y_train
        self.lay_par = defaultdict(None)
        self.loss_by_epochs = []
        self.nue_nums = nueron_num_arr
        # optimizer support: 'sgd' (default) or 'adam'
        self.optimizer = optimizer
        self.opt_state = {}
        # L2 regularization strength (lambda)
        self.l2_lambda = l2_lambda
        # Dropout rate (probability to drop a neuron in hidden layers)
        self.dropout_rate = dropout_rate
        self.training = True  # flag to control dropout (True for training, False for eval)
        # BatchNorm momentum for running statistics
        self.bn_momentum = 0.9
        self.use_batchnorm = use_batchnorm

    def layerInitialising(self, neuron_number):
        # initialise all hidden layers, and also the output layer
        for i in range(self.hidden_layers + 1):
            # He initialization for ReLU activations
            fan_in = max(1, self.input_size)
            self.weights = np.random.randn(fan_in, neuron_number[i]) * np.sqrt(2.0 / fan_in)
            self.biases = np.zeros((1, neuron_number[i]))
            self.lay_par[f"lay${i+1}_w"] = self.weights
            self.lay_par[f"lay${i+1}_b"] = self.biases

            # BatchNorm params for hidden layers only (not output)
            if self.use_batchnorm and i < self.hidden_layers:
                self.lay_par[f"bn${i+1}_gamma"] = np.ones((1, neuron_number[i]))
                self.lay_par[f"bn${i+1}_beta"] = np.zeros((1, neuron_number[i]))
                # Running mean/var for inference
                self.lay_par[f"bn${i+1}_running_mean"] = np.zeros((1, neuron_number[i]))
                self.lay_par[f"bn${i+1}_running_var"] = np.ones((1, neuron_number[i]))

            # if using Adam, prepare optimizer state entries
            if self.optimizer == 'adam':
                if 'm' not in self.opt_state:
                    self.opt_state['m'] = {}
                    self.opt_state['v'] = {}
                    self.opt_state['t'] = 0
                    # default betas and eps; can be changed by directly editing opt_state
                    self.opt_state['beta1'] = 0.9
                    self.opt_state['beta2'] = 0.999
                    self.opt_state['eps'] = 1e-8
                self.opt_state['m'][f"lay${i+1}_w"] = np.zeros_like(self.weights)
                self.opt_state['v'][f"lay${i+1}_w"] = np.zeros_like(self.weights)
                self.opt_state['m'][f"lay${i+1}_b"] = np.zeros_like(self.biases)
                self.opt_state['v'][f"lay${i+1}_b"] = np.zeros_like(self.biases)
                if self.use_batchnorm and i < self.hidden_layers:
                    self.opt_state['m'][f"bn${i+1}_gamma"] = np.zeros((1, neuron_number[i]))
                    self.opt_state['v'][f"bn${i+1}_gamma"] = np.zeros((1, neuron_number[i]))
                    self.opt_state['m'][f"bn${i+1}_beta"] = np.zeros((1, neuron_number[i]))
                    self.opt_state['v'][f"bn${i+1}_beta"] = np.zeros((1, neuron_number[i]))

            self.input_size = neuron_number[i]

    def for_prop(self):
        x_data = self.data
        self.bn_cache = {}  # store batch norm cache for backprop
        self.dropout_masks = {}  # store dropout masks for each layer
        for i in range(self.hidden_layers + 1):
            z = np.dot(x_data, self.lay_par[f"lay${i+1}_w"]) + self.lay_par[f"lay${i+1}_b"]
            # Apply batch norm and dropout for hidden layers only
            if self.use_batchnorm and i < self.hidden_layers:
                gamma = self.lay_par[f"bn${i+1}_gamma"]
                beta = self.lay_par[f"bn${i+1}_beta"]
                eps = 1e-5
                if self.training:
                    mu = np.mean(z, axis=0, keepdims=True)
                    var = np.var(z, axis=0, keepdims=True)
                    # update running stats
                    running_mean = self.lay_par.get(f"bn${i+1}_running_mean")
                    running_var = self.lay_par.get(f"bn${i+1}_running_var")
                    if running_mean is None:
                        running_mean = np.zeros_like(mu)
                        running_var = np.ones_like(var)
                    self.lay_par[f"bn${i+1}_running_mean"] = self.bn_momentum * running_mean + (1 - self.bn_momentum) * mu
                    self.lay_par[f"bn${i+1}_running_var"] = self.bn_momentum * running_var + (1 - self.bn_momentum) * var
                else:
                    # use running statistics at inference time
                    mu = self.lay_par.get(f"bn${i+1}_running_mean")
                    var = self.lay_par.get(f"bn${i+1}_running_var")
                    # fallback if not initialized
                    if mu is None:
                        mu = np.mean(z, axis=0, keepdims=True)
                    if var is None:
                        var = np.var(z, axis=0, keepdims=True)
                z_norm = (z - mu) / np.sqrt(var + eps)
                z_bn = gamma * z_norm + beta
                # Dropout (only during training)
                if self.training and self.dropout_rate > 0:
                    mask = (np.random.rand(*z_bn.shape) > self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
                    z_bn = z_bn * mask
                    self.dropout_masks[i+1] = mask
                else:
                    self.dropout_masks[i+1] = np.ones_like(z_bn)
                a = self.layerActivation(i, z_bn)
                self.bn_cache[i+1] = (z, z_norm, mu, var, gamma, beta)
                self.lay_par[f"lay${i+1}_out"] = z
                self.lay_par[f"act_lay${i+1}_out"] = a
                x_data = a
            else:
                # output layer or no batch norm: no batch norm, no dropout
                a = self.layerActivation(i, z)
                self.lay_par[f"lay${i+1}_out"] = z
                self.lay_par[f"act_lay${i+1}_out"] = a
                x_data = a


    # function to activate all layer outputs
    def layerActivation(self, i, input):
        #bassically we are making sure that if the iteration count gets to the number of hidden layers(we will be at the output later at that time)
        # so we want to use the right activation function for the output layer, else use the hiden layer activation function
        if i == self.hidden_layers:
            activated_out = act_softmax(input)
        else:
            activated_out = act_relu(input)
        return activated_out


    # function to calculate loss
    def cal_loss(self):
        base_loss = loss_fnc(pred_res=self.lay_par[f"act_lay${self.hidden_layers+1}_out"], act_res=self.label)
        # add L2 regularization term (lambda / (2m)) * sum(W^2)
        if self.l2_lambda and self.l2_lambda > 0:
            m = self.data.shape[0]
            reg_sum = 0.0
            for i in range(self.hidden_layers + 1):
                W = self.lay_par[f"lay${i+1}_w"]
                reg_sum += np.sum(W * W)
            reg_loss = (self.l2_lambda / (2.0 * m)) * reg_sum
            return base_loss + reg_loss
        return base_loss

    def cal_accuracy(self):
        from my_utils import model_accuracy
        # Ensure we have up-to-date output activations (run forward in inference mode)
        old_training = self.training
        try:
            self.training = False
            self.for_prop()
            out = self.lay_par.get(f"act_lay${self.hidden_layers+1}_out")
        finally:
            self.training = old_training
        return model_accuracy(output_layer=out, label=self.label)
      
        # function to calculate gadient
    def back_prop(self):
        grads = {}
        m = self.data.shape[0]  # number of samples

        # --- Output layer gradient (softmax + cross-entropy) ---
        A_out = self.lay_par[f"act_lay${self.hidden_layers + 1}_out"]
        Y = self.label
        dZ = A_out.copy()
        dZ[range(m), Y] -= 1
        dZ /= m  # normalize

        # Loop backward through layers
        for i in reversed(range(1, self.hidden_layers + 2)):
            A_prev = self.data if i == 1 else self.lay_par[f"act_lay${i-1}_out"]
            W = self.lay_par[f"lay${i}_w"]

            # BatchNorm backward for hidden layers only (not output)
            if self.use_batchnorm and i <= self.hidden_layers:
                # Get batch norm cache
                z, z_norm, mu, var, gamma, beta = self.bn_cache[i]
                # Gradient wrt activation after batch norm
                da = dZ
                # Apply dropout mask to gradient (if used in forward pass)
                if hasattr(self, 'dropout_masks') and (i in self.dropout_masks):
                    da = da * self.dropout_masks[i]
                # Gradient wrt batch norm output (before activation)
                dz_bn = da * relu_deriv(z_norm * gamma + beta)
                # Gradients for gamma and beta
                grads[f"d_gamma{i}"] = np.sum(dz_bn * z_norm, axis=0, keepdims=True)
                grads[f"d_beta{i}"] = np.sum(dz_bn, axis=0, keepdims=True)
                # Backprop through batch norm
                N = dz_bn.shape[0]
                dz_norm = dz_bn * gamma
                dvar = np.sum(dz_norm * (z - mu) * -0.5 * (var + 1e-5) ** (-1.5), axis=0, keepdims=True)
                dmu = np.sum(dz_norm * -1 / np.sqrt(var + 1e-5), axis=0, keepdims=True) + dvar * np.mean(-2 * (z - mu), axis=0, keepdims=True)
                dz = dz_norm / np.sqrt(var + 1e-5) + dvar * 2 * (z - mu) / N + dmu / N
                # Now dz is the gradient wrt z (pre-batchnorm pre-activation)
                dZ = dz
            else:
                # No batchnorm on this layer (output or BN disabled) — dZ already corresponds to post-activation gradient
                pass

            # Compute gradients
            grads[f"dw{i}"] = np.dot(A_prev.T, dZ)
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)
            # add L2 regularization gradient: (lambda/m) * W
            if self.l2_lambda and self.l2_lambda > 0:
                grads[f"dw{i}"] += (self.l2_lambda / m) * W

            # Propagate to previous layer
            if i > 1:
                dA_prev = np.dot(dZ, W.T)
                if (i-1) <= self.hidden_layers and self.use_batchnorm:
                    # If previous layer used batch norm, pass gradients through (BN-backward will handle activation)
                    dZ = dA_prev  # will be handled in next loop iteration by BN backward
                else:
                    Z_prev = self.lay_par[f"lay${i-1}_out"]
                    dZ = dA_prev * relu_deriv(Z_prev)

        return grads


    # updating parameters
    def update_param(self, lr):
        gradients = self.back_prop()

        if self.optimizer == 'adam':
            # Adam update across parameters
            # increment timestep
            self.opt_state['t'] += 1
            t = self.opt_state['t']
            b1 = self.opt_state.get('beta1', 0.9)
            b2 = self.opt_state.get('beta2', 0.999)
            eps = self.opt_state.get('eps', 1e-8)

            for i in range(self.hidden_layers + 1):
                w_key = f"lay${i+1}_w"
                b_key = f"lay${i+1}_b"
                gw = gradients[f"dw{i+1}"]
                gb = gradients[f"db{i+1}"]

                # update first and second moments for weights
                m_w = self.opt_state['m'][w_key]
                v_w = self.opt_state['v'][w_key]
                m_w = b1 * m_w + (1 - b1) * gw
                v_w = b2 * v_w + (1 - b2) * (gw ** 2)
                m_w_hat = m_w / (1 - b1 ** t)
                v_w_hat = v_w / (1 - b2 ** t)
                self.lay_par[w_key] -= lr * (m_w_hat / (np.sqrt(v_w_hat) + eps))
                self.opt_state['m'][w_key] = m_w
                self.opt_state['v'][w_key] = v_w

                # update first and second moments for biases
                m_b = self.opt_state['m'][b_key]
                v_b = self.opt_state['v'][b_key]
                m_b = b1 * m_b + (1 - b1) * gb
                v_b = b2 * v_b + (1 - b2) * (gb ** 2)
                m_b_hat = m_b / (1 - b1 ** t)
                v_b_hat = v_b / (1 - b2 ** t)
                self.lay_par[b_key] -= lr * (m_b_hat / (np.sqrt(v_b_hat) + eps))
                self.opt_state['m'][b_key] = m_b
                self.opt_state['v'][b_key] = v_b

                # BatchNorm Adam update for gamma and beta (hidden layers only)
                if i < self.hidden_layers and self.use_batchnorm:
                    g_key = f"bn${i+1}_gamma"
                    be_key = f"bn${i+1}_beta"
                    dgamma = gradients.get(f"d_gamma{i+1}", np.zeros_like(self.lay_par[g_key]))
                    dbeta = gradients.get(f"d_beta{i+1}", np.zeros_like(self.lay_par[be_key]))
                    m_g = self.opt_state['m'][g_key]
                    v_g = self.opt_state['v'][g_key]
                    m_g = b1 * m_g + (1 - b1) * dgamma
                    v_g = b2 * v_g + (1 - b2) * (dgamma ** 2)
                    m_g_hat = m_g / (1 - b1 ** t)
                    v_g_hat = v_g / (1 - b2 ** t)
                    self.lay_par[g_key] -= lr * (m_g_hat / (np.sqrt(v_g_hat) + eps))
                    self.opt_state['m'][g_key] = m_g
                    self.opt_state['v'][g_key] = v_g

                    m_be = self.opt_state['m'][be_key]
                    v_be = self.opt_state['v'][be_key]
                    m_be = b1 * m_be + (1 - b1) * dbeta
                    v_be = b2 * v_be + (1 - b2) * (dbeta ** 2)
                    m_be_hat = m_be / (1 - b1 ** t)
                    v_be_hat = v_be / (1 - b2 ** t)
                    self.lay_par[be_key] -= lr * (m_be_hat / (np.sqrt(v_be_hat) + eps))
                    self.opt_state['m'][be_key] = m_be
                    self.opt_state['v'][be_key] = v_be
        else:
            for i in range(self.hidden_layers + 1):
                self.lay_par[f"lay${i+1}_w"] -= (lr * gradients[f"dw{i+1}"])
                self.lay_par[f"lay${i+1}_b"] -= (lr * gradients[f"db{i+1}"])
                # BatchNorm SGD update for gamma and beta (hidden layers only)
                if i < self.hidden_layers and self.use_batchnorm:
                    g_key = f"bn${i+1}_gamma"
                    be_key = f"bn${i+1}_beta"
                    dgamma = gradients.get(f"d_gamma{i+1}", np.zeros_like(self.lay_par[g_key]))
                    dbeta = gradients.get(f"d_beta{i+1}", np.zeros_like(self.lay_par[be_key]))
                    self.lay_par[g_key] -= lr * dgamma
                    self.lay_par[be_key] -= lr * dbeta

    # function to train
    def train_model(self, iterations, lr, val_data=None, val_label=None, patience=10, batch_size=64):
        """Train the network for `iterations` epochs using mini-batches.
        `batch_size` default is 64; `iterations` remains number of epochs.
        """
        self.layerInitialising(self.nue_nums)

        best_val_loss = float('inf')
        best_epoch = 0
        wait = 0
        best_params = None

        # keep references to full training data
        X_full = self.data
        Y_full = self.label
        m = X_full.shape[0]

        for epoch in range(iterations):
            # shuffle training data each epoch
            perm = np.random.permutation(m)
            X_shuf = X_full[perm]
            Y_shuf = Y_full[perm]

            # mini-batch updates
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                self.data = X_shuf[start:end]
                self.label = Y_shuf[start:end]
                self.training = True
                self.for_prop()
                self.update_param(lr)

            # evaluate training loss on full training set
            self.data = X_full
            self.label = Y_full
            self.training = True
            self.for_prop()
            loss = self.cal_loss()
            self.loss_by_epochs.append(loss)

            # Early stopping / validation
            if val_data is not None and val_label is not None:
                # Evaluate on validation set (disable training-only behavior like dropout)
                old_data = self.data
                old_label = self.label
                old_training = self.training
                try:
                    self.training = False
                    self.data = val_data
                    self.label = val_label
                    self.for_prop()
                    val_loss = self.cal_loss()
                finally:
                    self.data = old_data
                    self.label = old_label
                    self.training = old_training

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    wait = 0
                    # Save only the learnable parameters & running stats (avoid caching activations)
                    best_params = {k: v.copy() for k, v in self.lay_par.items() if (not k.startswith('act_lay')) and (not k.endswith('_out'))}
                else:
                    wait += 1
                print(f"Epoch {epoch+1}: train loss={loss:.4f}, val loss={val_loss:.4f}")
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                    if best_params is not None:
                        # restore saved parameters
                        for k, v in best_params.items():
                            self.lay_par[k] = v.copy()
                        # remove any cached activation outputs to avoid size mismatch
                        for k in list(self.lay_par.keys()):
                            if k.startswith('act_lay') or k.endswith('_out'):
                                try:
                                    del self.lay_par[k]
                                except KeyError:
                                    pass
                    break
            else:
                print(f"Epoch {epoch+1}: train loss={loss:.4f}")

        # restore references to full training set
        self.data = X_full
        self.label = Y_full
    
    #function to test the model
    def test_model(self, X_test):
        # Run forward pass in inference mode (disable dropout, use batch stats)
        old_data = self.data
        old_label = self.label
        old_training = self.training
        try:
            self.training = False
            self.data = X_test
            # label isn't needed for forward pass; set placeholder to avoid accidental reads
            self.label = np.zeros((X_test.shape[0],), dtype=int)
            self.for_prop()
            output = self.lay_par[f"act_lay${self.hidden_layers+1}_out"]
        finally:
            self.data = old_data
            self.label = old_label
            self.training = old_training
        return output