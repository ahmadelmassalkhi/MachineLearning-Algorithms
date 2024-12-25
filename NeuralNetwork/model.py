import numpy as np
import lib
import lib.Loss
import lib.Layers
import lib.Optimizers


class model:
    def __init__(self, *layers:lib.Layers.Layer):
        self.layers = layers

    def compile(self, optimizer:str, loss:str):
        self.optimizer = lib.Optimizers.Optimizer.create(optimizer)
        self.loss = lib.Loss.Loss.create(loss)


    def predict(self, x):
        # forward pass
        output = x
        for layer in self.layers:
            if isinstance(layer, lib.Layers.Dropout): layer.set_training_mode(False)
            output = layer.forward(output)
        
        # return prediction
        return output
    
    
    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        loss = self.loss.forward(predictions, y_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
        return loss, accuracy


    def _fit_batch(self, x_train, y_train):
        # forward pass
        output = x_train
        for layer in self.layers:
            if isinstance(layer, lib.Layers.Dropout): layer.set_training_mode(True)
            output = layer.forward(output)

        # evaluate model prediction
        loss = self.loss.forward(output, y_train)
        accuracy = np.mean(np.argmax(output, axis=1) == y_train)
        
        # backward pass
        dLoss = self.loss.backward()
        self.optimizer.pre_update_params()
        for layer in self.layers[::-1]:
            dLoss = layer.backward(dLoss)
            if isinstance(layer, lib.Layers.Dense): self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

        # return result
        return loss, accuracy


    def fit(self, x_train, y_train, batch_size=32, epochs=5):
        total_loss, total_acc = 0, 0
        nbOfBatches = len(x_train) // batch_size
        
        for epoch in range(epochs):
            epoch_loss, epoch_acc = 0, 0
            
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                loss, acc = self._fit_batch(x_batch, y_batch)
                epoch_loss += loss
                epoch_acc += acc
            
            # Average loss and accuracy for the epoch
            epoch_loss /= nbOfBatches
            epoch_acc /= nbOfBatches
            
            # Update total loss and accuracy
            total_loss += epoch_loss
            total_acc += epoch_acc
            
            # Print progress after each epoch
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Return average loss and accuracy over all epochs
        return total_loss / epochs, total_acc / epochs



