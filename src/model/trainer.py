from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, batch_size, epochs, patience, save_path):
        y_train_cat = to_categorical(y_train, num_classes=3)
        y_val_cat = to_categorical(y_val, num_classes=3)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        y_test_cat = to_categorical(y_test, num_classes=3)
        results = self.model.evaluate(X_test, y_test_cat, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions.argmax(axis=1)
