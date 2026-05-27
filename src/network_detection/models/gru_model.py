"""
GRU IDS Model - Lightweight Alternative
CyberGuard AI için hafif ve hızlı IDS modeli

Avantajlar:
    - LSTM'e göre daha az parametre
    - Daha hızlı eğitim
    - IoT ve edge sistemler için ideal
    - Gerçek zamanlı tespit için uygun

Mimari:
    Conv1D → GRU → Dense → Output
"""


import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model, layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class GRUIDSModel:
    """
    GRU tabanlı hafif IDS modeli

    IoT ve edge cihazlar için optimize edilmiş.
    LSTM'e göre %30-40 daha az parametre.
    """

    def __init__(
        self,
        input_shape: tuple[int, int] = (10, 78),
        num_classes: int = 15,
        gru_units: int = 100,
        conv_filters: int = 32,
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
        model_name: str = "GRU_IDS",
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow gerekli!")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.gru_units = gru_units
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.model_name = model_name

        self.model: Model | None = None

        print(f"⚡ {model_name} başlatılıyor...")
        print(f"   📊 GRU units: {gru_units}")
        print(f"   🔄 Bidirectional: {'✅' if bidirectional else '❌'}")

    def build(self) -> Model:
        """Model mimarisini oluştur"""
        inputs = layers.Input(shape=self.input_shape, name="input")

        # Conv1D - Feature extraction
        x = layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="conv1d",
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # GRU Layer
        gru_layer = layers.GRU(
            units=self.gru_units,
            dropout=self.dropout_rate,
            recurrent_dropout=0.1,
            return_sequences=False,
            name="gru",
        )

        if self.bidirectional:
            x = layers.Bidirectional(gru_layer, name="bigru")(x)
        else:
            x = gru_layer(x)

        # Dense layers
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(128, activation="relu", name="dense_2")(x)

        # Output
        outputs = layers.Dense(self.num_classes, activation="softmax", name="output")(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.model_name)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"✅ GRU modeli oluşturuldu! Parametre: {self.model.count_params():,}")
        return self.model

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=50,
        batch_size=64,
        patience=5,
    ) -> dict:
        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return {"accuracy": float(history.history["accuracy"][-1])}

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)


# Test
if __name__ == "__main__":
    print("🧪 GRU Model Test\n")

    X = np.random.rand(100, 10, 78).astype(np.float32)
    y = np.random.randint(0, 5, 100)

    model = GRUIDSModel(input_shape=(10, 78), num_classes=5, gru_units=64)
    model.build()
    print("✅ Test başarılı!")
