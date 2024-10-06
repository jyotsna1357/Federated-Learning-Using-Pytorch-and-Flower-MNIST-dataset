class NoDefense:
    def apply(self, data, labels):
        return data, labels

class FlippedLabelsDefense:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def apply(self, data, labels):
        # Flip labels (example: flip between two classes)
        flipped_labels = (labels + 1) % self.num_classes
        return data, flipped_labels
