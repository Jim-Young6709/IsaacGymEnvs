from .base import FeatureBase


class FeatureRegistry:
    def __init__(self):
        self.features = []

    def register(self, feature: FeatureBase):
        self.features.append(feature)

    def init_all(self, ui, controller):
        for feat in self.features:
            feat.init(ui, controller, registry=self)

    def widgets(self):
        return [feat.widget() for feat in self.features if feat.widget() is not None]

    def notify_entry_loaded(self, entry):
        for feat in self.features:
            feat.on_entry_loaded(entry)

    def notify_transform_changed(self, state):
        for feat in self.features:
            feat.on_transform_changed(state)
