from abc import ABC, abstractmethod


class FeatureBase(ABC):
    name = "base"
    title = "Base"

    def init(self, ui, controller, registry=None):
        self.ui = ui
        self.controller = controller
        self.registry = registry

    def widget(self):
        return None

    def on_entry_loaded(self, entry):
        pass

    def on_transform_changed(self, state):
        pass
