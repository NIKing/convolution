class Module():
    def __init__(self):
        self.layers = {}

    def __call__(self, *arguments, **params):
        return self.forward(*arguments, **params)

    def forward(self, *arguments, **params):
         pass
