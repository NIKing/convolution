class Module():
    def __init__(self):
        self.layers = {}

    def __call__(self, *arguments, **params):
        self.forword(*arguments, **params)

    def forward(self, *arguments, **params):
        pass
