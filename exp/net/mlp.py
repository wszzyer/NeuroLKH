from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        self.module_list = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        for _ in range(L - 2):
            self.module_list.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU()
            ])
        self.V = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        Ux = x
        for U_i in self.module_list:
            Ux = U_i(Ux)
        y = self.V(Ux)
        return y
