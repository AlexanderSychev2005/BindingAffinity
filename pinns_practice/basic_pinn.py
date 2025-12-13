import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class BasicPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),  # for RELU 2 derivative is always 0, so use Tanh
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


model = BasicPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()

    t_physics = torch.rand(100, 1).requires_grad_(
        True
    )  # requires_grad=True important for derivatives

    y_pred = model(t_physics)

    # y = e ^ (-2t)
    # dy/dt = -2y

    # Calculating derivative dy/dt
    # We use PyTorch auto-differentiation to find the rate of change of y_pred with respect to t_physics.
    dy_dt = torch.autograd.grad(
        outputs=y_pred,  # What we differentiate, y
        inputs=t_physics,  # What we differentiate with respect to, (time, t)
        grad_outputs=torch.ones_like(
            y_pred
        ),  # vector from 1, for 100 examples, calculates gradients independently
        create_graph=True,  # history of calculations, critical for PINNs
    )[0]

    # Physical Loss dy/dt + 2y = 0
    physical_loss = torch.mean((dy_dt + 2 * y_pred) ** 2)

    # Initial condition, t = 0 -> 1.0
    t_0 = torch.zeros(1, 1)
    y_0_pred = model(t_0)
    initial_condition_loss = torch.mean((y_0_pred - 1.0) ** 2)

    loss = physical_loss + initial_condition_loss
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

t_test = torch.linspace(0, 2, 100).view(-1, 1)
with torch.no_grad():
    y_test_pred = model(t_test)


y_exact = torch.exp(-2 * t_test)
plt.plot(
    t_test.numpy(), y_test_pred.numpy(), label="PINN model", color="red", linestyle="--"
)
plt.plot(t_test.numpy(), y_exact.numpy(), label="Exact solution (Math)", alpha=0.5)
plt.legend()
plt.title("Solving the differential equation!!")
plt.show()
